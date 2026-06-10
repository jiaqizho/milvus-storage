// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus-storage/format/column_group_reader.h"

#include <map>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <utility>

#include <arrow/array/util.h>
#include <arrow/chunked_array.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/status.h>
#include <arrow/result.h>
#include <fmt/format.h>

#include <folly/executors/IOThreadPoolExecutor.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY
#include "milvus-storage/common/fiu_local.h"

namespace milvus_storage::api {

using milvus_storage::RowGroupInfo;
using ChunkRBMapResult = arrow::Result<std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>>>;

class ColumnGroupReaderImpl : public ColumnGroupReader {
  public:
  ColumnGroupReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                        const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                        const milvus_storage::api::Properties& properties,
                        const std::vector<std::string>& needed_columns,
                        const std::function<std::string(const std::string&)>& key_retriever,
                        const std::string& predicate = "");

  ~ColumnGroupReaderImpl() override = default;

  [[nodiscard]] arrow::Status open() override;
  [[nodiscard]] folly::SemiFuture<arrow::Status> open_async();
  [[nodiscard]] size_t total_number_of_chunks() const override;
  [[nodiscard]] size_t total_rows() const override;
  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) override;

  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, size_t parallelism = 1) override;

  [[nodiscard]] arrow::Result<uint64_t> get_chunk_size(int64_t chunk_index) override;
  [[nodiscard]] arrow::Result<uint64_t> get_chunk_rows(int64_t chunk_index) override;

  [[nodiscard]] std::vector<ChunkTask> get_natural_tasks(const std::vector<int64_t>& chunk_indices) override;
  [[nodiscard]] const ChunkInfo& get_chunk_info(int64_t chunk_index) const override;
  [[nodiscard]] folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>> get_chunks_async(
      const ChunkTask& task) override;

  [[nodiscard]] std::shared_ptr<arrow::Schema> get_schema() const override;

  private:
  struct OpenedFile {
    size_t file_index;
    ColumnGroupFile file;
    std::shared_ptr<FormatReader> reader;
    std::vector<RowGroupInfo> row_group_infos;
  };

  [[nodiscard]] arrow::Status append_file_metadata(size_t file_idx,
                                                   const ColumnGroupFile& cg_file,
                                                   const std::vector<RowGroupInfo>& row_group_in_file,
                                                   std::shared_ptr<FormatReader> format_reader,
                                                   size_t& rows_in_all_files);

  ChunkRBMapResult read_chunks_from_files(const std::vector<int64_t>& task_indices);

  protected:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;
  std::string predicate_;

  // will be initialized after call open()
  std::vector<ChunkInfo> chunk_infos_;
  std::vector<std::vector<RowGroupInfo>> row_group_infos_;
  size_t total_rows_;

  std::vector<std::shared_ptr<FormatReader>> format_readers_;
};  // ColumnGroupReaderImpl

arrow::Result<std::unique_ptr<ColumnGroupReader>> ColumnGroupReader::create(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const std::vector<std::string>& needed_columns,
    const milvus_storage::api::Properties& properties,
    const std::function<std::string(const std::string&)>& key_retriever,
    const std::string& predicate) {
  std::unique_ptr<ColumnGroupReader> reader = nullptr;
  if (!column_group) {
    return arrow::Status::Invalid("Column group cannot be null");
  }

  // Generate the output schema with only the needed columns
  std::shared_ptr<arrow::Schema> out_schema;
  std::vector<std::string> filtered_columns;
  for (const auto& col_name : needed_columns) {
    if (std::find(column_group->columns.begin(), column_group->columns.end(), col_name) !=
        column_group->columns.end()) {
      filtered_columns.emplace_back(col_name);
    }
  }

  if (schema) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& col_name : filtered_columns) {
      auto field = schema->GetFieldByName(col_name);
      if (!field) {
        return arrow::Status::Invalid(
            "ColumnGroupReader: column '" + col_name +
            "' found in column_group but not in schema. Schema fields: " + schema->ToString());
      }
      fields.emplace_back(field);
    }
    out_schema = std::make_shared<arrow::Schema>(fields);
  }
  reader = std::make_unique<milvus_storage::api::ColumnGroupReaderImpl>(out_schema, column_group, properties,
                                                                        filtered_columns, key_retriever, predicate);
  ARROW_RETURN_NOT_OK(reader->open());
  return std::move(reader);
}

folly::SemiFuture<arrow::Result<std::unique_ptr<ColumnGroupReader>>> ColumnGroupReader::create_async(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const std::vector<std::string>& needed_columns,
    const milvus_storage::api::Properties& properties,
    const std::function<std::string(const std::string&)>& key_retriever,
    const std::string& predicate) {
  if (!column_group) {
    return folly::makeSemiFuture(
        arrow::Result<std::unique_ptr<ColumnGroupReader>>(arrow::Status::Invalid("Column group cannot be null")));
  }

  std::shared_ptr<arrow::Schema> out_schema;
  std::vector<std::string> filtered_columns;
  for (const auto& col_name : needed_columns) {
    if (std::find(column_group->columns.begin(), column_group->columns.end(), col_name) !=
        column_group->columns.end()) {
      filtered_columns.emplace_back(col_name);
    }
  }

  if (schema) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& col_name : filtered_columns) {
      auto field = schema->GetFieldByName(col_name);
      if (!field) {
        return folly::makeSemiFuture(arrow::Result<std::unique_ptr<ColumnGroupReader>>(
            arrow::Status::Invalid("ColumnGroupReader: column '" + col_name +
                                   "' found in column_group but not in schema. Schema fields: " + schema->ToString())));
      }
      fields.emplace_back(field);
    }
    out_schema = std::make_shared<arrow::Schema>(fields);
  }

  auto reader = std::make_unique<milvus_storage::api::ColumnGroupReaderImpl>(
      out_schema, column_group, properties, filtered_columns, key_retriever, predicate);
  auto* reader_ptr = reader.get();
  return reader_ptr->open_async().deferValue(
      [reader = std::move(reader)](arrow::Status status) mutable -> arrow::Result<std::unique_ptr<ColumnGroupReader>> {
        ARROW_RETURN_NOT_OK(status);
        return std::move(reader);
      });
}

std::string ChunkInfo::ToString() const {
  std::stringstream ss;
  ss << "ChunkInfo{"
     << "file_index=" << file_index << ", row_offset_in_row_group=" << row_offset_in_row_group
     << ", row_offset_in_file=" << row_offset_in_file << ", number_of_rows=" << number_of_rows
     << ", row_group_index_in_file=" << row_group_index_in_file << ", global_row_end=" << global_row_end
     << ", avg_memory_size=" << avg_memory_size << "}";
  return ss.str();
}

ColumnGroupReaderImpl::ColumnGroupReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                                             const std::shared_ptr<api::ColumnGroup>& column_group,
                                             const api::Properties& properties,
                                             const std::vector<std::string>& needed_columns,
                                             const std::function<std::string(const std::string&)>& key_retriever,
                                             const std::string& predicate)
    : schema_(schema),
      column_group_(column_group),
      properties_(properties),
      needed_columns_(needed_columns),
      key_retriever_(key_retriever),
      predicate_(predicate) {}

arrow::Status ColumnGroupReaderImpl::append_file_metadata(size_t file_idx,
                                                          const ColumnGroupFile& cg_file,
                                                          const std::vector<RowGroupInfo>& row_group_in_file,
                                                          std::shared_ptr<FormatReader> format_reader,
                                                          size_t& rows_in_all_files) {
  if (row_group_in_file.empty()) {
    return arrow::Status::OK();
  }

  size_t rows_in_file = 0;
  if ((cg_file.start_index != 0 || cg_file.end_index != row_group_in_file.back().end_offset)) {
    const auto& start_index = cg_file.start_index;
    const auto& end_index = cg_file.end_index;

    assert(start_index >= 0 && end_index > 0 && start_index < end_index);

    for (size_t j = 0; j < row_group_in_file.size(); ++j) {
      size_t rg_start = row_group_in_file[j].start_offset;
      size_t rg_end = row_group_in_file[j].end_offset;

      size_t overlap_start = std::max(static_cast<size_t>(start_index), rg_start);
      size_t overlap_end = std::min(static_cast<size_t>(end_index), rg_end);

      if (overlap_start < overlap_end) {
        rows_in_file += (overlap_end - overlap_start);
        chunk_infos_.emplace_back(ChunkInfo{
            .file_index = file_idx,
            .row_offset_in_row_group = overlap_start - rg_start,
            .row_offset_in_file = overlap_start,
            .number_of_rows = overlap_end - overlap_start,
            .row_group_index_in_file = j,
            .global_row_end = rows_in_all_files + rows_in_file,
            .avg_memory_size = row_group_in_file[j].memory_size * (overlap_end - overlap_start) / (rg_end - rg_start),
        });
      }
    }
  } else {
    for (size_t j = 0; j < row_group_in_file.size(); ++j) {
      rows_in_file += (row_group_in_file[j].end_offset - row_group_in_file[j].start_offset);
      chunk_infos_.emplace_back(ChunkInfo{
          .file_index = file_idx,
          .row_offset_in_row_group = 0,
          .row_offset_in_file = row_group_in_file[j].start_offset,
          .number_of_rows = row_group_in_file[j].end_offset - row_group_in_file[j].start_offset,
          .row_group_index_in_file = j,
          .global_row_end = rows_in_all_files + rows_in_file,
          .avg_memory_size = row_group_in_file[j].memory_size,
      });
    }
  }

  row_group_infos_.emplace_back(row_group_in_file);
  format_readers_.emplace_back(std::move(format_reader));
  rows_in_all_files += rows_in_file;
  return arrow::Status::OK();
}

arrow::Status ColumnGroupReaderImpl::open() {
  const auto& cg_files = column_group_->files;

  // init chunk infos
  size_t rows_in_all_files = 0;
  for (size_t file_idx = 0; file_idx < cg_files.size(); ++file_idx) {
    auto& cg_file = cg_files[file_idx];

    if (cg_file.start_index < 0 || cg_file.end_index < 0 || cg_file.start_index >= cg_file.end_index) {
      return arrow::Status::Invalid(
          fmt::format("Invalid start/end index in [file_index={}, path={}]", file_idx, cg_file.path));
    }

    ARROW_ASSIGN_OR_RAISE(auto format_reader, FormatReader::create(schema_, column_group_->format, cg_file, properties_,
                                                                   needed_columns_, key_retriever_));
    if (!predicate_.empty()) {
      try {
        format_reader->set_predicate(predicate_);
      } catch (const std::exception& e) {
        return arrow::Status::Invalid(fmt::format("Failed to set predicate '{}': {}", predicate_, e.what()));
      }
    }
    ARROW_ASSIGN_OR_RAISE(auto row_group_in_file, format_reader->get_row_group_infos());
    ARROW_RETURN_NOT_OK(
        append_file_metadata(file_idx, cg_file, row_group_in_file, std::move(format_reader), rows_in_all_files));
  }

  total_rows_ = rows_in_all_files;
  return arrow::Status::OK();
}

folly::SemiFuture<arrow::Status> ColumnGroupReaderImpl::open_async() {
  const auto& cg_files = column_group_->files;
  std::vector<folly::SemiFuture<arrow::Result<OpenedFile>>> futures;
  futures.reserve(cg_files.size());

  for (size_t file_idx = 0; file_idx < cg_files.size(); ++file_idx) {
    auto cg_file = cg_files[file_idx];
    if (cg_file.start_index < 0 || cg_file.end_index < 0 || cg_file.start_index >= cg_file.end_index) {
      return folly::makeSemiFuture(arrow::Status::Invalid(
          fmt::format("Invalid start/end index in [file_index={}, path={}]", file_idx, cg_file.path)));
    }

    auto future =
        FormatReader::create_async(schema_, column_group_->format, cg_file, properties_, needed_columns_,
                                   key_retriever_)
            .deferValue([this, file_idx, cg_file](
                            arrow::Result<std::shared_ptr<FormatReader>>&& reader_result) -> arrow::Result<OpenedFile> {
              ARROW_ASSIGN_OR_RAISE(auto format_reader, std::move(reader_result));
              if (!predicate_.empty()) {
                try {
                  format_reader->set_predicate(predicate_);
                } catch (const std::exception& e) {
                  return arrow::Status::Invalid(fmt::format("Failed to set predicate '{}': {}", predicate_, e.what()));
                }
              }
              ARROW_ASSIGN_OR_RAISE(auto row_group_infos, format_reader->get_row_group_infos());
              return OpenedFile{file_idx, cg_file, std::move(format_reader), std::move(row_group_infos)};
            });
    futures.push_back(std::move(future));
  }

  return folly::collectAll(std::move(futures)).deferValue([this](auto&& open_results) mutable -> arrow::Status {
    chunk_infos_.clear();
    row_group_infos_.clear();
    format_readers_.clear();

    size_t rows_in_all_files = 0;
    for (auto& try_result : open_results) {
      if (try_result.hasException()) {
        return arrow::Status::IOError(try_result.exception().what().toStdString());
      }
      ARROW_ASSIGN_OR_RAISE(auto opened_file, std::move(try_result.value()));
      ARROW_RETURN_NOT_OK(append_file_metadata(opened_file.file_index, opened_file.file, opened_file.row_group_infos,
                                               std::move(opened_file.reader), rows_in_all_files));
    }

    total_rows_ = rows_in_all_files;
    return arrow::Status::OK();
  });
}

size_t ColumnGroupReaderImpl::total_number_of_chunks() const {
  assert(!format_readers_.empty());
  return chunk_infos_.size();
}

size_t ColumnGroupReaderImpl::total_rows() const {
  assert(!format_readers_.empty());
  return total_rows_;
}

arrow::Result<std::vector<int64_t>> ColumnGroupReaderImpl::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  assert(!format_readers_.empty());
  std::unordered_set<int64_t> unique_chunk_indices;
  std::vector<int64_t> chunk_indices;
  for (int64_t row_index : row_indices) {
    auto it = std::upper_bound(chunk_infos_.begin(), chunk_infos_.end(), row_index,
                               [](int64_t a, const ChunkInfo& b) { return a < b.global_row_end; });
    auto chunk_index = std::distance(chunk_infos_.begin(), it);
    if (chunk_index >= chunk_infos_.size()) {
      return arrow::Status::Invalid(fmt::format("Row index out of range: {}", row_index));
    }

    if (unique_chunk_indices.find(chunk_index) == unique_chunk_indices.end()) {
      unique_chunk_indices.insert(chunk_index);
      chunk_indices.emplace_back(chunk_index);
    }
  }

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ColumnGroupReaderImpl::get_chunk(int64_t chunk_index) {
  assert(!format_readers_.empty());
  FIU_RETURN_ON(FIUKEY_COLUMN_GROUP_READ_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_COLUMN_GROUP_READ_FAIL)));
  if (chunk_index < 0 || chunk_index >= chunk_infos_.size()) {
    return arrow::Status::Invalid(
        fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
  }
  auto chunk_info = chunk_infos_[chunk_index];

  ARROW_ASSIGN_OR_RAISE(auto rb, format_readers_[chunk_info.file_index]->get_chunk(chunk_info.row_group_index_in_file));

  // With predicate, Vortex's WithRowRange + WithFilter already produced the
  // correct subset — skip slicing since filtered row counts don't match
  // pre-filter chunk metadata.
  if (predicate_.empty() && (chunk_info.row_offset_in_row_group != 0 || chunk_info.number_of_rows != rb->num_rows())) {
    rb = rb->Slice(chunk_info.row_offset_in_row_group, chunk_info.number_of_rows);
  }

  return rb;
}

ChunkRBMapResult ColumnGroupReaderImpl::read_chunks_from_files(const std::vector<int64_t>& task_indices) {
  std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_rb_map;
  std::vector<std::vector<int64_t>> chunk_idxs_in_files(format_readers_.size());

  // Grouping row groups by file
  for (int64_t chunk_index : task_indices) {
    if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
      return arrow::Status::Invalid(
          fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
    }

    const auto& chunk_info = chunk_infos_[chunk_index];
    chunk_idxs_in_files[chunk_info.file_index].emplace_back(chunk_index);
  }

  // Read with range and fill chunk_rb_map
  for (size_t file_idx = 0; file_idx < chunk_idxs_in_files.size(); ++file_idx) {
    const auto& chunk_idxs = chunk_idxs_in_files[file_idx];
    if (chunk_idxs.empty()) {
      continue;
    }

    std::vector<std::pair<uint64_t, uint64_t>> ranges_in_file;

    // generate ranges_in_file and combine the range
    for (int64_t chunk_index : chunk_idxs) {
      const auto& chunk_info = chunk_infos_[chunk_index];
      if (ranges_in_file.empty()) {
        ranges_in_file.emplace_back(chunk_info.row_offset_in_file,
                                    chunk_info.row_offset_in_file + chunk_info.number_of_rows);
      } else {
        auto& last_range = ranges_in_file.back();

        // won't be overlay in same file
        assert(chunk_info.row_offset_in_file >= last_range.second);
        if (chunk_info.row_offset_in_file == last_range.second) {
          last_range.second = chunk_info.row_offset_in_file + chunk_info.number_of_rows;
        } else {
          ranges_in_file.emplace_back(chunk_info.row_offset_in_file,
                                      chunk_info.row_offset_in_file + chunk_info.number_of_rows);
        }
      }
    }

    ARROW_ASSIGN_OR_RAISE(auto reader, format_readers_[file_idx]->clone_reader());
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs_in_file;
    for (auto& range : ranges_in_file) {
      ARROW_ASSIGN_OR_RAISE(auto rbreader, reader->read_with_range(range.first, range.second));
      ARROW_ASSIGN_OR_RAISE(auto rbs, rbreader->ToRecordBatches());
      // append rbs to rbs_in_file
      std::move(rbs.begin(), rbs.end(), std::back_inserter(rbs_in_file));
    }

    // generate chunk_rb_map
    size_t rbs_idx = 0;
    size_t rbs_offset = 0;
    for (long long chunk_idx : chunk_idxs) {
      const auto& chunk_info = chunk_infos_[chunk_idx];
      if (UNLIKELY(((rbs_in_file[rbs_idx]->num_rows() - rbs_offset) < chunk_info.number_of_rows))) {
        return arrow::Status::Invalid(
            fmt::format("Invalid slice of record batchs: {} out of {}, [chunk info={}]", chunk_info.number_of_rows,
                        rbs_in_file[rbs_idx]->num_rows() - rbs_offset, chunk_info.ToString()));
      }

      auto rb = rbs_in_file[rbs_idx]->Slice(rbs_offset, chunk_info.number_of_rows);
      chunk_rb_map[chunk_idx] = rb;
      rbs_offset += chunk_info.number_of_rows;

      assert(rbs_offset <= rbs_in_file[rbs_idx]->num_rows());
      if (rbs_offset == rbs_in_file[rbs_idx]->num_rows()) {
        rbs_idx++;
        rbs_offset = 0;
      }
    }
  }
  return chunk_rb_map;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ColumnGroupReaderImpl::get_chunks(
    const std::vector<int64_t>& chunk_indices, size_t /*parallelism*/) {
  assert(!format_readers_.empty());

  // inject fault
  FIU_RETURN_ON(FIUKEY_COLUMN_GROUP_READ_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_COLUMN_GROUP_READ_FAIL)));

  // remove duplicate chunk indices and sort by chunk index
  std::vector<int64_t> unique_chunk_indices(chunk_indices.begin(), chunk_indices.end());
  std::sort(unique_chunk_indices.begin(), unique_chunk_indices.end());
  unique_chunk_indices.erase(std::unique(unique_chunk_indices.begin(), unique_chunk_indices.end()),
                             unique_chunk_indices.end());

  std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_rb_map;
  ARROW_ASSIGN_OR_RAISE(chunk_rb_map, read_chunks_from_files(unique_chunk_indices));

  // generate result
  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  for (const auto& chunk_idx : chunk_indices) {
    assert(chunk_rb_map.find(chunk_idx) != chunk_rb_map.end());
    result.emplace_back(chunk_rb_map[chunk_idx]);
  }

  return result;
}

arrow::Result<uint64_t> ColumnGroupReaderImpl::get_chunk_size(int64_t chunk_index) {
  assert(!format_readers_.empty());
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid(
        fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
  }
  return chunk_infos_[chunk_index].avg_memory_size;
}

arrow::Result<uint64_t> ColumnGroupReaderImpl::get_chunk_rows(int64_t chunk_index) {
  assert(!format_readers_.empty());
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid(
        fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
  }
  return chunk_infos_[chunk_index].number_of_rows;
}

std::vector<ChunkTask> ColumnGroupReaderImpl::get_natural_tasks(const std::vector<int64_t>& chunk_indices) {
  std::map<size_t, std::vector<int64_t>> file_groups;
  for (auto idx : chunk_indices) {
    file_groups[chunk_infos_[idx].file_index].push_back(idx);
  }

  std::vector<ChunkTask> tasks;
  for (auto& [file_idx, chunks] : file_groups) {
    auto& first_info = chunk_infos_[chunks[0]];
    ChunkTask current;
    current.file_index = file_idx;
    current.chunk_indices.push_back(chunks[0]);
    current.range_start = first_info.row_offset_in_file;
    current.range_end = first_info.row_offset_in_file + first_info.number_of_rows;

    for (size_t i = 1; i < chunks.size(); i++) {
      auto& prev = chunk_infos_[chunks[i - 1]];
      auto& curr = chunk_infos_[chunks[i]];

      if (curr.row_offset_in_file == prev.row_offset_in_file + prev.number_of_rows) {
        current.chunk_indices.push_back(chunks[i]);
        current.range_end = curr.row_offset_in_file + curr.number_of_rows;
      } else {
        tasks.push_back(std::move(current));
        current = ChunkTask{
            .file_index = file_idx,
            .chunk_indices = {chunks[i]},
            .range_start = curr.row_offset_in_file,
            .range_end = curr.row_offset_in_file + curr.number_of_rows,
        };
      }
    }
    tasks.push_back(std::move(current));
  }
  return tasks;
}

const ChunkInfo& ColumnGroupReaderImpl::get_chunk_info(int64_t chunk_index) const {
  assert(chunk_index >= 0 && static_cast<size_t>(chunk_index) < chunk_infos_.size());
  return chunk_infos_[chunk_index];
}

folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>>
ColumnGroupReaderImpl::get_chunks_async(const ChunkTask& task) {
  FIU_RETURN_ON(FIUKEY_COLUMN_GROUP_READ_FAIL,
                folly::makeSemiFuture(arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>(
                    arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_COLUMN_GROUP_READ_FAIL)))));

  auto cloned = format_readers_[task.file_index]->clone_reader();
  if (!cloned.ok()) {
    return folly::makeSemiFuture(arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>(cloned.status()));
  }

  auto reader = cloned.MoveValueUnsafe();
  auto chunk_idxs = task.chunk_indices;
  auto* self = this;

  auto slice_batches_to_chunks =
      [reader, chunk_idxs,
       self](auto&& rb_reader_result) -> arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> {
    ARROW_ASSIGN_OR_RAISE(auto rb_reader, std::move(rb_reader_result));
    ARROW_ASSIGN_OR_RAISE(auto rbs, rb_reader->ToRecordBatches());

    std::vector<std::shared_ptr<arrow::RecordBatch>> result;
    result.reserve(chunk_idxs.size());
    size_t rbs_idx = 0;
    size_t rbs_offset = 0;
    for (auto chunk_index : chunk_idxs) {
      const auto& chunk_info = self->chunk_infos_[chunk_index];
      if (UNLIKELY(rbs_idx >= rbs.size() || (rbs[rbs_idx]->num_rows() - rbs_offset) < chunk_info.number_of_rows)) {
        return arrow::Status::Invalid(
            fmt::format("Invalid slice of record batches in async read: [chunk_info={}]", chunk_info.ToString()));
      }
      auto rb = rbs[rbs_idx]->Slice(rbs_offset, chunk_info.number_of_rows);
      result.push_back(std::move(rb));
      rbs_offset += chunk_info.number_of_rows;
      if (rbs_offset == rbs[rbs_idx]->num_rows()) {
        rbs_idx++;
        rbs_offset = 0;
      }
    }
    return result;
  };

  return reader->read_with_range_async(task.range_start, task.range_end).deferValue(std::move(slice_batches_to_chunks));
}

std::shared_ptr<arrow::Schema> ColumnGroupReaderImpl::get_schema() const {
  if (format_readers_.empty()) {
    return nullptr;
  }
  return format_readers_[0]->get_schema();
}

}  // namespace milvus_storage::api
