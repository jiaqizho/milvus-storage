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

#include "milvus-storage/format/column_group_lazy_reader.h"

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/result.h>
#include <fmt/format.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY
#include "milvus-storage/common/fiu_local.h"

namespace milvus_storage::api {

class ColumnGroupLazyReaderImpl : public ColumnGroupLazyReader {
  public:
  ColumnGroupLazyReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                            const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                            const milvus_storage::api::Properties& properties,
                            const std::vector<std::string>& needed_columns,
                            const std::function<std::string(const std::string&)>& key_retriever);

  ~ColumnGroupLazyReaderImpl() override = default;

  arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices,
                                                    size_t parallelism = 1) override;
  arrow::Result<std::vector<TakeTask>> get_natural_tasks(const std::vector<int64_t>& row_indices) override;
  folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> take_async(const TakeTask& task) override;

  private:
  struct ReaderState {
    explicit ReaderState(size_t file_count) : loaded_format_readers(file_count) {}

    std::mutex prepare_mutex;
    std::vector<std::shared_ptr<FormatReader>> loaded_format_readers;
  };

  arrow::Status prepare_format_readers(const std::vector<int64_t>& row_indices);
  folly::SemiFuture<arrow::Status> prepare_format_reader_async(uint32_t file_index);
  arrow::Result<std::shared_ptr<arrow::Table>> take_rows_from_files(const std::vector<int64_t>& row_indices);

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;

  std::shared_ptr<ReaderState> state_;
};

ColumnGroupLazyReaderImpl::ColumnGroupLazyReaderImpl(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const milvus_storage::api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever)
    : schema_(schema),
      column_group_(column_group),
      properties_(properties),
      needed_columns_(needed_columns),
      key_retriever_(key_retriever),
      state_(std::make_shared<ReaderState>(column_group->files.size())) {}

static inline arrow::Result<std::pair<uint32_t, int64_t>> get_index_and_offset_of_file(
    const std::vector<ColumnGroupFile>& files, const int64_t& global_row_index) {
  int64_t row_index_remain = global_row_index;

  for (uint32_t i = 0; i < files.size(); i++) {
    if (files[i].start_index < 0 || files[i].end_index < 0) {
      return arrow::Status::Invalid(
          fmt::format("Invalid start/end index in [file_index={}, path={}]", i, files[i].path));
    }

    int64_t num_of_rows_in_file = (files[i].end_index - files[i].start_index);
    if (row_index_remain < num_of_rows_in_file) {
      // use the physical row index in file
      return std::make_pair(i, row_index_remain + files[i].start_index);
    }

    row_index_remain -= num_of_rows_in_file;
  }

  return arrow::Status::Invalid(
      fmt::format("Row index is greater than the maximum range, [row_index={}]", global_row_index));
}

static arrow::Status validate_sorted_unique_row_indices(const std::vector<int64_t>& row_indices) {
  for (size_t i = 1; i < row_indices.size(); i++) {
    if (row_indices[i] <= row_indices[i - 1]) {
      return arrow::Status::Invalid(
          fmt::format("Input row indices is not sorted or not unique,[index={}, row_index={}]", i, row_indices[i]));
    }
  }
  return arrow::Status::OK();
}

arrow::Status ColumnGroupLazyReaderImpl::prepare_format_readers(const std::vector<int64_t>& row_indices) {
  std::lock_guard<std::mutex> lock(state_->prepare_mutex);
  const auto& cg_files = column_group_->files;
  for (const auto& row_index : row_indices) {
    uint32_t file_index;
    [[maybe_unused]] int64_t _unused_row_index_in_file;

    if (row_index < 0) {
      return arrow::Status::Invalid(fmt::format("Row index is less than 0, [row_index={}]", row_index));
    }
    ARROW_ASSIGN_OR_RAISE(std::tie(file_index, _unused_row_index_in_file),
                          get_index_and_offset_of_file(cg_files, row_index));
    if (!state_->loaded_format_readers[file_index]) {
      ARROW_ASSIGN_OR_RAISE(state_->loaded_format_readers[file_index],
                            FormatReader::create(schema_, column_group_->format, cg_files[file_index], properties_,
                                                 needed_columns_, key_retriever_));
    }
  }
  return arrow::Status::OK();
}

folly::SemiFuture<arrow::Status> ColumnGroupLazyReaderImpl::prepare_format_reader_async(uint32_t file_index) {
  const auto& cg_files = column_group_->files;
  if (file_index >= cg_files.size()) {
    return folly::makeSemiFuture(
        arrow::Status::Invalid(fmt::format("File index out of range: {} out of {}", file_index, cg_files.size())));
  }

  auto state = state_;
  {
    std::lock_guard<std::mutex> lock(state->prepare_mutex);
    if (state->loaded_format_readers[file_index]) {
      return folly::makeSemiFuture(arrow::Status::OK());
    }
  }

  return FormatReader::create_async(schema_, column_group_->format, cg_files[file_index], properties_, needed_columns_,
                                    key_retriever_)
      .deferValue([state, file_index](arrow::Result<std::shared_ptr<FormatReader>>&& reader_result) -> arrow::Status {
        ARROW_ASSIGN_OR_RAISE(auto format_reader, std::move(reader_result));
        std::lock_guard<std::mutex> lock(state->prepare_mutex);
        if (!state->loaded_format_readers[file_index]) {
          state->loaded_format_readers[file_index] = std::move(format_reader);
        }
        return arrow::Status::OK();
      });
}

arrow::Result<std::shared_ptr<arrow::Table>> ColumnGroupLazyReaderImpl::take_rows_from_files(
    const std::vector<int64_t>& row_indices) {
  const auto& cg_files = column_group_->files;
  std::vector<std::vector<int64_t>> indices_in_files(cg_files.size());
  for (const auto& row_index : row_indices) {
    uint32_t file_index;
    int64_t row_index_in_file;
    ARROW_ASSIGN_OR_RAISE(std::tie(file_index, row_index_in_file), get_index_and_offset_of_file(cg_files, row_index));
    indices_in_files[file_index].emplace_back(row_index_in_file);
  }

  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (size_t file_index = 0; file_index < indices_in_files.size(); file_index++) {
    if (indices_in_files[file_index].empty()) {
      continue;
    }

    ARROW_ASSIGN_OR_RAISE(auto cloned_reader, state_->loaded_format_readers[file_index]->clone_reader());
    ARROW_ASSIGN_OR_RAISE(auto table, cloned_reader->take(indices_in_files[file_index]));
    tables.emplace_back(table);
  }

  // won't copy table with same schema
  return arrow::ConcatenateTables(tables);
}

arrow::Result<std::shared_ptr<arrow::Table>> ColumnGroupLazyReaderImpl::take(const std::vector<int64_t>& row_indices,
                                                                             size_t /*parallelism*/) {
  FIU_RETURN_ON(FIUKEY_TAKE_ROWS_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_TAKE_ROWS_FAIL)));

  ARROW_RETURN_NOT_OK(validate_sorted_unique_row_indices(row_indices));
  ARROW_RETURN_NOT_OK(prepare_format_readers(row_indices));
  return take_rows_from_files(row_indices);
}

arrow::Result<std::unique_ptr<ColumnGroupLazyReader>> ColumnGroupLazyReader::create(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const milvus_storage::api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever) {
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
      assert(field);
      fields.emplace_back(field);
    }
    out_schema = std::make_shared<arrow::Schema>(fields);
  }
  // When schema is nullptr, out_schema stays nullptr;
  // the RecordBatches returned by the format reader will carry the file schema.

  return std::make_unique<ColumnGroupLazyReaderImpl>(out_schema, column_group, properties, filtered_columns,
                                                     key_retriever);
}

arrow::Result<std::vector<TakeTask>> ColumnGroupLazyReaderImpl::get_natural_tasks(
    const std::vector<int64_t>& row_indices) {
  ARROW_RETURN_NOT_OK(validate_sorted_unique_row_indices(row_indices));

  const auto& files = column_group_->files;
  // file_idx -> [(global_row_index, original_position)]
  std::map<uint32_t, std::vector<std::pair<int64_t, size_t>>> file_groups;

  for (size_t pos = 0; pos < row_indices.size(); ++pos) {
    if (row_indices[pos] < 0) {
      return arrow::Status::Invalid(fmt::format("Row index is less than 0, [row_index={}]", row_indices[pos]));
    }
    ARROW_ASSIGN_OR_RAISE(auto file_and_offset, get_index_and_offset_of_file(files, row_indices[pos]));
    auto [file_idx, _] = file_and_offset;
    file_groups[file_idx].push_back({row_indices[pos], pos});
  }

  std::vector<TakeTask> tasks;
  tasks.reserve(file_groups.size());
  for (auto& [file_idx, rows_and_positions] : file_groups) {
    TakeTask task;
    task.file_index = file_idx;
    task.row_indices.reserve(rows_and_positions.size());
    task.original_positions.reserve(rows_and_positions.size());
    for (auto& [row, pos] : rows_and_positions) {
      task.row_indices.push_back(row);
      task.original_positions.push_back(pos);
    }
    tasks.push_back(std::move(task));
  }
  return tasks;
}

folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> ColumnGroupLazyReaderImpl::take_async(
    const TakeTask& task) {
  FIU_RETURN_ON(FIUKEY_TAKE_ROWS_FAIL,
                folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(
                    arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_TAKE_ROWS_FAIL)))));

  auto state = state_;
  auto column_group = column_group_;
  return prepare_format_reader_async(task.file_index)
      .deferValue(
          [state, column_group, task](arrow::Status prepare_status) -> arrow::Result<std::shared_ptr<FormatReader>> {
            ARROW_RETURN_NOT_OK(prepare_status);

            const auto& cg_files = column_group->files;
            for (auto global_row : task.row_indices) {
              ARROW_ASSIGN_OR_RAISE(auto file_and_offset, get_index_and_offset_of_file(cg_files, global_row));
              auto [file_idx, _] = file_and_offset;
              if (file_idx != task.file_index) {
                return arrow::Status::Invalid(
                    fmt::format("TakeTask row does not belong to task file. [row={}, expected_file={}, actual_file={}]",
                                global_row, task.file_index, file_idx));
              }
            }

            std::lock_guard<std::mutex> lock(state->prepare_mutex);
            return state->loaded_format_readers[task.file_index]->clone_reader();
          })
      .deferValue([column_group, task](arrow::Result<std::shared_ptr<FormatReader>>&& reader_result)
                      -> folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> {
        if (!reader_result.ok()) {
          return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(reader_result.status()));
        }
        auto reader = reader_result.MoveValueUnsafe();

        const auto& cg_files = column_group->files;
        std::vector<int64_t> rows_in_file;
        rows_in_file.reserve(task.row_indices.size());
        for (auto global_row : task.row_indices) {
          auto result = get_index_and_offset_of_file(cg_files, global_row);
          if (!result.ok()) {
            return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(result.status()));
          }
          rows_in_file.push_back(result.MoveValueUnsafe().second);
        }

        auto forward_result = [reader](auto&& table_result) -> arrow::Result<std::shared_ptr<arrow::Table>> {
          return std::move(table_result);
        };
        return reader->take_async(rows_in_file).deferValue(std::move(forward_result));
      });
}

};  // namespace milvus_storage::api
