// Copyright 2024 Zilliz
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

// V2 vs V3 Reader Benchmark
// Compare performance between packed/ low-level reader (V2: PackedRecordBatchReader)
// and top-level Reader API (V3: get_record_batch_reader / get_chunk_reader).
// V2 uses PackedRecordBatchWriter + PackedRecordBatchReader directly.
// V3 uses the high-level Writer + Reader API.

#include "benchmark_format_common.h"

#include <arrow/table.h>

#include "milvus-storage/packed/writer.h"
#include "milvus-storage/packed/reader.h"
#include "milvus-storage/thread_pool.h"

namespace milvus_storage {
namespace benchmark {

using namespace milvus_storage::api;

//=============================================================================
// V2 vs V3 Benchmark Fixture
//=============================================================================

inline size_t ComputeNumBatches(size_t num_rows) {
  return std::max<size_t>(1, num_rows / 1000);  // ~1000 rows per batch
}

class V2V3BenchFixture : public FormatBenchFixture {
  public:
  void SetUp(::benchmark::State& st) override {
    FormatBenchFixture::SetUp(st);

    if (!CheckFormatAvailable(st, LOON_FORMAT_PARQUET)) {
      return;
    }

    // Reuse common schema (already has PARQUET:field_id metadata)
    BENCH_ASSERT_AND_ASSIGN(schema_, CreateSchema(), st);

    ThreadPoolHolder::WithSingleton(1);
  }

  void TearDown(::benchmark::State& st) override {
    ThreadPoolHolder::Release();
    FormatBenchFixture::TearDown(st);
  }

  protected:
  // Write data using PackedRecordBatchWriter (V2 path) and return the file path
  arrow::Status PrepareV2Data(const DataSizeConfig& config, std::string& out_path) {
    out_path = GetUniquePath("v2_test") + "/data.parquet";
    std::vector<std::string> paths = {out_path};
    auto column_groups = std::vector<std::vector<int>>{{0, 1, 2, 3}};
    StorageConfig storage_config;

    ARROW_ASSIGN_OR_RAISE(auto writer, PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config, column_groups,
                                                                     DEFAULT_WRITE_BUFFER_SIZE));

    size_t num_batches = ComputeNumBatches(config.num_rows);
    size_t rows_per_batch = config.num_rows / num_batches;
    DataSizeConfig batch_config{"v2", rows_per_batch, config.vector_dim, config.string_length};
    for (size_t i = 0; i < num_batches; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto batch,
                            CreateBatch(schema_, batch_config, true, static_cast<int64_t>(i * rows_per_batch)));
      ARROW_RETURN_NOT_OK(writer->Write(batch));
    }
    auto result = writer->Close();

    return arrow::Status::OK();
  }

  // Write data using Writer API (V3 path) and return column groups
  arrow::Status PrepareV3Data(const DataSizeConfig& config, std::shared_ptr<ColumnGroups>& out_cgs) {
    std::string path = GetUniquePath("v3_test");
    ARROW_ASSIGN_OR_RAISE(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_));
    auto writer = Writer::create(path, schema_, std::move(policy), properties_);
    if (!writer) {
      return arrow::Status::Invalid("Failed to create writer");
    }

    size_t num_batches = ComputeNumBatches(config.num_rows);
    size_t rows_per_batch = config.num_rows / num_batches;
    DataSizeConfig batch_config{"v3", rows_per_batch, config.vector_dim, config.string_length};
    for (size_t i = 0; i < num_batches; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto batch,
                            CreateBatch(schema_, batch_config, true, static_cast<int64_t>(i * rows_per_batch)));
      ARROW_RETURN_NOT_OK(writer->write(batch));
    }
    ARROW_ASSIGN_OR_RAISE(out_cgs, writer->close());

    return arrow::Status::OK();
  }

  std::shared_ptr<arrow::Schema> schema_;
};

//=============================================================================
// V2: PackedRecordBatchReader Benchmark (Low-level)
//=============================================================================

// Args: [config_idx]
BENCHMARK_DEFINE_F(V2V3BenchFixture, V2_PackedRecordBatchReader)(::benchmark::State& st) {
  size_t config_idx = static_cast<size_t>(st.range(0));
  DataSizeConfig config = DataSizeConfig::FromIndex(config_idx);

  // Write data using packed writer
  std::string path;
  BENCH_ASSERT_STATUS_OK(PrepareV2Data(config, path), st);

  std::vector<std::string> paths = {path};

  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;

  for (auto _ : st) {
    PackedRecordBatchReader reader(fs_, paths, schema_, DEFAULT_READ_BUFFER_SIZE);

    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      BENCH_ASSERT_STATUS_OK(reader.ReadNext(&batch), st);
      if (batch == nullptr)
        break;
      total_rows_read += batch->num_rows();
      total_bytes_read += CalculateRawDataSize(batch);
    }
    BENCH_ASSERT_STATUS_OK(reader.Close(), st);
  }

  ReportThroughput(st, total_bytes_read, total_rows_read);
  st.SetLabel("v2/" + config.name);
}

BENCHMARK_REGISTER_F(V2V3BenchFixture, V2_PackedRecordBatchReader)
    ->Args({0})  // Small
    ->Args({1})  // Medium
    ->Args({2})  // Large
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// V3: Reader::get_record_batch_reader Benchmark (Top-level)
//=============================================================================

// Args: [config_idx]
BENCHMARK_DEFINE_F(V2V3BenchFixture, V3_RecordBatchReader)(::benchmark::State& st) {
  size_t config_idx = static_cast<size_t>(st.range(0));
  DataSizeConfig config = DataSizeConfig::FromIndex(config_idx);

  // Write data using Writer API
  std::shared_ptr<ColumnGroups> cgs;
  BENCH_ASSERT_STATUS_OK(PrepareV3Data(config, cgs), st);

  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;

  for (auto _ : st) {
    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    BENCH_ASSERT_NOT_NULL(reader, st);

    BENCH_ASSERT_AND_ASSIGN(auto batch_reader, reader->get_record_batch_reader(), st);

    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      BENCH_ASSERT_STATUS_OK(batch_reader->ReadNext(&batch), st);
      if (batch == nullptr) {
        break;
      }
      total_rows_read += batch->num_rows();
      total_bytes_read += CalculateRawDataSize(batch);
    }
  }

  ReportThroughput(st, total_bytes_read, total_rows_read);
  st.SetLabel("v3-rb/" + config.name);
}

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_RecordBatchReader)
    ->Args({0})  // Small
    ->Args({1})  // Medium
    ->Args({2})  // Large
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// V3: Reader::get_chunk_reader Benchmark (Top-level Chunk Access)
//=============================================================================

// Args: [config_idx]
BENCHMARK_DEFINE_F(V2V3BenchFixture, V3_ChunkReader)(::benchmark::State& st) {
  size_t config_idx = static_cast<size_t>(st.range(0));
  DataSizeConfig config = DataSizeConfig::FromIndex(config_idx);

  // Write data using Writer API
  std::shared_ptr<ColumnGroups> cgs;
  BENCH_ASSERT_STATUS_OK(PrepareV3Data(config, cgs), st);

  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;

  for (auto _ : st) {
    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    BENCH_ASSERT_NOT_NULL(reader, st);

    BENCH_ASSERT_AND_ASSIGN(auto chunk_reader, reader->get_chunk_reader(0), st);

    size_t total_chunks = chunk_reader->total_number_of_chunks();
    for (size_t i = 0; i < total_chunks; ++i) {
      BENCH_ASSERT_AND_ASSIGN(auto batch, chunk_reader->get_chunk(i), st);
      total_rows_read += batch->num_rows();
      total_bytes_read += CalculateRawDataSize(batch);
    }
  }

  ReportThroughput(st, total_bytes_read, total_rows_read);
  st.SetLabel("v3-chunk/" + config.name);
}

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_ChunkReader)
    ->Args({0})  // Small
    ->Args({1})  // Medium
    ->Args({2})  // Large
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// V2: PackedRecordBatchWriter Benchmark (Low-level)
//=============================================================================

// Args: [config_idx]
BENCHMARK_DEFINE_F(V2V3BenchFixture, V2_PackedRecordBatchWriter)(::benchmark::State& st) {
  size_t config_idx = static_cast<size_t>(st.range(0));
  DataSizeConfig config = DataSizeConfig::FromIndex(config_idx);

  // Prepare batches
  size_t num_batches = ComputeNumBatches(config.num_rows);
  size_t rows_per_batch = config.num_rows / num_batches;
  DataSizeConfig batch_config{"v2", rows_per_batch, config.vector_dim, config.string_length};

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  for (size_t i = 0; i < num_batches; ++i) {
    BENCH_ASSERT_AND_ASSIGN(auto batch,
                            CreateBatch(schema_, batch_config, true, static_cast<int64_t>(i * rows_per_batch)), st);
    batches.push_back(batch);
  }

  int64_t total_rows_written = 0;
  int64_t total_bytes_written = 0;

  std::string base_path = GetUniquePath("v2_write_bench");

  for (auto _ : st) {
    std::string path = base_path + "/data.parquet";
    std::vector<std::string> paths = {path};
    auto column_groups = std::vector<std::vector<int>>{{0, 1, 2, 3}};
    StorageConfig storage_config;

    BENCH_ASSERT_AND_ASSIGN(
        auto writer,
        PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config, column_groups, DEFAULT_WRITE_BUFFER_SIZE),
        st);

    for (const auto& batch : batches) {
      BENCH_ASSERT_STATUS_OK(writer->Write(batch), st);
      total_rows_written += batch->num_rows();
      total_bytes_written += CalculateRawDataSize(batch);
    }
    auto result = writer->Close();

    // Cleanup for next iteration
    BENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path), st);
    BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, base_path), st);
  }

  ReportThroughput(st, total_bytes_written, total_rows_written);
  st.SetLabel("v2-writer/" + config.name);
}

BENCHMARK_REGISTER_F(V2V3BenchFixture, V2_PackedRecordBatchWriter)
    ->Args({0})  // Small
    ->Args({1})  // Medium
    ->Args({2})  // Large
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// V3: Writer API Benchmark (Top-level)
//=============================================================================

// Args: [config_idx]
BENCHMARK_DEFINE_F(V2V3BenchFixture, V3_Writer)(::benchmark::State& st) {
  size_t config_idx = static_cast<size_t>(st.range(0));
  DataSizeConfig config = DataSizeConfig::FromIndex(config_idx);

  // Prepare batches
  size_t num_batches = ComputeNumBatches(config.num_rows);
  size_t rows_per_batch = config.num_rows / num_batches;
  DataSizeConfig batch_config{"v3", rows_per_batch, config.vector_dim, config.string_length};

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  for (size_t i = 0; i < num_batches; ++i) {
    BENCH_ASSERT_AND_ASSIGN(auto batch,
                            CreateBatch(schema_, batch_config, true, static_cast<int64_t>(i * rows_per_batch)), st);
    batches.push_back(batch);
  }

  int64_t total_rows_written = 0;
  int64_t total_bytes_written = 0;

  std::string base_path = GetUniquePath("v3_write_bench");

  for (auto _ : st) {
    BENCH_ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema_), st);
    auto writer = Writer::create(base_path, schema_, std::move(policy), properties_);
    BENCH_ASSERT_NOT_NULL(writer, st);

    for (const auto& batch : batches) {
      BENCH_ASSERT_STATUS_OK(writer->write(batch), st);
      total_rows_written += batch->num_rows();
      total_bytes_written += CalculateRawDataSize(batch);
    }
    BENCH_ASSERT_AND_ASSIGN(auto cgs, writer->close(), st);

    // Cleanup for next iteration
    BENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path), st);
    BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, base_path), st);
  }

  ReportThroughput(st, total_bytes_written, total_rows_written);
  st.SetLabel("v3-writer/" + config.name);
}

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_Writer)
    ->Args({0})  // Small
    ->Args({1})  // Medium
    ->Args({2})  // Large
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Typical Benchmarks
// Run with: --benchmark_filter="Typical/"
//=============================================================================

BENCHMARK_REGISTER_F(V2V3BenchFixture, V2_PackedRecordBatchReader)
    ->Name("Typical/V2_Reader")
    ->Args({1})  // Medium
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_RecordBatchReader)
    ->Name("Typical/V3_Reader")
    ->Args({1})  // Medium
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(V2V3BenchFixture, V2_PackedRecordBatchWriter)
    ->Name("Typical/V2_Writer")
    ->Args({1})  // Medium
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_Writer)
    ->Name("Typical/V3_Writer")
    ->Args({1})  // Medium
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

}  // namespace benchmark
}  // namespace milvus_storage
