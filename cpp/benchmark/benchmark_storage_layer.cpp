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

// Storage Layer Benchmark (Phase 3)
// Compare Milvus-Storage (Parquet/Vortex + Transaction) vs Lance Native.
// Measures end-to-end performance including transaction overhead.

#include "benchmark_format_common.h"

#include <filesystem>
#include <iostream>
#include <arrow/table.h>
#include <arrow/record_batch.h>

#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/thread_pool.h"

#include <sys/resource.h>

#include <arrow/c/bridge.h>
#include "format/bridge/rust/include/lance_bridge.h"
#include "format/bridge/rust/include/vortex_bridge.h"

namespace {

struct CpuTime {
  double user_ms = 0;
  double sys_ms = 0;
};

CpuTime GetProcessCpuTime() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return {
      ru.ru_utime.tv_sec * 1e3 + ru.ru_utime.tv_usec / 1e3,
      ru.ru_stime.tv_sec * 1e3 + ru.ru_stime.tv_usec / 1e3,
  };
}

CpuTime operator-(const CpuTime& a, const CpuTime& b) { return {a.user_ms - b.user_ms, a.sys_ms - b.sys_ms}; }

void ReportCpuTime(::benchmark::State& st, const CpuTime& elapsed) {
  double iters = static_cast<double>(st.iterations());
  st.counters["cpu_user_ms/iter"] = ::benchmark::Counter(elapsed.user_ms / iters, ::benchmark::Counter::kDefaults);
  st.counters["cpu_sys_ms/iter"] = ::benchmark::Counter(elapsed.sys_ms / iters, ::benchmark::Counter::kDefaults);
  st.counters["cpu_total_ms/iter"] =
      ::benchmark::Counter((elapsed.user_ms + elapsed.sys_ms) / iters, ::benchmark::Counter::kDefaults);
}

}  // namespace
#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"

namespace milvus_storage {
namespace benchmark {

using namespace milvus_storage::api;
using namespace milvus_storage::api::transaction;

//=============================================================================
// Storage Format Types
//=============================================================================

enum class StorageFormatType { PARQUET = 0, VORTEX = 1, MIXED = 2 };

inline const char* StorageFormatTypeName(StorageFormatType type) {
  switch (type) {
    case StorageFormatType::PARQUET:
      return "parquet";
    case StorageFormatType::VORTEX:
      return "vortex";
    case StorageFormatType::MIXED:
      return "mixed(pq+vtx)";
    default:
      return "unknown";
  }
}

//=============================================================================
// Helper: Project columns from a RecordBatch
//=============================================================================

inline std::shared_ptr<arrow::RecordBatch> ProjectColumns(const std::shared_ptr<arrow::RecordBatch>& batch,
                                                          const std::vector<int>& column_indices) {
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (int idx : column_indices) {
    arrays.push_back(batch->column(idx));
    fields.push_back(batch->schema()->field(idx));
  }
  return arrow::RecordBatch::Make(arrow::schema(fields), batch->num_rows(), arrays);
}

//=============================================================================
// Storage Layer Benchmark Fixture
//=============================================================================

class StorageLayerFixture : public FormatBenchFixtureBase<false> {
  public:
  void SetUp(::benchmark::State& st) override {
    FormatBenchFixtureBase<false>::SetUp(st);

    // Get schema from data loader
    schema_ = GetLoaderSchema();

    // Batches are loaded lazily via EnsureBatchesLoaded() — only needed by write benchmarks
    // and cache-miss paths. This avoids re-reading source data for every read benchmark.
  }

  void TearDown(::benchmark::State& st) override {
    schema_.reset();
    ThreadPoolHolder::Release();
    FormatBenchFixtureBase<false>::TearDown(st);
  }

  void ConfigureThreadPool(int num_threads) { ThreadPoolHolder::WithSingleton(num_threads); }

  protected:
  //-----------------------------------------------------------------------
  // MilvusStorage: Write + Transaction Commit (using pre-loaded batches)
  //-----------------------------------------------------------------------
  arrow::Status WriteMilvusStorage(StorageFormatType format_type, const std::string& path) {
    ARROW_RETURN_NOT_OK(EnsureBatchesLoaded());
    std::string format = (format_type == StorageFormatType::PARQUET) ? LOON_FORMAT_PARQUET : LOON_FORMAT_VORTEX;

    // Use format-specific policy: Vortex=Single, Parquet=SchemaBase
    std::string patterns = GetSchemaBasePatterns();
    ARROW_ASSIGN_OR_RAISE(auto policy, CreatePolicyForFormat(patterns, format, schema_));

    auto writer = Writer::create(path, schema_, std::move(policy), properties_);
    if (!writer)
      return arrow::Status::Invalid("Failed to create writer");

    // Write all pre-loaded batches
    for (const auto& batch : batches_) {
      ARROW_RETURN_NOT_OK(writer->write(batch));
    }
    ARROW_ASSIGN_OR_RAISE(auto cgs, writer->close());

    ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
    txn->AppendFiles(*cgs);
    ARROW_ASSIGN_OR_RAISE(auto version, txn->Commit());

    return arrow::Status::OK();
  }

  //-----------------------------------------------------------------------
  // MilvusStorage: Open Transaction + Read (with stats collection)
  //-----------------------------------------------------------------------
  arrow::Status ReadMilvusStorageWithStats(const std::string& path, int64_t& out_rows, int64_t& out_bytes) {
    ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
    ARROW_ASSIGN_OR_RAISE(auto manifest, txn->GetManifest());
    auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());

    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    if (!reader)
      return arrow::Status::Invalid("Failed to create reader");

    ARROW_ASSIGN_OR_RAISE(auto batch_reader, reader->get_record_batch_reader());

    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
      if (!batch)
        break;
      out_rows += batch->num_rows();
      out_bytes += CalculateRawDataSize(batch);
    }
    return arrow::Status::OK();
  }

  //-----------------------------------------------------------------------
  // MilvusStorage: Open Transaction + Read (no stats, for benchmark loop)
  //-----------------------------------------------------------------------
  arrow::Status ReadMilvusStorage(const std::string& path) {
    ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
    ARROW_ASSIGN_OR_RAISE(auto manifest, txn->GetManifest());
    auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());

    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    if (!reader)
      return arrow::Status::Invalid("Failed to create reader");

    ARROW_ASSIGN_OR_RAISE(auto batch_reader, reader->get_record_batch_reader());

    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
      if (!batch)
        break;
      // No stats collection in benchmark loop
    }
    return arrow::Status::OK();
  }

  //-----------------------------------------------------------------------
  // MilvusStorage: Take with pre-cached ColumnGroups (no transaction in hot path)
  //-----------------------------------------------------------------------
  arrow::Result<std::shared_ptr<ColumnGroups>> LoadColumnGroups(const std::string& path) {
    ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
    ARROW_ASSIGN_OR_RAISE(auto manifest, txn->GetManifest());
    return std::make_shared<ColumnGroups>(manifest->columnGroups());
  }

  //-----------------------------------------------------------------------
  // Verify Take results: check row count and schema match expectations
  //-----------------------------------------------------------------------
  arrow::Status VerifyTakeResult(const std::string& label,
                                 int64_t actual_rows,
                                 int64_t expected_rows,
                                 int64_t actual_bytes) {
    if (actual_rows != expected_rows) {
      return arrow::Status::Invalid(label, " take verification failed: expected ", expected_rows, " rows but got ",
                                    actual_rows);
    }
    if (actual_bytes <= 0) {
      return arrow::Status::Invalid(label, " take verification failed: got ", actual_bytes, " bytes (expected > 0)");
    }
    return arrow::Status::OK();
  }

  //-----------------------------------------------------------------------
  // Format availability check
  //-----------------------------------------------------------------------
  bool CheckStorageFormatAvailable(::benchmark::State& st, StorageFormatType format_type) {
    if (format_type == StorageFormatType::VORTEX || format_type == StorageFormatType::MIXED) {
      if (!IsFormatAvailable(LOON_FORMAT_VORTEX)) {
        st.SkipWithError("Vortex format not available");
        return false;
      }
    }
    return true;
  }

  //-----------------------------------------------------------------------
  // Cached data path: deterministic, survives across runs (outside base_path_)
  //-----------------------------------------------------------------------
  std::string GetCachedDataPath(const std::string& format_name) const {
    namespace fs = std::filesystem;
    return (fs::path("bench_cache") / GetDataDescription() / format_name).lexically_normal().string();
  }

  // Ensure MilvusStorage data exists at path, write only if missing
  arrow::Status EnsureMilvusStorageData(StorageFormatType format_type, const std::string& path) {
    auto txn_result = Transaction::Open(fs_, path);
    if (txn_result.ok()) {
      // Validate that manifest has non-empty column groups
      auto manifest_result = (*txn_result)->GetManifest();
      if (manifest_result.ok() && !(*manifest_result)->columnGroups().empty()) {
        return arrow::Status::OK();
      }
    }
    // Data missing or invalid: (re-)create
    ARROW_RETURN_NOT_OK(DeleteTestDir(fs_, path));
    ARROW_RETURN_NOT_OK(CreateTestDir(fs_, path));
    return WriteMilvusStorage(format_type, path);
  }

  // Ensure Lance dataset exists at URI, write only if missing
  arrow::Status EnsureLanceData(const std::string& lance_uri,
                                const lance::LanceStorageOptions& storage_options,
                                const std::string& path) {
    try {
      auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);
      if (!dataset->GetAllFragmentIds().empty()) {
        return arrow::Status::OK();
      }
    } catch (...) {
    }
    // Data missing or invalid: (re-)create
    ARROW_RETURN_NOT_OK(DeleteTestDir(fs_, path));
    ARROW_RETURN_NOT_OK(CreateTestDir(fs_, path));
    return WriteLanceDataset(lance_uri, storage_options);
  }

  //-----------------------------------------------------------------------
  // Lance: Build URI and storage options for cloud storage support
  //-----------------------------------------------------------------------
  arrow::Result<std::string> BuildLanceUri(const std::string& relative_path) {
    ArrowFileSystemConfig fs_config;
    ARROW_RETURN_NOT_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
    return lance::BuildLanceBaseUri(fs_config, relative_path);
  }

  lance::LanceStorageOptions GetLanceStorageOptions() {
    ArrowFileSystemConfig fs_config;
    auto status = ArrowFileSystemConfig::create_file_system_config(properties_, fs_config);
    if (!status.ok()) {
      return {};
    }
    return lance::ToLanceStorageOptions(fs_config);
  }

  // Write test data to a lance dataset using pre-loaded batches
  arrow::Status WriteLanceDataset(const std::string& lance_uri, const lance::LanceStorageOptions& storage_options) {
    ARROW_RETURN_NOT_OK(EnsureBatchesLoaded());
    // Create a RecordBatchReader from pre-loaded batches
    ARROW_ASSIGN_OR_RAISE(auto batch_reader, arrow::RecordBatchReader::Make(batches_, schema_));

    ArrowArrayStream stream;
    ARROW_RETURN_NOT_OK(arrow::ExportRecordBatchReader(batch_reader, &stream));

    try {
      auto dataset = lance::BlockingDataset::WriteDataset(lance_uri, &stream, storage_options);
    } catch (const lance::LanceException& e) {
      return arrow::Status::IOError("Lance write failed: ", e.what());
    }
    return arrow::Status::OK();
  }

  //-----------------------------------------------------------------------
  // Scalar-only data: filter out FixedSizeBinary/LIST vector columns
  //-----------------------------------------------------------------------
  arrow::Result<std::pair<std::shared_ptr<arrow::Schema>, std::vector<std::shared_ptr<arrow::RecordBatch>>>>
  GetScalarOnlyData() {
    ARROW_RETURN_NOT_OK(EnsureBatchesLoaded());
    auto scalar_cols = GetScalarProjection();

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<int> col_indices;
    for (const auto& name : *scalar_cols) {
      auto idx = schema_->GetFieldIndex(name);
      if (idx >= 0) {
        fields.push_back(schema_->field(idx));
        col_indices.push_back(idx);
      }
    }
    auto scalar_schema = arrow::schema(fields);

    std::vector<std::shared_ptr<arrow::RecordBatch>> scalar_batches;
    for (const auto& batch : batches_) {
      std::vector<std::shared_ptr<arrow::Array>> arrays;
      for (int idx : col_indices) {
        arrays.push_back(batch->column(idx));
      }
      scalar_batches.push_back(arrow::RecordBatch::Make(scalar_schema, batch->num_rows(), arrays));
    }
    return std::make_pair(scalar_schema, std::move(scalar_batches));
  }

  arrow::Status WriteMilvusStorageScalarOnly(StorageFormatType format_type, const std::string& path) {
    ARROW_ASSIGN_OR_RAISE(auto scalar_data, GetScalarOnlyData());
    auto& scalar_schema = scalar_data.first;
    auto& scalar_batches = scalar_data.second;
    std::string format = (format_type == StorageFormatType::PARQUET) ? LOON_FORMAT_PARQUET : LOON_FORMAT_VORTEX;

    // All scalar columns in one group (use regex OR '|' so they match a single pattern = one file)
    std::string patterns;
    for (int i = 0; i < scalar_schema->num_fields(); ++i) {
      if (i > 0)
        patterns += "|";
      patterns += scalar_schema->field(i)->name();
    }
    ARROW_ASSIGN_OR_RAISE(auto policy, CreatePolicyForFormat(patterns, format, scalar_schema));

    auto writer = Writer::create(path, scalar_schema, std::move(policy), properties_);
    if (!writer)
      return arrow::Status::Invalid("Failed to create writer");

    for (const auto& batch : scalar_batches) {
      ARROW_RETURN_NOT_OK(writer->write(batch));
    }
    ARROW_ASSIGN_OR_RAISE(auto cgs, writer->close());

    ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
    txn->AppendFiles(*cgs);
    ARROW_ASSIGN_OR_RAISE(auto version, txn->Commit());
    return arrow::Status::OK();
  }

  arrow::Status EnsureMilvusStorageScalarOnlyData(StorageFormatType format_type, const std::string& path) {
    auto txn_result = Transaction::Open(fs_, path);
    if (txn_result.ok()) {
      auto manifest_result = (*txn_result)->GetManifest();
      if (manifest_result.ok() && !(*manifest_result)->columnGroups().empty()) {
        return arrow::Status::OK();
      }
    }
    ARROW_RETURN_NOT_OK(DeleteTestDir(fs_, path));
    ARROW_RETURN_NOT_OK(CreateTestDir(fs_, path));
    return WriteMilvusStorageScalarOnly(format_type, path);
  }

  arrow::Status WriteLanceDatasetScalarOnly(const std::string& lance_uri,
                                            const lance::LanceStorageOptions& storage_options) {
    ARROW_ASSIGN_OR_RAISE(auto scalar_data, GetScalarOnlyData());
    auto& scalar_schema = scalar_data.first;
    auto& scalar_batches = scalar_data.second;
    ARROW_ASSIGN_OR_RAISE(auto batch_reader, arrow::RecordBatchReader::Make(scalar_batches, scalar_schema));

    ArrowArrayStream stream;
    ARROW_RETURN_NOT_OK(arrow::ExportRecordBatchReader(batch_reader, &stream));

    try {
      auto dataset = lance::BlockingDataset::WriteDataset(lance_uri, &stream, storage_options);
    } catch (const lance::LanceException& e) {
      return arrow::Status::IOError("Lance write failed: ", e.what());
    }
    return arrow::Status::OK();
  }

  arrow::Status EnsureLanceScalarOnlyData(const std::string& lance_uri,
                                          const lance::LanceStorageOptions& storage_options,
                                          const std::string& path) {
    try {
      auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);
      if (!dataset->GetAllFragmentIds().empty()) {
        return arrow::Status::OK();
      }
    } catch (...) {
    }
    ARROW_RETURN_NOT_OK(DeleteTestDir(fs_, path));
    ARROW_RETURN_NOT_OK(CreateTestDir(fs_, path));
    return WriteLanceDatasetScalarOnly(lance_uri, storage_options);
  }

  //-----------------------------------------------------------------------
  // Column Group Projection Helper
  //-----------------------------------------------------------------------
  struct ProjectionInfo {
    std::shared_ptr<std::vector<std::string>> columns;  // nullptr = all columns
    std::shared_ptr<arrow::Schema> schema;
  };

  // Build projection from column group index.
  // cg_index < 0: all columns (returns nullptr columns + full schema)
  // cg_index >= 0: columns from specified column group (requires CUSTOM_SEGMENT_PATH)
  ProjectionInfo BuildCGProjection(int cg_index, ::benchmark::State& st) {
    if (cg_index < 0) {
      return {nullptr, schema_};
    }
    auto* loader = GetDataLoader();
    if (loader->GetNumColumnGroups() == 0) {
      std::cerr << "[BENCH] Column group projection requires CUSTOM_SEGMENT_PATH; using all columns" << std::endl;
      return {nullptr, schema_};
    }
    auto columns = loader->GetColumnGroupProjection(static_cast<size_t>(cg_index));
    if (!columns) {
      st.SkipWithError(("Column group index " + std::to_string(cg_index) + " out of range [0," +
                        std::to_string(loader->GetNumColumnGroups()) + ")")
                           .c_str());
      return {nullptr, nullptr};
    }
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& name : *columns) {
      auto field = schema_->GetFieldByName(name);
      if (field)
        fields.push_back(field);
    }
    return {columns, arrow::schema(fields)};
  }

  // Read CG index from BENCH_CG_INDEX environment variable
  // Returns -1 if not set (all columns)
  static int GetCGIndexFromEnv() {
    auto* env = std::getenv("BENCH_CG_INDEX");
    if (!env || env[0] == '\0')
      return -1;
    return std::atoi(env);
  }

  std::string CGLabel(int cg_index) {
    if (cg_index < 0)
      return "";
    return "/cg" + std::to_string(cg_index);
  }

  std::shared_ptr<arrow::Schema> schema_;
};

//=============================================================================
// MilvusStorage Write + Commit Benchmark
//=============================================================================

// Args: [format_type]
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_WriteCommit)(::benchmark::State& st) {
  auto format_type = static_cast<StorageFormatType>(st.range(0));

  if (!CheckStorageFormatAvailable(st, format_type))
    return;

  ConfigureThreadPool(1);

  std::string path = GetUniquePath("ms_write");
  BENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, path), st);
  BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, path), st);

  ResetFsMetrics();

  for (auto _ : st) {
    BENCH_ASSERT_STATUS_OK(WriteMilvusStorage(format_type, path), st);
  }

  int64_t total_bytes = total_bytes_ * static_cast<int64_t>(st.iterations());
  int64_t total_rows = total_rows_ * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  ReportFsMetrics(st);
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/" + GetDataDescription());
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_WriteCommit)
    ->ArgsProduct({
        {0, 1}  // FormatType: parquet(0), vortex(1)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// MilvusStorage Open + Read Benchmark
//=============================================================================

// Args: [format_type, num_threads]
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_OpenRead)(::benchmark::State& st) {
  auto format_type = static_cast<StorageFormatType>(st.range(0));
  int num_threads = static_cast<int>(st.range(1));

  if (!CheckStorageFormatAvailable(st, format_type))
    return;

  ConfigureThreadPool(num_threads);

  std::string path = GetCachedDataPath(StorageFormatTypeName(format_type));
  BENCH_ASSERT_STATUS_OK(EnsureMilvusStorageData(format_type, path), st);

  // Collect stats once before benchmark loop
  int64_t rows_per_iter = 0;
  int64_t bytes_per_iter = 0;
  BENCH_ASSERT_STATUS_OK(ReadMilvusStorageWithStats(path, rows_per_iter, bytes_per_iter), st);

  ResetFsMetrics();

  for (auto _ : st) {
    BENCH_ASSERT_STATUS_OK(ReadMilvusStorage(path), st);
  }

  // Calculate totals using iteration count
  int64_t total_rows = rows_per_iter * static_cast<int64_t>(st.iterations());
  int64_t total_bytes = bytes_per_iter * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  ReportFsMetrics(st);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/" + GetDataDescription() + "/" +
              std::to_string(num_threads) + "T");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_OpenRead)
    ->ArgsProduct({
        {0, 1},        // FormatType: parquet(0), vortex(1)
        {1, 4, 8, 16}  // Threads: 1, 4, 8, 16
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// MilvusStorage Take Benchmark
//=============================================================================

// Args: [format_type, take_count, num_threads]
// Column group projection via env var BENCH_CG_INDEX (requires CUSTOM_SEGMENT_PATH)
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_Take)(::benchmark::State& st) {
  auto format_type = static_cast<StorageFormatType>(st.range(0));
  size_t take_count = static_cast<size_t>(st.range(1));
  int num_threads = static_cast<int>(st.range(2));

  if (!CheckStorageFormatAvailable(st, format_type))
    return;

  ConfigureThreadPool(num_threads);

  // Build projection from BENCH_CG_INDEX env var
  int cg_index = GetCGIndexFromEnv();
  auto proj = BuildCGProjection(cg_index, st);
  if (cg_index >= 0 && !proj.columns)
    return;

  std::string path = GetCachedDataPath(StorageFormatTypeName(format_type));
  BENCH_ASSERT_STATUS_OK(EnsureMilvusStorageData(format_type, path), st);

  auto indices = GenerateRandomIndices(take_count, GetLoaderNumRows());

  // Open transaction once and reuse across iterations
  BENCH_ASSERT_AND_ASSIGN(auto txn, Transaction::Open(fs_, path), st);
  BENCH_ASSERT_AND_ASSIGN(auto manifest, txn->GetManifest(), st);
  auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());

  // Create reader with projection
  auto reader = Reader::create(cgs, proj.schema, proj.columns, properties_);
  if (!reader) {
    st.SkipWithError("Failed to create reader");
    return;
  }

  ResetFsMetrics();

  bool io_trace_enabled = format_type == StorageFormatType::VORTEX && std::getenv("BENCH_IO_TRACE");
  bool io_trace_printed = false;
  for (auto _ : st) {
    // Enable IO trace for first iteration only (Vortex format, when BENCH_IO_TRACE is set)
    if (!io_trace_printed && io_trace_enabled) {
      vortex::ResetIOTrace();
    }

    auto result = reader->take(indices, num_threads);
    if (!result.ok()) {
      st.SkipWithError(result.status().message());
      return;
    }

    if (!io_trace_printed && io_trace_enabled) {
      vortex::PrintIOTrace();
      vortex::DisableIOTrace();
      io_trace_printed = true;
    }
  }

  ReportFsMetrics(st);
  st.counters["rows_taken"] = ::benchmark::Counter(static_cast<double>(take_count), ::benchmark::Counter::kDefaults);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/" + std::to_string(take_count) + "rows/" +
              std::to_string(num_threads) + "T" + CGLabel(cg_index));
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_Take)
    ->ArgsProduct({
        {0, 1},           // FormatType: parquet(0), vortex(1)
        {1, 5, 10, 100},  // Take count
        {1, 4, 8, 16}     // Threads: 1, 4, 8, 16
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// MilvusStorage Take Time Breakdown Benchmark
// Reports total decode time summed across all threads for each format.
//=============================================================================

// Args: [format_type, take_count, num_threads]
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_TakeTimeBreakdown)(::benchmark::State& st) {
  auto format_type = static_cast<StorageFormatType>(st.range(0));
  size_t take_count = static_cast<size_t>(st.range(1));
  int num_threads = static_cast<int>(st.range(2));

  if (!CheckStorageFormatAvailable(st, format_type))
    return;

  ConfigureThreadPool(num_threads);

  std::string path = GetCachedDataPath(StorageFormatTypeName(format_type));
  BENCH_ASSERT_STATUS_OK(EnsureMilvusStorageData(format_type, path), st);

  auto indices = GenerateRandomIndices(take_count, GetLoaderNumRows());

  // Open transaction once and reuse across iterations
  BENCH_ASSERT_AND_ASSIGN(auto txn, Transaction::Open(fs_, path), st);
  BENCH_ASSERT_AND_ASSIGN(auto manifest, txn->GetManifest(), st);
  auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());

  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  if (!reader) {
    st.SkipWithError("Failed to create reader");
    return;
  }

  ResetFsMetrics();
  auto cpu_before = GetProcessCpuTime();

  for (auto _ : st) {
    auto result = reader->take(indices, num_threads);
    if (!result.ok()) {
      st.SkipWithError(result.status().message());
      return;
    }
  }

  auto cpu_elapsed = GetProcessCpuTime() - cpu_before;
  ReportCpuTime(st, cpu_elapsed);

  ReportFsMetrics(st);
  st.counters["rows_taken"] = ::benchmark::Counter(static_cast<double>(take_count), ::benchmark::Counter::kDefaults);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/" + std::to_string(take_count) + "rows/" +
              std::to_string(num_threads) + "T/breakdown");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_TakeTimeBreakdown)
    ->ArgsProduct({
        {0, 1},   // FormatType: parquet(0), vortex(1)
        {1, 10},  // Take count
        {1}       // Threads: single thread only
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// MilvusStorage Take (No Transaction, Single Column Projection) Benchmark
// Projects single column by BENCH_PROJ_COL_INDEX env var (default 0)
//=============================================================================

// Args: [format_type]
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_TakeScalarOnly)(::benchmark::State& st) {
  auto format_type = static_cast<StorageFormatType>(st.range(0));
  size_t take_count = 1;
  if (auto* env = std::getenv("BENCH_TAKE_COUNT")) {
    take_count = static_cast<size_t>(std::atoi(env));
  }
  constexpr int num_threads = 1;

  if (!CheckStorageFormatAvailable(st, format_type))
    return;

  ConfigureThreadPool(num_threads);

  // Write scalar-only data (excludes FixedSizeBinary/LIST vector columns)
  std::string format_suffix = std::string(StorageFormatTypeName(format_type)) + "_scalar";
  std::string path = GetCachedDataPath(format_suffix);
  BENCH_ASSERT_STATUS_OK(EnsureMilvusStorageScalarOnlyData(format_type, path), st);

  // Pre-cache ColumnGroups (transaction happens here, outside benchmark loop)
  BENCH_ASSERT_AND_ASSIGN(auto cgs, LoadColumnGroups(path), st);

  auto indices = GenerateRandomIndices(take_count, GetLoaderNumRows());

  // Get scalar-only columns (excludes FixedSizeBinary/LIST vector columns)
  auto scalar_cols = GetScalarProjection();
  if (scalar_cols->empty()) {
    st.SkipWithError("No scalar columns available");
    return;
  }

  // Get projection column index from env (indexes into scalar columns)
  int col_index = 0;
  if (auto* env = std::getenv("BENCH_PROJ_COL_INDEX")) {
    col_index = std::atoi(env);
  }
  if (col_index < 0 || col_index >= static_cast<int>(scalar_cols->size())) {
    st.SkipWithError(("BENCH_PROJ_COL_INDEX=" + std::to_string(col_index) + " out of range [0," +
                      std::to_string(scalar_cols->size()) + ")")
                         .c_str());
    return;
  }
  auto col_name = (*scalar_cols)[col_index];
  auto proj_field = schema_->GetFieldByName(col_name);
  auto proj_schema = arrow::schema({proj_field});
  auto proj_columns = std::make_shared<std::vector<std::string>>(std::vector<std::string>{col_name});

  ResetFsMetrics();

  auto reader = Reader::create(cgs, proj_schema, proj_columns, properties_);
  if (!reader) {
    st.SkipWithError("Failed to create reader");
    return;
  }

  for (auto _ : st) {
    auto table_result = reader->take(indices, num_threads);
    if (!table_result.ok()) {
      st.SkipWithError(table_result.status().ToString().c_str());
      return;
    }
  }

  ReportFsMetrics(st);
  st.counters["rows_taken"] = ::benchmark::Counter(static_cast<double>(take_count), ::benchmark::Counter::kDefaults);
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/col=" + col_name + "/" + std::to_string(take_count) +
              "rows/1T");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_TakeScalarOnly)
    ->ArgsProduct({
        {0, 1}  // FormatType: parquet(0), vortex(1)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Lance Native Benchmarks
//=============================================================================

BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_WriteCommit)(::benchmark::State& st) {
  ConfigureThreadPool(1);

  std::string path = GetUniquePath("lance_write");
  BENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, path), st);
  BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, path), st);

  // Build Lance URI and get storage options for cloud storage support
  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  ResetFsMetrics();

  for (auto _ : st) {
    BENCH_ASSERT_STATUS_OK(WriteLanceDataset(lance_uri, storage_options), st);
  }

  int64_t total_bytes = total_bytes_ * static_cast<int64_t>(st.iterations());
  int64_t total_rows = total_rows_ * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  ReportFsMetrics(st);
  st.SetLabel("lance/" + GetDataDescription());
}

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_WriteCommit)->Unit(::benchmark::kMillisecond)->UseRealTime();

//=============================================================================
// Lance Native Open + Read Benchmark
//=============================================================================

// Args: [num_threads]
BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_OpenRead)(::benchmark::State& st) {
  int num_threads = static_cast<int>(st.range(0));

  lance::ReplaceLanceRuntime(static_cast<uint32_t>(num_threads));

  std::string path = GetCachedDataPath("lance");

  // Build Lance URI and get storage options for cloud storage support
  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  BENCH_ASSERT_STATUS_OK(EnsureLanceData(lance_uri, storage_options, path), st);

  // Open dataset once and reuse across iterations
  auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);

  // Lambda to read lance dataset using pre-opened dataset
  auto read_lance = [&](bool collect_stats, int64_t& out_rows, int64_t& out_bytes) -> arrow::Status {
    ArrowSchema c_schema;
    ARROW_RETURN_NOT_OK(arrow::ExportSchema(*schema_, &c_schema));

    auto scanner = dataset->Scan(c_schema, 8192);
    auto stream = scanner->OpenStream();

    ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ImportRecordBatchReader(&stream));

    std::shared_ptr<arrow::RecordBatch> rb;
    while (true) {
      ARROW_RETURN_NOT_OK(reader->ReadNext(&rb));
      if (!rb)
        break;
      if (collect_stats) {
        out_rows += rb->num_rows();
        out_bytes += CalculateRawDataSize(rb);
      }
    }
    return arrow::Status::OK();
  };

  // Collect stats once before benchmark loop
  int64_t rows_per_iter = 0;
  int64_t bytes_per_iter = 0;
  BENCH_ASSERT_STATUS_OK(read_lance(true, rows_per_iter, bytes_per_iter), st);

  ResetFsMetrics();
  dataset->IOStatsIncremental();  // reset Lance IO counters

  for (auto _ : st) {
    int64_t dummy_rows = 0, dummy_bytes = 0;
    BENCH_ASSERT_STATUS_OK(read_lance(false, dummy_rows, dummy_bytes), st);
  }

  auto lance_io = dataset->IOStatsIncremental();

  // Calculate totals using iteration count
  int64_t total_rows = rows_per_iter * static_cast<int64_t>(st.iterations());
  int64_t total_bytes = bytes_per_iter * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  ReportIOMetrics(st, lance_io.read_iops, lance_io.read_bytes);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel("lance/" + GetDataDescription() + "/" + std::to_string(num_threads) + "T");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_OpenRead)
    ->ArgsProduct({
        {1, 4, 8, 16}  // Threads: 1, 4, 8, 16
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Lance Native Take Benchmark
//=============================================================================

// Args: [take_count]
// Column group projection via env var BENCH_CG_INDEX (requires CUSTOM_SEGMENT_PATH)
BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_Take)(::benchmark::State& st) {
  size_t take_count = static_cast<size_t>(st.range(0));

  // Build projection from BENCH_CG_INDEX env var
  int cg_index = GetCGIndexFromEnv();
  auto proj = BuildCGProjection(cg_index, st);
  if (cg_index >= 0 && !proj.columns)
    return;

  std::string path = GetCachedDataPath("lance");

  // Build Lance URI and get storage options for cloud storage support
  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  BENCH_ASSERT_STATUS_OK(EnsureLanceData(lance_uri, storage_options, path), st);

  auto indices = GenerateRandomIndices(take_count, GetLoaderNumRows());

  // Open dataset once and reuse across iterations
  auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);

  // Lambda to take from pre-opened dataset
  auto take_lance = [&](bool collect_stats, int64_t& out_rows, int64_t& out_bytes) -> arrow::Status {
    ArrowSchema c_schema;
    ARROW_RETURN_NOT_OK(arrow::ExportSchema(*proj.schema, &c_schema));

    auto stream = dataset->Take(indices, c_schema);

    ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ImportRecordBatchReader(&stream));

    std::shared_ptr<arrow::RecordBatch> rb;
    while (true) {
      ARROW_RETURN_NOT_OK(reader->ReadNext(&rb));
      if (!rb)
        break;
      if (collect_stats) {
        out_rows += rb->num_rows();
        out_bytes += CalculateRawDataSize(rb);
      }
    }
    return arrow::Status::OK();
  };

  // Collect stats once and verify before benchmark loop
  int64_t rows_per_iter = 0;
  int64_t bytes_per_iter = 0;
  BENCH_ASSERT_STATUS_OK(take_lance(true, rows_per_iter, bytes_per_iter), st);
  BENCH_ASSERT_STATUS_OK(VerifyTakeResult("Lance", rows_per_iter, static_cast<int64_t>(take_count), bytes_per_iter),
                         st);

  ResetFsMetrics();
  dataset->IOStatsIncremental();  // reset Lance IO counters

  for (auto _ : st) {
    int64_t dummy_rows = 0, dummy_bytes = 0;
    BENCH_ASSERT_STATUS_OK(take_lance(false, dummy_rows, dummy_bytes), st);
  }

  auto lance_io = dataset->IOStatsIncremental();

  // Calculate totals using iteration count
  int64_t total_rows = rows_per_iter * static_cast<int64_t>(st.iterations());
  int64_t total_bytes = bytes_per_iter * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  ReportIOMetrics(st, lance_io.read_iops, lance_io.read_bytes);
  st.counters["rows_taken"] = ::benchmark::Counter(static_cast<double>(take_count), ::benchmark::Counter::kDefaults);
  st.SetLabel("lance/" + std::to_string(take_count) + "rows" + CGLabel(cg_index));
}

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_Take)
    ->ArgsProduct({
        {1, 5, 10, 100}  // Take count
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Lance Native Take Time Breakdown Benchmark
// Reports total decode time summed across all threads.
//=============================================================================

// Args: [take_count, num_threads]
BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_TakeTimeBreakdown)(::benchmark::State& st) {
  size_t take_count = static_cast<size_t>(st.range(0));
  int num_threads = static_cast<int>(st.range(1));

  lance::ReplaceLanceRuntime(static_cast<uint32_t>(num_threads));

  std::string path = GetCachedDataPath("lance");

  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  BENCH_ASSERT_STATUS_OK(EnsureLanceData(lance_uri, storage_options, path), st);

  auto indices = GenerateRandomIndices(take_count, GetLoaderNumRows());

  auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);

  auto take_lance = [&](bool collect_stats, int64_t& out_rows, int64_t& out_bytes) -> arrow::Status {
    ArrowSchema c_schema;
    ARROW_RETURN_NOT_OK(arrow::ExportSchema(*schema_, &c_schema));

    auto stream = dataset->Take(indices, c_schema);

    ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ImportRecordBatchReader(&stream));

    std::shared_ptr<arrow::RecordBatch> rb;
    while (true) {
      ARROW_RETURN_NOT_OK(reader->ReadNext(&rb));
      if (!rb)
        break;
      if (collect_stats) {
        out_rows += rb->num_rows();
        out_bytes += CalculateRawDataSize(rb);
      }
    }
    return arrow::Status::OK();
  };

  // Verify before benchmark loop
  int64_t rows_per_iter = 0;
  int64_t bytes_per_iter = 0;
  BENCH_ASSERT_STATUS_OK(take_lance(true, rows_per_iter, bytes_per_iter), st);
  BENCH_ASSERT_STATUS_OK(VerifyTakeResult("Lance", rows_per_iter, static_cast<int64_t>(take_count), bytes_per_iter),
                         st);

  ResetFsMetrics();
  dataset->IOStatsIncremental();  // reset Lance IO counters
  auto cpu_before = GetProcessCpuTime();

  for (auto _ : st) {
    int64_t dummy_rows = 0, dummy_bytes = 0;
    BENCH_ASSERT_STATUS_OK(take_lance(false, dummy_rows, dummy_bytes), st);
  }

  auto cpu_elapsed = GetProcessCpuTime() - cpu_before;
  auto lance_io = dataset->IOStatsIncremental();

  ReportCpuTime(st, cpu_elapsed);
  ReportIOMetrics(st, lance_io.read_iops, lance_io.read_bytes);
  st.counters["rows_taken"] = ::benchmark::Counter(static_cast<double>(take_count), ::benchmark::Counter::kDefaults);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel("lance/" + std::to_string(take_count) + "rows/" + std::to_string(num_threads) + "T/breakdown");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_TakeTimeBreakdown)
    ->ArgsProduct({
        {1, 10},  // Take count
        {1}       // Threads: single thread only
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Lance Native Take (Single Column Projection) Benchmark
// Projects single column by BENCH_PROJ_COL_INDEX env var (default 0)
//=============================================================================

BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_TakeScalarOnly)(::benchmark::State& st) {
  size_t take_count = 1;
  if (auto* env = std::getenv("BENCH_TAKE_COUNT")) {
    take_count = static_cast<size_t>(std::atoi(env));
  }
  constexpr int num_threads = 1;

  lance::ReplaceLanceRuntime(static_cast<uint32_t>(num_threads));

  // Write scalar-only data (excludes FixedSizeBinary/LIST vector columns)
  std::string path = GetCachedDataPath("lance_scalar");

  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  BENCH_ASSERT_STATUS_OK(EnsureLanceScalarOnlyData(lance_uri, storage_options, path), st);

  auto indices = GenerateRandomIndices(take_count, GetLoaderNumRows());

  // Get scalar-only columns (excludes FixedSizeBinary/LIST vector columns)
  auto scalar_cols = GetScalarProjection();
  if (scalar_cols->empty()) {
    st.SkipWithError("No scalar columns available");
    return;
  }

  // Get projection column index from env (indexes into scalar columns)
  int col_index = 0;
  if (auto* env = std::getenv("BENCH_PROJ_COL_INDEX")) {
    col_index = std::atoi(env);
  }
  if (col_index < 0 || col_index >= static_cast<int>(scalar_cols->size())) {
    st.SkipWithError(("BENCH_PROJ_COL_INDEX=" + std::to_string(col_index) + " out of range [0," +
                      std::to_string(scalar_cols->size()) + ")")
                         .c_str());
    return;
  }
  auto col_name = (*scalar_cols)[col_index];
  auto proj_field = schema_->GetFieldByName(col_name);
  auto proj_schema = arrow::schema({proj_field});

  // Open dataset once and reuse across iterations
  auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);

  auto take_lance = [&](bool collect_stats, int64_t& out_rows, int64_t& out_bytes) -> arrow::Status {
    ArrowSchema c_schema;
    ARROW_RETURN_NOT_OK(arrow::ExportSchema(*proj_schema, &c_schema));

    auto stream = dataset->Take(indices, c_schema);

    ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ImportRecordBatchReader(&stream));

    std::shared_ptr<arrow::RecordBatch> rb;
    while (true) {
      ARROW_RETURN_NOT_OK(reader->ReadNext(&rb));
      if (!rb)
        break;
      if (collect_stats) {
        out_rows += rb->num_rows();
        out_bytes += CalculateRawDataSize(rb);
      }
    }
    return arrow::Status::OK();
  };

  // Verify before benchmark loop
  int64_t rows_per_iter = 0;
  int64_t bytes_per_iter = 0;
  BENCH_ASSERT_STATUS_OK(take_lance(true, rows_per_iter, bytes_per_iter), st);
  BENCH_ASSERT_STATUS_OK(
      VerifyTakeResult("Lance(proj)", rows_per_iter, static_cast<int64_t>(take_count), bytes_per_iter), st);

  ResetFsMetrics();
  dataset->IOStatsIncremental();  // reset Lance IO counters

  for (auto _ : st) {
    int64_t dummy_rows = 0, dummy_bytes = 0;
    BENCH_ASSERT_STATUS_OK(take_lance(false, dummy_rows, dummy_bytes), st);
  }

  auto lance_io = dataset->IOStatsIncremental();

  int64_t total_rows = rows_per_iter * static_cast<int64_t>(st.iterations());
  int64_t total_bytes = bytes_per_iter * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  ReportIOMetrics(st, lance_io.read_iops, lance_io.read_bytes);
  st.counters["rows_taken"] = ::benchmark::Counter(static_cast<double>(take_count), ::benchmark::Counter::kDefaults);
  st.SetLabel("lance/col=" + col_name + "/" + std::to_string(take_count) + "rows/1T");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_TakeScalarOnly)->Unit(::benchmark::kMillisecond)->UseRealTime();

// Typical: Lance benchmarks
BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_WriteCommit)
    ->Name("Typical/Lance_Write")
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_OpenRead)
    ->Name("Typical/Lance_Read")
    ->Args({8})  // 8 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_Take)
    ->Name("Typical/Lance_Take")
    ->Args({10})  // 10 rows
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Lance Multi-Take Concurrency Benchmark
// Measures concurrent take performance to find throughput upper limit
//=============================================================================

// Args: [take_count, num_readers, skip_vector]
// Column group projection via env var BENCH_CG_INDEX (overrides skip_vector, requires CUSTOM_SEGMENT_PATH)
BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_MultiTake)(::benchmark::State& st) {
  size_t take_count = static_cast<size_t>(st.range(0));
  int num_readers = static_cast<int>(st.range(1));
  bool skip_vector = static_cast<bool>(st.range(2));

  // Build projection: BENCH_CG_INDEX overrides skip_vector when set
  int cg_index = GetCGIndexFromEnv();
  auto proj_schema = schema_;
  if (cg_index >= 0) {
    auto proj = BuildCGProjection(cg_index, st);
    if (!proj.columns)
      return;
    proj_schema = proj.schema;
  } else if (skip_vector) {
    std::vector<std::shared_ptr<arrow::Field>> scalar_fields;
    for (const auto& field : schema_->fields()) {
      if (field->type()->id() != arrow::Type::FIXED_SIZE_LIST &&
          field->type()->id() != arrow::Type::FIXED_SIZE_BINARY) {
        scalar_fields.push_back(field);
      }
    }
    proj_schema = arrow::schema(scalar_fields);
  }

  std::string path = GetCachedDataPath("lance");

  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  BENCH_ASSERT_STATUS_OK(EnsureLanceData(lance_uri, storage_options, path), st);

  std::vector<std::vector<int64_t>> per_reader_indices(num_readers);
  for (int i = 0; i < num_readers; ++i) {
    per_reader_indices[i] = GenerateRandomIndices(take_count, GetLoaderNumRows(), 42 + i);
  }

  auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);

  dataset->IOStatsIncremental();  // reset Lance IO counters

  for (auto _ : st) {
    std::vector<std::thread> reader_threads;
    std::atomic<bool> has_error{false};

    for (int i = 0; i < num_readers; ++i) {
      reader_threads.emplace_back([&, i]() {
        auto take_fn = [&]() -> arrow::Status {
          ArrowSchema c_schema;
          ARROW_RETURN_NOT_OK(arrow::ExportSchema(*proj_schema, &c_schema));

          auto stream = dataset->Take(per_reader_indices[i], c_schema);

          ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ImportRecordBatchReader(&stream));

          std::shared_ptr<arrow::RecordBatch> rb;
          while (true) {
            ARROW_RETURN_NOT_OK(reader->ReadNext(&rb));
            if (!rb)
              break;
          }
          return arrow::Status::OK();
        };

        if (!take_fn().ok()) {
          has_error = true;
        }
      });
    }

    for (auto& t : reader_threads) {
      t.join();
    }

    if (has_error) {
      st.SkipWithError("Take error in concurrent take");
      return;
    }
  }
  auto lance_io = dataset->IOStatsIncremental();
  ReportIOMetrics(st, lance_io.read_iops, lance_io.read_bytes);

  int64_t total_takes = static_cast<int64_t>(num_readers) * static_cast<int64_t>(st.iterations());
  st.counters["takes/s"] = ::benchmark::Counter(static_cast<double>(total_takes), ::benchmark::Counter::kIsRate);
  std::string proj_label;
  if (cg_index >= 0) {
    proj_label = "/cg" + std::to_string(cg_index);
  } else if (skip_vector) {
    proj_label = "/no_vec";
  }
  st.SetLabel("lance/" + std::to_string(take_count) + "rows/" + std::to_string(num_readers) + "readers" + proj_label);
}

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_MultiTake)
    ->ArgsProduct({
        {10},                        // TakeCount
        {1, 16, 64, 128, 256, 512},  // NumReaders: 1, 16, 64, 128, 256, 512
        {0, 1}                       // SkipVector: 0=all columns, 1=skip vector columns
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Multi-Take Concurrency Benchmark
// Measures concurrent take performance to find throughput upper limit
//=============================================================================

// Args: [format_type, take_count, num_readers, num_threads, skip_vector]
// Column group projection via env var BENCH_CG_INDEX (overrides skip_vector, requires CUSTOM_SEGMENT_PATH)
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_MultiTake)(::benchmark::State& st) {
  auto format_type = static_cast<StorageFormatType>(st.range(0));
  size_t take_count = static_cast<size_t>(st.range(1));
  int num_readers = static_cast<int>(st.range(2));
  int num_threads = static_cast<int>(st.range(3));
  bool skip_vector = static_cast<bool>(st.range(4));

  if (!CheckStorageFormatAvailable(st, format_type))
    return;

  // Build needed_columns: BENCH_CG_INDEX overrides skip_vector when set
  int cg_index = GetCGIndexFromEnv();
  std::shared_ptr<std::vector<std::string>> needed_columns;
  if (cg_index >= 0) {
    auto proj = BuildCGProjection(cg_index, st);
    if (!proj.columns)
      return;
    needed_columns = proj.columns;
  } else if (skip_vector) {
    needed_columns = std::make_shared<std::vector<std::string>>();
    for (const auto& field : schema_->fields()) {
      if (field->type()->id() != arrow::Type::FIXED_SIZE_LIST &&
          field->type()->id() != arrow::Type::FIXED_SIZE_BINARY) {
        needed_columns->push_back(field->name());
      }
    }
  }

  ConfigureThreadPool(num_threads);
  ResetFsMetrics();

  std::string path = GetCachedDataPath(StorageFormatTypeName(format_type));
  BENCH_ASSERT_STATUS_OK(EnsureMilvusStorageData(format_type, path), st);

  // Each reader gets its own random indices (different seed per reader)
  std::vector<std::vector<int64_t>> per_reader_indices(num_readers);
  for (int i = 0; i < num_readers; ++i) {
    per_reader_indices[i] = GenerateRandomIndices(take_count, GetLoaderNumRows(), 42 + i);
  }

  BENCH_ASSERT_AND_ASSIGN(auto txn, Transaction::Open(fs_, path), st);
  BENCH_ASSERT_AND_ASSIGN(auto manifest, txn->GetManifest(), st);
  auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());
  auto reader = Reader::create(cgs, schema_, needed_columns, properties_);
  if (!reader) {
    st.SkipWithError("Failed to create reader");
    return;
  }

  for (auto _ : st) {
    std::vector<std::thread> reader_threads;
    std::atomic<bool> has_error{false};

    for (int i = 0; i < num_readers; ++i) {
      reader_threads.emplace_back([&, i]() {
        auto take_fn = [&]() -> arrow::Status {
          auto result = reader->take(per_reader_indices[i], num_threads);
          if (!result.ok()) {
            return result.status();
          }
          return arrow::Status::OK();
        };

        if (!take_fn().ok()) {
          has_error = true;
        }
      });
    }

    for (auto& t : reader_threads) {
      t.join();
    }

    if (has_error) {
      st.SkipWithError("Take error in concurrent take");
      return;
    }
  }

  int64_t total_takes = static_cast<int64_t>(num_readers) * static_cast<int64_t>(st.iterations());
  st.counters["takes/s"] = ::benchmark::Counter(static_cast<double>(total_takes), ::benchmark::Counter::kIsRate);
  ReportFsMetrics(st);
  std::string proj_label;
  if (cg_index >= 0) {
    proj_label = "/cg" + std::to_string(cg_index);
  } else if (skip_vector) {
    proj_label = "/no_vec";
  }
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/" + std::to_string(take_count) + "rows/" +
              std::to_string(num_readers) + "readers/" + std::to_string(num_threads) + "pool" + proj_label);
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_MultiTake)
    ->ArgsProduct({
        {0, 1},                      // FormatType: parquet(0), vortex(1)
        {10},                        // TakeCount
        {1, 16, 64, 128, 256, 512},  // NumReaders: 1, 16, 64, 128, 256, 512
        {1, 16},                     // NumThreads: 1, 16
        {0, 1}                       // SkipVector: 0=all columns, 1=skip vector columns
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Typical Benchmarks (Quick validation with representative parameters)
// Run with: --benchmark_filter="Typical/"
//=============================================================================

// Typical: MilvusStorage Parquet
BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_WriteCommit)
    ->Name("Typical/MilvusStorage_Write_Parquet")
    ->Args({0})  // Parquet
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_OpenRead)
    ->Name("Typical/MilvusStorage_Read_Parquet_1T")
    ->Args({0, 1})  // Parquet + 1 thread
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_OpenRead)
    ->Name("Typical/MilvusStorage_Read_Parquet_8T")
    ->Args({0, 8})  // Parquet + 8 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_Take)
    ->Name("Typical/MilvusStorage_Take_Parquet_1T")
    ->Args({0, 10, 1})  // Parquet + 10 rows + 1 thread
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_Take)
    ->Name("Typical/MilvusStorage_Take_Parquet")
    ->Args({0, 10, 8})  // Parquet + 10 rows + 8 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

// Typical: MilvusStorage Vortex
BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_WriteCommit)
    ->Name("Typical/MilvusStorage_Write_Vortex")
    ->Args({1})  // Vortex
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_OpenRead)
    ->Name("Typical/MilvusStorage_Read_Vortex_1T")
    ->Args({1, 1})  // Vortex + 1 thread
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_OpenRead)
    ->Name("Typical/MilvusStorage_Read_Vortex_8T")
    ->Args({1, 8})  // Vortex + 8 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_Take)
    ->Name("Typical/MilvusStorage_Take_Vortex_1T")
    ->Args({1, 10, 1})  // Vortex + 10 rows + 1 thread
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_Take)
    ->Name("Typical/MilvusStorage_Take_Vortex_8T")
    ->Args({1, 10, 8})  // Vortex + 10 rows + 8 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

}  // namespace benchmark
}  // namespace milvus_storage
