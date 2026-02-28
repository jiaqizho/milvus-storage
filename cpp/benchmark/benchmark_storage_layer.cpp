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
#include <arrow/table.h>
#include <arrow/record_batch.h>

#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/thread_pool.h"

#include <arrow/c/bridge.h>
#include "format/bridge/rust/include/lance_bridge.h"
#include "milvus-storage/format/lance/lance_common.h"

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

class StorageLayerFixture : public FormatBenchFixtureBase<> {
  public:
  void SetUp(::benchmark::State& st) override {
    FormatBenchFixtureBase<>::SetUp(st);

    // Get schema from data loader
    schema_ = GetLoaderSchema();

    // Batches are loaded lazily via EnsureBatchesLoaded() — only needed by write benchmarks
    // and cache-miss paths. This avoids re-reading source data for every read benchmark.
  }

  void TearDown(::benchmark::State& st) override {
    // Clear pre-loaded batches to release memory
    batches_.clear();
    batches_.shrink_to_fit();
    schema_.reset();
    ThreadPoolHolder::Release();
    FormatBenchFixtureBase<>::TearDown(st);
  }

  void ConfigureThreadPool(int num_threads) { ThreadPoolHolder::WithSingleton(num_threads); }

  protected:
  //-----------------------------------------------------------------------
  // Lazy batch loading: only load source data when actually needed for writing
  //-----------------------------------------------------------------------
  arrow::Status EnsureBatchesLoaded() {
    if (!batches_.empty()) {
      return arrow::Status::OK();
    }
    ARROW_ASSIGN_OR_RAISE(auto batch_reader, GetLoaderBatchReader());
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
      if (!batch)
        break;
      batches_.push_back(batch);
      total_bytes_ += CalculateRawDataSize(batch);
      total_rows_ += batch->num_rows();
    }
    return arrow::Status::OK();
  }

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
  // MilvusStorage: Open Transaction + Take (with stats collection)
  //-----------------------------------------------------------------------
  arrow::Status TakeMilvusStorageWithStats(const std::string& path,
                                           const std::vector<int64_t>& indices,
                                           int64_t& out_rows,
                                           int64_t& out_bytes) {
    ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
    ARROW_ASSIGN_OR_RAISE(auto manifest, txn->GetManifest());
    auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());

    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    if (!reader)
      return arrow::Status::Invalid("Failed to create reader");

    ARROW_ASSIGN_OR_RAISE(auto table, reader->take(indices));
    out_rows += table->num_rows();
    for (const auto& batch : arrow::TableBatchReader(*table)) {
      if (!batch.ok())
        return batch.status();
      out_bytes += CalculateRawDataSize(*batch);
    }
    return arrow::Status::OK();
  }

  //-----------------------------------------------------------------------
  // MilvusStorage: Open Transaction + Take (no stats, for benchmark loop)
  //-----------------------------------------------------------------------
  arrow::Status TakeMilvusStorage(const std::string& path, const std::vector<int64_t>& indices) {
    ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
    ARROW_ASSIGN_OR_RAISE(auto manifest, txn->GetManifest());
    auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());

    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    if (!reader)
      return arrow::Status::Invalid("Failed to create reader");

    ARROW_ASSIGN_OR_RAISE(auto table, reader->take(indices));
    // No stats collection in benchmark loop
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

  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  int64_t total_bytes_ = 0;
  int64_t total_rows_ = 0;
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
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_Take)(::benchmark::State& st) {
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

  // Create reader once outside benchmark loop
  auto reader = Reader::create(cgs, schema_, nullptr, properties_);
  if (!reader) {
    st.SkipWithError("Failed to create reader");
    return;
  }

  ResetFsMetrics();

  for (auto _ : st) {
    auto result = reader->take(indices);
    if (!result.ok()) {
      st.SkipWithError(result.status().message());
      return;
    }
  }

  ReportFsMetrics(st);
  st.counters["rows_taken"] = ::benchmark::Counter(static_cast<double>(take_count), ::benchmark::Counter::kDefaults);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/" + std::to_string(take_count) + "rows/" +
              std::to_string(num_threads) + "T");
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
    auto table_result = reader->take(indices);
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

// Args: [take_count, num_threads]
BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_Take)(::benchmark::State& st) {
  size_t take_count = static_cast<size_t>(st.range(0));
  int num_threads = static_cast<int>(st.range(1));

  lance::ReplaceLanceRuntime(static_cast<uint32_t>(num_threads));

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
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel("lance/" + std::to_string(take_count) + "rows/" + std::to_string(num_threads) + "T");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_Take)
    ->ArgsProduct({
        {1, 5, 10, 100},  // Take count
        {1, 4, 8, 16}     // Threads: 1, 4, 8, 16
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
    ->Args({10, 8})  // 10 rows + 8 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Lance Multi-Reader Concurrency Benchmark
//=============================================================================

// Args: [num_readers, thread_pool_size]
BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_MultiReader)(::benchmark::State& st) {
  int num_readers = static_cast<int>(st.range(0));
  int thread_pool_size = static_cast<int>(st.range(1));

  lance::ReplaceLanceRuntime(static_cast<uint32_t>(thread_pool_size));

  std::string path = GetCachedDataPath("lance");

  // Build Lance URI and get storage options for cloud storage support
  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  BENCH_ASSERT_STATUS_OK(EnsureLanceData(lance_uri, storage_options, path), st);

  // Start thread tracker
  ThreadTracker thread_tracker;
  thread_tracker.Start(std::chrono::milliseconds(1));

  int64_t total_rows = 0;
  int64_t total_bytes = 0;

  ResetFsMetrics();

  for (auto _ : st) {
    std::vector<std::thread> reader_threads;
    std::atomic<int64_t> rows_read{0};
    std::atomic<int64_t> bytes_read{0};
    std::atomic<bool> has_error{false};

    // Launch N reader threads
    for (int i = 0; i < num_readers; ++i) {
      reader_threads.emplace_back([&, i]() {
        auto read_all = [&]() -> arrow::Status {
          auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);

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
            rows_read += rb->num_rows();
            bytes_read += CalculateRawDataSize(rb);
          }
          return arrow::Status::OK();
        };

        if (!read_all().ok()) {
          has_error = true;
        }
      });
    }

    // Wait for all readers to complete
    for (auto& t : reader_threads) {
      t.join();
    }

    if (has_error) {
      st.SkipWithError("Reader error in concurrent read");
      return;
    }

    total_rows += rows_read.load();
    total_bytes += bytes_read.load();
  }

  thread_tracker.Stop();

  ReportThroughput(st, total_bytes, total_rows);
  ReportFsMetrics(st);
  thread_tracker.ReportToState(st);
  st.counters["num_readers"] = ::benchmark::Counter(static_cast<double>(num_readers), ::benchmark::Counter::kDefaults);
  st.counters["pool_size"] =
      ::benchmark::Counter(static_cast<double>(thread_pool_size), ::benchmark::Counter::kDefaults);
  st.SetLabel("lance/" + std::to_string(num_readers) + "readers/" + std::to_string(thread_pool_size) + "pool");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_MultiReader)
    ->ArgsProduct({
        {1, 16, 64, 256},  // NumReaders: 1, 16, 64, 256
        {1, 8, 16, 32}     // ThreadPoolSize: 1, 8, 16, 32
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Multi-Reader Concurrency Benchmark
// Measures memory usage and thread count with N concurrent readers
//=============================================================================

// Args: [format_type, num_readers, thread_pool_size]
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_MultiReader)(::benchmark::State& st) {
  auto format_type = static_cast<StorageFormatType>(st.range(0));
  int num_readers = static_cast<int>(st.range(1));
  int thread_pool_size = static_cast<int>(st.range(2));

  if (!CheckStorageFormatAvailable(st, format_type))
    return;

  ConfigureThreadPool(thread_pool_size);

  std::string path = GetCachedDataPath(StorageFormatTypeName(format_type));
  BENCH_ASSERT_STATUS_OK(EnsureMilvusStorageData(format_type, path), st);

  // Start thread tracker
  ThreadTracker thread_tracker;
  thread_tracker.Start(std::chrono::milliseconds(1));

  int64_t total_rows = 0;
  int64_t total_bytes = 0;

  ResetFsMetrics();

  for (auto _ : st) {
    std::vector<std::thread> reader_threads;
    std::atomic<int64_t> rows_read{0};
    std::atomic<int64_t> bytes_read{0};
    std::atomic<bool> has_error{false};

    // Launch N reader threads, each opens its own transaction
    for (int i = 0; i < num_readers; ++i) {
      reader_threads.emplace_back([&, i]() {
        auto read_all = [&]() -> arrow::Status {
          ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
          ARROW_ASSIGN_OR_RAISE(auto manifest, txn->GetManifest());
          auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());

          auto reader = Reader::create(cgs, schema_, nullptr, properties_);
          if (!reader)
            return arrow::Status::Invalid("Failed to create reader");

          ARROW_ASSIGN_OR_RAISE(auto batch_reader, reader->get_record_batch_reader());

          std::shared_ptr<arrow::RecordBatch> rb;
          while (true) {
            ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&rb));
            if (!rb)
              break;
            rows_read += rb->num_rows();
            bytes_read += CalculateRawDataSize(rb);
          }
          return arrow::Status::OK();
        };

        if (!read_all().ok()) {
          has_error = true;
        }
      });
    }

    // Wait for all readers to complete
    for (auto& t : reader_threads) {
      t.join();
    }

    if (has_error) {
      st.SkipWithError("Reader error in concurrent read");
      return;
    }

    total_rows += rows_read.load();
    total_bytes += bytes_read.load();
  }

  thread_tracker.Stop();

  ReportThroughput(st, total_bytes, total_rows);
  ReportFsMetrics(st);
  thread_tracker.ReportToState(st);
  st.counters["num_readers"] = ::benchmark::Counter(static_cast<double>(num_readers), ::benchmark::Counter::kDefaults);
  st.counters["pool_size"] =
      ::benchmark::Counter(static_cast<double>(thread_pool_size), ::benchmark::Counter::kDefaults);
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/" + std::to_string(num_readers) + "readers/" +
              std::to_string(thread_pool_size) + "pool");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_MultiReader)
    ->ArgsProduct({
        {0, 1},            // FormatType: parquet(0), vortex(1)
        {1, 16, 64, 256},  // NumReaders: 1, 16, 64, 256
        {1, 8, 16, 32}     // ThreadPoolSize: 1, 8, 16, 32
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
