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

class StorageLayerFixture : public FormatBenchFixture {
  public:
  void SetUp(::benchmark::State& st) override {
    FormatBenchFixture::SetUp(st);
    BENCH_ASSERT_AND_ASSIGN(schema_, CreateSchema(), st);
    // Thread pool will be configured per-benchmark
  }

  void TearDown(::benchmark::State& st) override {
    ThreadPoolHolder::Release();
    FormatBenchFixture::TearDown(st);
  }

  void ConfigureThreadPool(int num_threads) { ThreadPoolHolder::WithSingleton(num_threads); }

  protected:
  //-----------------------------------------------------------------------
  // MilvusStorage: Write + Transaction Commit
  //-----------------------------------------------------------------------
  arrow::Status WriteMilvusStorage(StorageFormatType format_type,
                                   const std::shared_ptr<arrow::RecordBatch>& batch,
                                   const std::string& path) {
    if (format_type == StorageFormatType::MIXED) {
      // Write scalar columns (id, name, value) with Parquet
      ARROW_ASSIGN_OR_RAISE(auto scalar_schema, CreateTestSchema({true, true, true, false}));
      ARROW_ASSIGN_OR_RAISE(auto scalar_policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, scalar_schema));
      auto scalar_writer = Writer::create(path, scalar_schema, std::move(scalar_policy), properties_);
      if (!scalar_writer)
        return arrow::Status::Invalid("Failed to create scalar writer");

      ARROW_RETURN_NOT_OK(scalar_writer->write(ProjectColumns(batch, {0, 1, 2})));
      ARROW_ASSIGN_OR_RAISE(auto scalar_cgs, scalar_writer->close());

      // Write vector column with Vortex
      ARROW_ASSIGN_OR_RAISE(auto vector_schema, CreateTestSchema({false, false, false, true}));
      ARROW_ASSIGN_OR_RAISE(auto vector_policy, CreateSinglePolicy(LOON_FORMAT_VORTEX, vector_schema));
      auto vector_writer = Writer::create(path, vector_schema, std::move(vector_policy), properties_);
      if (!vector_writer)
        return arrow::Status::Invalid("Failed to create vector writer");

      ARROW_RETURN_NOT_OK(vector_writer->write(ProjectColumns(batch, {3})));
      ARROW_ASSIGN_OR_RAISE(auto vector_cgs, vector_writer->close());

      // combine column groups
      for (const auto& cg : *vector_cgs) {
        scalar_cgs->emplace_back(cg);
      }

      // Combine via Transaction
      ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
      txn->AppendFiles(*scalar_cgs);
      ARROW_ASSIGN_OR_RAISE(auto version, txn->Commit());
    } else {
      std::string format = (format_type == StorageFormatType::PARQUET) ? LOON_FORMAT_PARQUET : LOON_FORMAT_VORTEX;
      ARROW_ASSIGN_OR_RAISE(auto policy, CreateSinglePolicy(format, schema_));
      auto writer = Writer::create(path, schema_, std::move(policy), properties_);
      if (!writer)
        return arrow::Status::Invalid("Failed to create writer");

      ARROW_RETURN_NOT_OK(writer->write(batch));
      ARROW_ASSIGN_OR_RAISE(auto cgs, writer->close());

      ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
      txn->AppendFiles(*cgs);
      ARROW_ASSIGN_OR_RAISE(auto version, txn->Commit());
    }
    return arrow::Status::OK();
  }

  //-----------------------------------------------------------------------
  // MilvusStorage: Write Only (No Transaction)
  //-----------------------------------------------------------------------
  arrow::Status WriteMilvusStorageNoTxn(StorageFormatType format_type,
                                        const std::shared_ptr<arrow::RecordBatch>& batch,
                                        const std::string& path) {
    if (format_type == StorageFormatType::MIXED) {
      // Write scalar columns (id, name, value) with Parquet
      ARROW_ASSIGN_OR_RAISE(auto scalar_schema, CreateTestSchema({true, true, true, false}));
      ARROW_ASSIGN_OR_RAISE(auto scalar_policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, scalar_schema));
      auto scalar_writer = Writer::create(path, scalar_schema, std::move(scalar_policy), properties_);
      if (!scalar_writer)
        return arrow::Status::Invalid("Failed to create scalar writer");

      ARROW_RETURN_NOT_OK(scalar_writer->write(ProjectColumns(batch, {0, 1, 2})));
      ARROW_ASSIGN_OR_RAISE(auto scalar_cgs, scalar_writer->close());

      // Write vector column with Vortex
      ARROW_ASSIGN_OR_RAISE(auto vector_schema, CreateTestSchema({false, false, false, true}));
      ARROW_ASSIGN_OR_RAISE(auto vector_policy, CreateSinglePolicy(LOON_FORMAT_VORTEX, vector_schema));
      auto vector_writer = Writer::create(path, vector_schema, std::move(vector_policy), properties_);
      if (!vector_writer)
        return arrow::Status::Invalid("Failed to create vector writer");

      ARROW_RETURN_NOT_OK(vector_writer->write(ProjectColumns(batch, {3})));
      ARROW_ASSIGN_OR_RAISE(auto vector_cgs, vector_writer->close());
      // No transaction commit
    } else {
      std::string format = (format_type == StorageFormatType::PARQUET) ? LOON_FORMAT_PARQUET : LOON_FORMAT_VORTEX;
      ARROW_ASSIGN_OR_RAISE(auto policy, CreateSinglePolicy(format, schema_));
      auto writer = Writer::create(path, schema_, std::move(policy), properties_);
      if (!writer)
        return arrow::Status::Invalid("Failed to create writer");

      ARROW_RETURN_NOT_OK(writer->write(batch));
      ARROW_ASSIGN_OR_RAISE(auto cgs, writer->close());
      // No transaction commit
    }
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
    for (int i = 0; i < table->num_columns(); ++i) {
      for (const auto& chunk : table->column(i)->chunks()) {
        for (const auto& buffer : chunk->data()->buffers) {
          if (buffer)
            out_bytes += buffer->size();
        }
      }
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

  std::shared_ptr<arrow::Schema> schema_;
};

//=============================================================================
// MilvusStorage Write + Commit Benchmark
//=============================================================================

// Args: [format_type, data_config_idx]
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_WriteCommit)(::benchmark::State& st) {
  auto format_type = static_cast<StorageFormatType>(st.range(0));
  size_t data_config_idx = static_cast<size_t>(st.range(1));

  if (!CheckStorageFormatAvailable(st, format_type))
    return;

  ConfigureThreadPool(1);

  DataSizeConfig data_config = DataSizeConfig::FromIndex(data_config_idx);
  BENCH_ASSERT_AND_ASSIGN(auto batch, CreateBatch(schema_, data_config, true), st);
  int64_t raw_data_size = CalculateRawDataSize(batch);

  int64_t total_bytes = 0;
  int64_t total_rows = 0;

  std::string path = GetUniquePath("ms_write");

  BENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, path), st);
  BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, path), st);
  for (auto _ : st) {
    BENCH_ASSERT_STATUS_OK(WriteMilvusStorage(format_type, batch, path), st);
    total_bytes += raw_data_size;
    total_rows += batch->num_rows();
  }

  ReportThroughput(st, total_bytes, total_rows);
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/" + data_config.name);
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_WriteCommit)
    ->ArgsProduct({
        {0, 1, 2},  // FormatType: parquet(0), vortex(1), mixed(2)
        {0, 1, 2}   // DataConfig: Small(0), Medium(1), Large(2)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// MilvusStorage Write Only (No Transaction) Benchmark
//=============================================================================

// Args: [format_type, data_config_idx]
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_WriteOnly)(::benchmark::State& st) {
  auto format_type = static_cast<StorageFormatType>(st.range(0));
  size_t data_config_idx = static_cast<size_t>(st.range(1));

  if (!CheckStorageFormatAvailable(st, format_type))
    return;

  ConfigureThreadPool(1);

  DataSizeConfig data_config = DataSizeConfig::FromIndex(data_config_idx);
  BENCH_ASSERT_AND_ASSIGN(auto batch, CreateBatch(schema_, data_config, true), st);
  int64_t raw_data_size = CalculateRawDataSize(batch);

  int64_t total_bytes = 0;
  int64_t total_rows = 0;

  std::string path = GetUniquePath("ms_write_only");

  BENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, path), st);
  BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, path), st);
  for (auto _ : st) {
    BENCH_ASSERT_STATUS_OK(WriteMilvusStorageNoTxn(format_type, batch, path), st);
    total_bytes += raw_data_size;
    total_rows += batch->num_rows();
  }

  ReportThroughput(st, total_bytes, total_rows);
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/" + data_config.name + "/no-txn");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_WriteOnly)
    ->ArgsProduct({
        {0, 1, 2},  // FormatType: parquet(0), vortex(1), mixed(2)
        {0, 1, 2}   // DataConfig: Small(0), Medium(1), Large(2)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// MilvusStorage Open + Read Benchmark
//=============================================================================

// Args: [format_type, data_config_idx, num_threads]
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_OpenRead)(::benchmark::State& st) {
  auto format_type = static_cast<StorageFormatType>(st.range(0));
  size_t data_config_idx = static_cast<size_t>(st.range(1));
  int num_threads = static_cast<int>(st.range(2));

  if (!CheckStorageFormatAvailable(st, format_type))
    return;

  ConfigureThreadPool(num_threads);

  DataSizeConfig data_config = DataSizeConfig::FromIndex(data_config_idx);
  BENCH_ASSERT_AND_ASSIGN(auto batch, CreateBatch(schema_, data_config, true), st);

  std::string path = GetUniquePath("ms_read");
  BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, path), st);
  BENCH_ASSERT_STATUS_OK(WriteMilvusStorage(format_type, batch, path), st);

  // Collect stats once before benchmark loop
  int64_t rows_per_iter = 0;
  int64_t bytes_per_iter = 0;
  BENCH_ASSERT_STATUS_OK(ReadMilvusStorageWithStats(path, rows_per_iter, bytes_per_iter), st);

  for (auto _ : st) {
    BENCH_ASSERT_STATUS_OK(ReadMilvusStorage(path), st);
  }

  // Calculate totals using iteration count
  int64_t total_rows = rows_per_iter * static_cast<int64_t>(st.iterations());
  int64_t total_bytes = bytes_per_iter * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/" + data_config.name + "/" +
              std::to_string(num_threads) + "T");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_OpenRead)
    ->ArgsProduct({
        {0, 1, 2},     // FormatType: parquet(0), vortex(1), mixed(2)
        {0, 1, 2},     // DataConfig: Small(0), Medium(1), Large(2)
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

  DataSizeConfig data_config = DataSizeConfig::Large();  // Use Large (409K rows) for Take benchmark
  BENCH_ASSERT_AND_ASSIGN(auto batch, CreateBatch(schema_, data_config, true), st);

  std::string path = GetUniquePath("ms_take");
  BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, path), st);
  BENCH_ASSERT_STATUS_OK(WriteMilvusStorage(format_type, batch, path), st);

  auto indices = GenerateRandomIndices(take_count, static_cast<int64_t>(data_config.num_rows));

  // Collect stats once before benchmark loop
  int64_t rows_per_iter = 0;
  int64_t bytes_per_iter = 0;
  BENCH_ASSERT_STATUS_OK(TakeMilvusStorageWithStats(path, indices, rows_per_iter, bytes_per_iter), st);

  for (auto _ : st) {
    BENCH_ASSERT_STATUS_OK(TakeMilvusStorage(path, indices), st);
  }

  // Calculate totals using iteration count
  int64_t total_rows = rows_per_iter * static_cast<int64_t>(st.iterations());
  int64_t total_bytes = bytes_per_iter * static_cast<int64_t>(st.iterations());

  ReportThroughput(st, total_bytes, total_rows);
  st.counters["rows_taken"] = ::benchmark::Counter(static_cast<double>(take_count), ::benchmark::Counter::kDefaults);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel(std::string(StorageFormatTypeName(format_type)) + "/" + std::to_string(take_count) + "rows/" +
              std::to_string(num_threads) + "T");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_Take)
    ->ArgsProduct({
        {0, 1, 2},              // FormatType: parquet(0), vortex(1), mixed(2)
        {100, 200, 500, 1000},  // Take count
        {1, 4, 8, 16}           // Threads: 1, 4, 8, 16
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Lance Native Benchmarks
//=============================================================================

// Helper: write test data to a lance dataset
static arrow::Status WriteLanceDataset(const std::shared_ptr<arrow::Schema>& schema,
                                       const std::shared_ptr<arrow::RecordBatch>& batch,
                                       const std::string& lance_uri,
                                       const lance::LanceStorageOptions& storage_options) {
  // Create a RecordBatchReader from the batch
  ARROW_ASSIGN_OR_RAISE(auto batch_reader, arrow::RecordBatchReader::Make({batch}, schema));

  ArrowArrayStream stream;
  ARROW_RETURN_NOT_OK(arrow::ExportRecordBatchReader(batch_reader, &stream));

  try {
    auto dataset = lance::BlockingDataset::WriteDataset(lance_uri, &stream, storage_options);
  } catch (const lance::LanceException& e) {
    return arrow::Status::IOError("Lance write failed: ", e.what());
  }
  return arrow::Status::OK();
}

//=============================================================================
// Lance Native Write + Commit Benchmark
//=============================================================================

// Args: [data_config_idx]
BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_WriteCommit)(::benchmark::State& st) {
  size_t data_config_idx = static_cast<size_t>(st.range(0));
  DataSizeConfig data_config = DataSizeConfig::FromIndex(data_config_idx);

  ConfigureThreadPool(1);

  BENCH_ASSERT_AND_ASSIGN(auto batch, CreateBatch(schema_, data_config, true), st);
  int64_t raw_data_size = CalculateRawDataSize(batch);

  int64_t total_bytes = 0;
  int64_t total_rows = 0;

  std::string path = GetUniquePath("lance_write");
  BENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, path), st);
  BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, path), st);

  // Build Lance URI and get storage options for cloud storage support
  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  for (auto _ : st) {
    BENCH_ASSERT_STATUS_OK(WriteLanceDataset(schema_, batch, lance_uri, storage_options), st);
    total_bytes += raw_data_size;
    total_rows += batch->num_rows();
  }

  ReportThroughput(st, total_bytes, total_rows);
  st.SetLabel("lance/" + data_config.name);
}

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_WriteCommit)
    ->Args({0})  // Small
    ->Args({1})  // Medium
    ->Args({2})  // Large
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Lance Native Open + Read Benchmark
//=============================================================================

// Args: [data_config_idx, num_threads]
BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_OpenRead)(::benchmark::State& st) {
  size_t data_config_idx = static_cast<size_t>(st.range(0));
  int num_threads = static_cast<int>(st.range(1));
  DataSizeConfig data_config = DataSizeConfig::FromIndex(data_config_idx);

  lance::ReplaceLanceRuntime(static_cast<uint32_t>(num_threads));

  BENCH_ASSERT_AND_ASSIGN(auto batch, CreateBatch(schema_, data_config, true), st);

  std::string path = GetUniquePath("lance_read");
  BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, path), st);

  // Build Lance URI and get storage options for cloud storage support
  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  BENCH_ASSERT_STATUS_OK(WriteLanceDataset(schema_, batch, lance_uri, storage_options), st);

  // Lambda to read lance dataset
  auto read_lance = [&](bool collect_stats, int64_t& out_rows, int64_t& out_bytes) -> arrow::Status {
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

  for (auto _ : st) {
    int64_t dummy_rows = 0, dummy_bytes = 0;
    BENCH_ASSERT_STATUS_OK(read_lance(false, dummy_rows, dummy_bytes), st);
  }

  // Calculate totals using iteration count
  int64_t total_rows = rows_per_iter * static_cast<int64_t>(st.iterations());
  int64_t total_bytes = bytes_per_iter * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel("lance/" + data_config.name + "/" + std::to_string(num_threads) + "T");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_OpenRead)
    ->ArgsProduct({
        {0, 1, 2},     // DataConfig: Small(0), Medium(1), Large(2)
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
  DataSizeConfig data_config = DataSizeConfig::Large();  // Use Large (409K rows) for Take benchmark

  lance::ReplaceLanceRuntime(static_cast<uint32_t>(num_threads));

  BENCH_ASSERT_AND_ASSIGN(auto batch, CreateBatch(schema_, data_config, true), st);

  std::string path = GetUniquePath("lance_take");
  BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, path), st);

  // Build Lance URI and get storage options for cloud storage support
  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  BENCH_ASSERT_STATUS_OK(WriteLanceDataset(schema_, batch, lance_uri, storage_options), st);

  auto indices = GenerateRandomIndices(take_count, static_cast<int64_t>(data_config.num_rows));

  // Lambda to take from lance dataset
  auto take_lance = [&](bool collect_stats, int64_t& out_rows, int64_t& out_bytes) -> arrow::Status {
    auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);

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

  // Collect stats once before benchmark loop
  int64_t rows_per_iter = 0;
  int64_t bytes_per_iter = 0;
  BENCH_ASSERT_STATUS_OK(take_lance(true, rows_per_iter, bytes_per_iter), st);

  for (auto _ : st) {
    int64_t dummy_rows = 0, dummy_bytes = 0;
    BENCH_ASSERT_STATUS_OK(take_lance(false, dummy_rows, dummy_bytes), st);
  }

  // Calculate totals using iteration count
  int64_t total_rows = rows_per_iter * static_cast<int64_t>(st.iterations());
  int64_t total_bytes = bytes_per_iter * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  st.counters["rows_taken"] = ::benchmark::Counter(static_cast<double>(take_count), ::benchmark::Counter::kDefaults);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel("lance/" + std::to_string(take_count) + "rows/" + std::to_string(num_threads) + "T");
}

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_Take)
    ->ArgsProduct({
        {100, 200, 500, 1000},  // Take count
        {1, 4, 8, 16}           // Threads: 1, 4, 8, 16
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

// Typical: Lance benchmarks
BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_WriteCommit)
    ->Name("Typical/Lance_Write")
    ->Args({1})  // Medium
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_OpenRead)
    ->Name("Typical/Lance_Read")
    ->Args({1, 8})  // Medium + 8 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_Take)
    ->Name("Typical/Lance_Take")
    ->Args({1000, 8})  // 1000 rows + 8 threads
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

  DataSizeConfig data_config = DataSizeConfig::Medium();
  BENCH_ASSERT_AND_ASSIGN(auto batch, CreateBatch(schema_, data_config, true), st);

  std::string path = GetUniquePath("lance_multi_reader");
  BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, path), st);

  // Build Lance URI and get storage options for cloud storage support
  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  BENCH_ASSERT_STATUS_OK(WriteLanceDataset(schema_, batch, lance_uri, storage_options), st);

  // Start thread tracker
  ThreadTracker thread_tracker;
  thread_tracker.Start(std::chrono::milliseconds(1));

  int64_t total_rows = 0;
  int64_t total_bytes = 0;

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

  DataSizeConfig data_config = DataSizeConfig::Medium();
  BENCH_ASSERT_AND_ASSIGN(auto batch, CreateBatch(schema_, data_config, true), st);

  std::string path = GetUniquePath("ms_multi_reader");
  BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, path), st);
  BENCH_ASSERT_STATUS_OK(WriteMilvusStorage(format_type, batch, path), st);

  // Start thread tracker
  ThreadTracker thread_tracker;
  thread_tracker.Start(std::chrono::milliseconds(1));

  int64_t total_rows = 0;
  int64_t total_bytes = 0;

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
    ->Args({0, 1})  // Parquet + Medium
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_OpenRead)
    ->Name("Typical/MilvusStorage_Read_Parquet_1T")
    ->Args({0, 1, 1})  // Parquet + Medium + 1 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_OpenRead)
    ->Name("Typical/MilvusStorage_Read_Parquet_8T")
    ->Args({0, 1, 8})  // Parquet + Medium + 8 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_Take)
    ->Name("Typical/MilvusStorage_Take_Parquet_1T")
    ->Args({0, 1000, 1})  // Parquet + 1000 rows + 1 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_Take)
    ->Name("Typical/MilvusStorage_Take_Parquet")
    ->Args({0, 1000, 8})  // Parquet + 1000 rows + 8 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

// Typical: MilvusStorage Vortex
BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_WriteCommit)
    ->Name("Typical/MilvusStorage_Write_Vortex")
    ->Args({1, 1})  // Vortex + Medium
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_OpenRead)
    ->Name("Typical/MilvusStorage_Read_Vortex_1T")
    ->Args({1, 1, 1})  // Vortex + Medium + 1 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_OpenRead)
    ->Name("Typical/MilvusStorage_Read_Vortex_8T")
    ->Args({1, 1, 8})  // Vortex + Medium + 8 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_Take)
    ->Name("Typical/MilvusStorage_Take_Vortex_1T")
    ->Args({1, 1000, 1})  // Vortex + 1000 rows + 1 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_Take)
    ->Name("Typical/MilvusStorage_Take_Vortex_8T")
    ->Args({1, 1000, 8})  // Vortex + 1000 rows + 8 threads
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

}  // namespace benchmark
}  // namespace milvus_storage
