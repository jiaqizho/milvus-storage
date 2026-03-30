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

// Compression Advantage Benchmark
// Synthetic data designed to showcase Vortex encoding advantages over Parquet/Lance on S3.
// Schema: 10 sorted int64 + 5 low-cardinality string + 5 narrow-range float64 = 20 columns
// 10M rows, no vector columns, no CUSTOM_SEGMENT_PATH dependency.
//
// Vortex advantages on this data:
//   - sorted int64: delta encoding with bitpack → near-zero bits per value
//   - low-cardinality strings: global dictionary → 5 bits per value
//   - narrow-range floats: frame-of-reference → reduced bit width
// Parquet disadvantage: 1MB row group limit resets encoding state every ~5700 rows,
//   reducing compression efficiency. Also creates ~1750 row groups with per-RG metadata.

#include "benchmark_format_common.h"

#include <iostream>
#include <arrow/table.h>
#include <arrow/record_batch.h>
#include <arrow/c/bridge.h>
#include <sys/resource.h>

#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/thread_pool.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"

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

namespace milvus_storage {
namespace benchmark {

using namespace milvus_storage::api;
using namespace milvus_storage::api::transaction;

// ============================================================================
// Data Generation Constants
// ============================================================================

static constexpr int64_t kNumRows = 10'000'000;
static constexpr int64_t kBatchSize = 100'000;
static constexpr int kNumIntCols = 10;
static constexpr int kNumStrCols = 5;
static constexpr int kNumFloatCols = 5;
static constexpr int kStringCardinality = 20;

// ============================================================================
// Schema & Data Generation
// ============================================================================

static std::shared_ptr<arrow::Schema> MakeCompressionSchema() {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (int i = 0; i < kNumIntCols; ++i)
    fields.push_back(arrow::field("sorted_int_" + std::to_string(i), arrow::int64()));
  for (int i = 0; i < kNumStrCols; ++i) fields.push_back(arrow::field("category_" + std::to_string(i), arrow::utf8()));
  for (int i = 0; i < kNumFloatCols; ++i)
    fields.push_back(arrow::field("narrow_float_" + std::to_string(i), arrow::float64()));
  return arrow::schema(fields);
}

static arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakeBatch(const std::shared_ptr<arrow::Schema>& schema,
                                                                    int64_t num_rows,
                                                                    int64_t global_offset) {
  static const std::vector<std::string> kStringPool = [] {
    std::vector<std::string> pool;
    for (int i = 0; i < kStringCardinality; ++i) pool.push_back("category_" + std::to_string(i));
    return pool;
  }();

  std::mt19937 gen(42 + static_cast<uint32_t>(global_offset / kBatchSize));
  std::uniform_int_distribution<int> str_dist(0, kStringCardinality - 1);
  std::uniform_real_distribution<double> float_dist(50.0, 150.0);

  std::vector<std::shared_ptr<arrow::Array>> arrays;

  // Sorted int64 columns: monotonically increasing, each column offset by c*1M
  for (int c = 0; c < kNumIntCols; ++c) {
    arrow::Int64Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(num_rows));
    for (int64_t r = 0; r < num_rows; ++r) {
      builder.UnsafeAppend(global_offset + r + static_cast<int64_t>(c) * 1'000'000LL);
    }
    std::shared_ptr<arrow::Array> arr;
    ARROW_RETURN_NOT_OK(builder.Finish(&arr));
    arrays.push_back(std::move(arr));
  }

  // Low-cardinality string columns (20 unique values)
  for (int c = 0; c < kNumStrCols; ++c) {
    arrow::StringBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(num_rows));
    ARROW_RETURN_NOT_OK(builder.ReserveData(num_rows * 12));
    for (int64_t r = 0; r < num_rows; ++r) {
      builder.UnsafeAppend(kStringPool[str_dist(gen)]);
    }
    std::shared_ptr<arrow::Array> arr;
    ARROW_RETURN_NOT_OK(builder.Finish(&arr));
    arrays.push_back(std::move(arr));
  }

  // Narrow-range float64 columns (values in [50.0, 150.0])
  for (int c = 0; c < kNumFloatCols; ++c) {
    arrow::DoubleBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(num_rows));
    for (int64_t r = 0; r < num_rows; ++r) {
      builder.UnsafeAppend(float_dist(gen));
    }
    std::shared_ptr<arrow::Array> arr;
    ARROW_RETURN_NOT_OK(builder.Finish(&arr));
    arrays.push_back(std::move(arr));
  }

  return arrow::RecordBatch::Make(schema, num_rows, arrays);
}

// Cached data to avoid regenerating 10M rows for every benchmark case
struct CachedData {
  std::shared_ptr<arrow::Schema> schema;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  int64_t total_bytes = 0;
  int64_t total_rows = 0;
  bool ready = false;
};

static CachedData& GetCachedData() {
  static CachedData data;
  return data;
}

static arrow::Status EnsureDataGenerated() {
  auto& data = GetCachedData();
  if (data.ready)
    return arrow::Status::OK();

  std::cerr << "[CompressionBench] Generating " << kNumRows << " rows (" << kNumIntCols << " int64 + " << kNumStrCols
            << " string + " << kNumFloatCols << " float64)..." << std::endl;
  data.schema = MakeCompressionSchema();

  int64_t remaining = kNumRows;
  int64_t offset = 0;
  while (remaining > 0) {
    int64_t batch_rows = std::min(kBatchSize, remaining);
    ARROW_ASSIGN_OR_RAISE(auto batch, MakeBatch(data.schema, batch_rows, offset));
    // Compute raw data size inline (CalculateRawDataSize is protected)
    int64_t batch_bytes = 0;
    for (int i = 0; i < batch->num_columns(); ++i) {
      auto type = batch->column(i)->type();
      if (arrow::is_fixed_width(*type)) {
        batch_bytes += batch->num_rows() * type->byte_width();
      } else if (type->id() == arrow::Type::STRING) {
        auto offsets = batch->column(i)->data()->buffers[1];
        auto off_ptr = reinterpret_cast<const int32_t*>(offsets->data());
        auto offset_base = batch->column(i)->offset();
        batch_bytes += off_ptr[offset_base + batch->num_rows()] - off_ptr[offset_base];
      }
    }
    data.total_bytes += batch_bytes;
    data.total_rows += batch->num_rows();
    data.batches.push_back(std::move(batch));
    offset += batch_rows;
    remaining -= batch_rows;
  }

  std::cerr << "[CompressionBench] Generated " << data.total_rows << " rows, " << (data.total_bytes / (1024 * 1024))
            << " MB raw data" << std::endl;
  data.ready = true;
  return arrow::Status::OK();
}

// ============================================================================
// Benchmark Fixture
// ============================================================================

class CompressionBenchFixture : public FormatBenchFixtureBase<false> {
  public:
  void SetUp(::benchmark::State& st) override {
    BENCH_ASSERT_STATUS_OK(InitTestProperties(properties_), st);
    ConfigureGlobalProp();
    BENCH_ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_), st);

    // Use a path that won't collide with other benchmarks
    base_path_ = GetTestBasePath("compression_bench");
    BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_), st);
    available_formats_ = GetAvailableFormats();

    BENCH_ASSERT_STATUS_OK(EnsureDataGenerated(), st);
    schema_ = GetCachedData().schema;
  }

  void TearDown(::benchmark::State& st) override {
    schema_.reset();
    ThreadPoolHolder::Release();
    // Do NOT delete base_path_ — cached data must persist across benchmark cases
  }

  void ConfigureThreadPool(int num_threads) { ThreadPoolHolder::WithSingleton(num_threads); }

  protected:
  // -----------------------------------------------------------------------
  // Milvus Storage Write / Read
  // -----------------------------------------------------------------------

  arrow::Status WriteMilvusFormat(const std::string& format, const std::string& path) {
    auto& data = GetCachedData();
    ARROW_ASSIGN_OR_RAISE(auto policy, CreateSinglePolicy(format, data.schema));

    auto writer = Writer::create(path, data.schema, std::move(policy), properties_);
    if (!writer)
      return arrow::Status::Invalid("Failed to create writer");

    for (const auto& batch : data.batches) {
      ARROW_RETURN_NOT_OK(writer->write(batch));
    }
    ARROW_ASSIGN_OR_RAISE(auto cgs, writer->close());

    ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
    txn->AppendFiles(*cgs);
    ARROW_ASSIGN_OR_RAISE(auto version, txn->Commit());
    return arrow::Status::OK();
  }

  arrow::Status ReadMilvusFormat(const std::string& path) {
    auto& data = GetCachedData();
    ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
    ARROW_ASSIGN_OR_RAISE(auto manifest, txn->GetManifest());
    auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());

    auto reader = Reader::create(cgs, data.schema, nullptr, properties_);
    if (!reader)
      return arrow::Status::Invalid("Failed to create reader");

    ARROW_ASSIGN_OR_RAISE(auto batch_reader, reader->get_record_batch_reader());
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
      if (!batch)
        break;
    }
    return arrow::Status::OK();
  }

  arrow::Status ReadMilvusFormatWithStats(const std::string& path, int64_t& out_rows, int64_t& out_bytes) {
    auto& data = GetCachedData();
    ARROW_ASSIGN_OR_RAISE(auto txn, Transaction::Open(fs_, path));
    ARROW_ASSIGN_OR_RAISE(auto manifest, txn->GetManifest());
    auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());

    auto reader = Reader::create(cgs, data.schema, nullptr, properties_);
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

  // -----------------------------------------------------------------------
  // Lance Write / Read
  // -----------------------------------------------------------------------

  arrow::Result<std::string> BuildLanceUri(const std::string& relative_path) {
    ArrowFileSystemConfig fs_config;
    ARROW_RETURN_NOT_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
    return lance::BuildLanceBaseUri(fs_config, relative_path);
  }

  lance::LanceStorageOptions GetLanceStorageOptions() {
    ArrowFileSystemConfig fs_config;
    auto status = ArrowFileSystemConfig::create_file_system_config(properties_, fs_config);
    if (!status.ok())
      return {};
    return lance::ToLanceStorageOptions(fs_config);
  }

  arrow::Status WriteLanceDataset(const std::string& lance_uri, const lance::LanceStorageOptions& storage_options) {
    auto& data = GetCachedData();
    ARROW_ASSIGN_OR_RAISE(auto batch_reader, arrow::RecordBatchReader::Make(data.batches, data.schema));
    ArrowArrayStream stream;
    ARROW_RETURN_NOT_OK(arrow::ExportRecordBatchReader(batch_reader, &stream));
    try {
      auto dataset = lance::BlockingDataset::WriteDataset(lance_uri, &stream, storage_options);
    } catch (const lance::LanceException& e) {
      return arrow::Status::IOError("Lance write failed: ", e.what());
    }
    return arrow::Status::OK();
  }

  // -----------------------------------------------------------------------
  // Cache management: write once, read many
  // -----------------------------------------------------------------------

  std::string CachedPath(const std::string& format_name) const { return "compression_cache/" + format_name; }

  arrow::Status EnsureMilvusData(const std::string& format, const std::string& path) {
    auto txn_result = Transaction::Open(fs_, path);
    if (txn_result.ok()) {
      auto manifest_result = (*txn_result)->GetManifest();
      if (manifest_result.ok() && !(*manifest_result)->columnGroups().empty()) {
        return arrow::Status::OK();
      }
    }
    std::cerr << "[CompressionBench] Writing " << format << " data to " << path << "..." << std::endl;
    ARROW_RETURN_NOT_OK(DeleteTestDir(fs_, path));
    ARROW_RETURN_NOT_OK(CreateTestDir(fs_, path));
    return WriteMilvusFormat(format, path);
  }

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
    std::cerr << "[CompressionBench] Writing lance data to " << path << "..." << std::endl;
    ARROW_RETURN_NOT_OK(DeleteTestDir(fs_, path));
    ARROW_RETURN_NOT_OK(CreateTestDir(fs_, path));
    return WriteLanceDataset(lance_uri, storage_options);
  }

  std::shared_ptr<arrow::Schema> schema_;
};

// ============================================================================
// Full Scan Read Benchmark — Milvus Storage (Parquet / Vortex)
// ============================================================================

// Args: [format_type, num_threads]
BENCHMARK_DEFINE_F(CompressionBenchFixture, FullScan)(::benchmark::State& st) {
  int format_type = static_cast<int>(st.range(0));
  int num_threads = static_cast<int>(st.range(1));
  std::string format = format_type == 0 ? LOON_FORMAT_PARQUET : LOON_FORMAT_VORTEX;
  std::string format_name = format_type == 0 ? "parquet" : "vortex";

  ConfigureThreadPool(num_threads);

  std::string path = CachedPath(format_name);
  BENCH_ASSERT_STATUS_OK(EnsureMilvusData(format, path), st);

  // Collect stats once before benchmark loop
  int64_t rows_per_iter = 0, bytes_per_iter = 0;
  BENCH_ASSERT_STATUS_OK(ReadMilvusFormatWithStats(path, rows_per_iter, bytes_per_iter), st);

  ResetFsMetrics();
  if (format_type == 1) {
    vortex::ResetVortexDecodeMetrics();
  } else {
    parquet::ResetParquetDecodeMetrics();
  }

  auto cpu_before = GetProcessCpuTime();

  for (auto _ : st) {
    BENCH_ASSERT_STATUS_OK(ReadMilvusFormat(path), st);
  }

  auto cpu_elapsed = GetProcessCpuTime() - cpu_before;
  ReportCpuTime(st, cpu_elapsed);

  int64_t total_rows = rows_per_iter * static_cast<int64_t>(st.iterations());
  int64_t total_bytes = bytes_per_iter * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  ReportFsMetrics(st);

  double iters = static_cast<double>(st.iterations());
  if (format_type == 1) {
    auto metrics = vortex::GetVortexDecodeMetrics();
    double decode_ms = static_cast<double>(metrics.decode_ns) / 1e6;
    double io_wait_ms = static_cast<double>(metrics.io_wait_ns) / 1e6;
    st.counters["decode_ms/iter"] = ::benchmark::Counter(decode_ms / iters, ::benchmark::Counter::kDefaults);
    st.counters["io_wait_ms/iter"] = ::benchmark::Counter(io_wait_ms / iters, ::benchmark::Counter::kDefaults);
  } else {
    auto metrics = parquet::GetParquetDecodeMetrics();
    double io_decode_ms = static_cast<double>(metrics.read_decode_ns) / 1e6;
    double decode_ms = static_cast<double>(metrics.decode_only_ns) / 1e6;
    st.counters["io+decode_ms/iter"] = ::benchmark::Counter(io_decode_ms / iters, ::benchmark::Counter::kDefaults);
    st.counters["decode_ms/iter"] = ::benchmark::Counter(decode_ms / iters, ::benchmark::Counter::kDefaults);
  }

  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel(format_name + "/" + std::to_string(num_threads) + "T/10M_rows");
}

BENCHMARK_REGISTER_F(CompressionBenchFixture, FullScan)
    ->ArgsProduct({
        {0, 1},  // Format: parquet(0), vortex(1)
        {1, 16}  // Threads
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

// ============================================================================
// Full Scan Read Benchmark — Lance
// ============================================================================

BENCHMARK_DEFINE_F(CompressionBenchFixture, LanceFullScan)(::benchmark::State& st) {
  std::string path = CachedPath("lance");

  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  BENCH_ASSERT_STATUS_OK(EnsureLanceData(lance_uri, storage_options, path), st);

  auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);

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

  int64_t rows_per_iter = 0, bytes_per_iter = 0;
  BENCH_ASSERT_STATUS_OK(read_lance(true, rows_per_iter, bytes_per_iter), st);

  ResetFsMetrics();
  dataset->IOStatsIncremental();

  auto cpu_before = GetProcessCpuTime();

  for (auto _ : st) {
    int64_t dummy_rows = 0, dummy_bytes = 0;
    BENCH_ASSERT_STATUS_OK(read_lance(false, dummy_rows, dummy_bytes), st);
  }

  auto cpu_elapsed = GetProcessCpuTime() - cpu_before;
  ReportCpuTime(st, cpu_elapsed);

  auto lance_io = dataset->IOStatsIncremental();

  int64_t total_rows = rows_per_iter * static_cast<int64_t>(st.iterations());
  int64_t total_bytes = bytes_per_iter * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  ReportIOMetrics(st, lance_io.read_iops, lance_io.read_bytes);
  st.SetLabel("lance/10M_rows");
}

BENCHMARK_REGISTER_F(CompressionBenchFixture, LanceFullScan)->Unit(::benchmark::kMillisecond)->UseRealTime();

// ============================================================================
// Column Projection: 3 columns — one per type (int64, string, float64)
// ============================================================================

static std::shared_ptr<std::vector<std::string>> MakeProjColumns() {
  return std::make_shared<std::vector<std::string>>(
      std::vector<std::string>{"sorted_int_0", "category_0", "narrow_float_0"});
}

static std::shared_ptr<arrow::Schema> MakeProjSchema(const std::shared_ptr<arrow::Schema>& full_schema) {
  return arrow::schema({full_schema->GetFieldByName("sorted_int_0"), full_schema->GetFieldByName("category_0"),
                        full_schema->GetFieldByName("narrow_float_0")});
}

// ============================================================================
// Multi-Take Concurrency Benchmark — Milvus Storage (Parquet / Vortex)
// ============================================================================

// Args: [format_type, take_count, num_readers, num_threads]
BENCHMARK_DEFINE_F(CompressionBenchFixture, MultiTake)(::benchmark::State& st) {
  int format_type = static_cast<int>(st.range(0));
  size_t take_count = static_cast<size_t>(st.range(1));
  int num_readers = static_cast<int>(st.range(2));
  int num_threads = static_cast<int>(st.range(3));
  std::string format = format_type == 0 ? LOON_FORMAT_PARQUET : LOON_FORMAT_VORTEX;
  std::string format_name = format_type == 0 ? "parquet" : "vortex";

  ConfigureThreadPool(num_threads);

  std::string path = CachedPath(format_name);
  BENCH_ASSERT_STATUS_OK(EnsureMilvusData(format, path), st);

  auto& data = GetCachedData();
  auto needed_columns = MakeProjColumns();

  // Each reader gets its own random indices (different seed per reader)
  std::vector<std::vector<int64_t>> per_reader_indices(num_readers);
  for (int i = 0; i < num_readers; ++i) {
    per_reader_indices[i] = GenerateRandomIndices(take_count, data.total_rows, 42 + i);
  }

  // Open transaction once and reuse across iterations
  BENCH_ASSERT_AND_ASSIGN(auto txn, Transaction::Open(fs_, path), st);
  BENCH_ASSERT_AND_ASSIGN(auto manifest, txn->GetManifest(), st);
  auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());
  auto reader = Reader::create(cgs, schema_, needed_columns, properties_);
  if (!reader) {
    st.SkipWithError("Failed to create reader");
    return;
  }

  ResetFsMetrics();

  for (auto _ : st) {
    std::vector<std::thread> reader_threads;
    std::atomic<bool> has_error{false};

    for (int i = 0; i < num_readers; ++i) {
      reader_threads.emplace_back([&, i]() {
        auto result = reader->take(per_reader_indices[i], num_threads);
        if (!result.ok()) {
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
  st.SetLabel(format_name + "/" + std::to_string(take_count) + "rows/" + std::to_string(num_readers) + "readers/" +
              std::to_string(num_threads) + "pool/proj3col");
}

BENCHMARK_REGISTER_F(CompressionBenchFixture, MultiTake)
    ->ArgsProduct({
        {0, 1},                      // Format: parquet(0), vortex(1)
        {10},                        // TakeCount
        {1, 16, 64, 128, 256, 512},  // NumReaders
        {1},                         // NumThreads (decode thread pool)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

// ============================================================================
// Multi-Take Concurrency Benchmark — Lance
// ============================================================================

// Args: [take_count, num_readers]
BENCHMARK_DEFINE_F(CompressionBenchFixture, LanceMultiTake)(::benchmark::State& st) {
  size_t take_count = static_cast<size_t>(st.range(0));
  int num_readers = static_cast<int>(st.range(1));

  std::string path = CachedPath("lance");

  BENCH_ASSERT_AND_ASSIGN(auto lance_uri, BuildLanceUri(path), st);
  auto storage_options = GetLanceStorageOptions();

  BENCH_ASSERT_STATUS_OK(EnsureLanceData(lance_uri, storage_options, path), st);

  auto& data = GetCachedData();
  auto proj_schema = MakeProjSchema(schema_);

  std::vector<std::vector<int64_t>> per_reader_indices(num_readers);
  for (int i = 0; i < num_readers; ++i) {
    per_reader_indices[i] = GenerateRandomIndices(take_count, data.total_rows, 42 + i);
  }

  auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);
  dataset->IOStatsIncremental();  // reset

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
  st.SetLabel("lance/" + std::to_string(take_count) + "rows/" + std::to_string(num_readers) + "readers/proj3col");
}

BENCHMARK_REGISTER_F(CompressionBenchFixture, LanceMultiTake)
    ->ArgsProduct({
        {10},                        // TakeCount
        {1, 16, 64, 128, 256, 512},  // NumReaders
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

}  // namespace benchmark
}  // namespace milvus_storage
