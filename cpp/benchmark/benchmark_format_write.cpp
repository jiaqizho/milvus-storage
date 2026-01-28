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

#include "benchmark_format_common.h"

#include <arrow/filesystem/filesystem.h>

namespace milvus_storage {
namespace benchmark {

using namespace milvus_storage::api;

//=============================================================================
// Write Performance Benchmark
//=============================================================================

class FormatWriteBenchmark : public FormatBenchFixture {
  public:
  void SetUp(::benchmark::State& st) override {
    FormatBenchFixture::SetUp(st);

    // Create schema
    BENCH_ASSERT_AND_ASSIGN(schema_, CreateSchema(), st);
  }

  protected:
  std::shared_ptr<arrow::Schema> schema_;
};

// Write comparison benchmark across formats
// Args: [format_idx, data_config_idx, memory_config_idx]
BENCHMARK_DEFINE_F(FormatWriteBenchmark, WriteComparison)(::benchmark::State& st) {
  size_t format_idx = static_cast<size_t>(st.range(0));
  size_t data_config_idx = static_cast<size_t>(st.range(1));
  size_t memory_config_idx = static_cast<size_t>(st.range(2));

  std::string format = GetFormatByIndex(format_idx);
  if (!CheckFormatAvailable(st, format)) {
    return;
  }

  DataSizeConfig data_config = DataSizeConfig::FromIndex(data_config_idx);
  MemoryConfig memory_config = MemoryConfig::FromIndex(memory_config_idx);

  // Configure memory settings
  ConfigureMemory(memory_config);

  // Create test data
  BENCH_ASSERT_AND_ASSIGN(auto batch, CreateBatch(schema_, data_config, true), st);
  int64_t raw_data_size = CalculateRawDataSize(batch);
  int64_t total_rows = static_cast<int64_t>(data_config.num_rows);

  // Track total bytes and rows for throughput calculation
  int64_t total_bytes_written = 0;
  int64_t total_rows_written = 0;

  for (auto _ : st) {
    std::string path = GetUniquePath(format);

    // Create policy and writer
    BENCH_ASSERT_AND_ASSIGN(auto policy, CreatePolicyForFormat(format, schema_), st);
    auto writer = Writer::create(path, schema_, std::move(policy), properties_);
    BENCH_ASSERT_NOT_NULL(writer, st);

    // Write data
    BENCH_ASSERT_STATUS_OK(writer->write(batch), st);

    // Close and get column groups
    BENCH_ASSERT_AND_ASSIGN(auto cgs, writer->close(), st);

    total_bytes_written += raw_data_size;
    total_rows_written += total_rows;
  }

  // Report metrics
  ReportThroughput(st, total_bytes_written, total_rows_written);

  // Add labels for better output readability
  st.SetLabel(format + "/" + data_config.name + "/" + memory_config.name);
}

// Register write benchmark with all combinations
BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->ArgsProduct({
        {0, 1},     // Format: parquet(0), vortex(1)
        {0, 1, 2},  // DataConfig: Small(0), Medium(1), Large(2)
        {1}         // MemoryConfig: Default(1) only for basic tests
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

// Extended write benchmark with all memory configurations (for detailed analysis)
BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->Name("FormatWriteBenchmark/WriteComparisonExtended")
    ->ArgsProduct({
        {0, 1},    // Format: parquet(0), vortex(1)
        {0},       // DataConfig: Small(0) only
        {0, 1, 2}  // MemoryConfig: Low(0), Default(1), High(2)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

// High-dimensional vector write benchmark
BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->Name("FormatWriteBenchmark/WriteHighDim")
    ->ArgsProduct({
        {0, 1},  // Format: parquet(0), vortex(1)
        {3},     // DataConfig: HighDim(3)
        {1}      // MemoryConfig: Default(1)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

// Long string write benchmark
BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->Name("FormatWriteBenchmark/WriteLongString")
    ->ArgsProduct({
        {0, 1},  // Format: parquet(0), vortex(1)
        {4},     // DataConfig: LongString(4)
        {1}      // MemoryConfig: Default(1)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// File Size / Compression Analysis Benchmark
//=============================================================================

// This benchmark measures file size and compression ratio
// Args: [format_idx, data_config_idx]
BENCHMARK_DEFINE_F(FormatWriteBenchmark, CompressionAnalysis)(::benchmark::State& st) {
  size_t format_idx = static_cast<size_t>(st.range(0));
  size_t data_config_idx = static_cast<size_t>(st.range(1));

  std::string format = GetFormatByIndex(format_idx);
  if (!CheckFormatAvailable(st, format)) {
    return;
  }

  DataSizeConfig data_config = DataSizeConfig::FromIndex(data_config_idx);

  // Create test data
  BENCH_ASSERT_AND_ASSIGN(auto batch, CreateBatch(schema_, data_config, true), st);
  int64_t raw_data_size = CalculateRawDataSize(batch);

  int64_t total_file_size = 0;
  int64_t iteration_count = 0;

  for (auto _ : st) {
    std::string path = GetUniquePath(format);

    // Create policy and writer
    BENCH_ASSERT_AND_ASSIGN(auto policy, CreatePolicyForFormat(format, schema_), st);
    auto writer = Writer::create(path, schema_, std::move(policy), properties_);
    BENCH_ASSERT_NOT_NULL(writer, st);

    // Write data
    BENCH_ASSERT_STATUS_OK(writer->write(batch), st);

    // Close and get column groups
    BENCH_ASSERT_AND_ASSIGN(auto cgs, writer->close(), st);

    // Calculate total file size from column groups
    int64_t file_size = 0;
    for (const auto& cg : *cgs) {
      for (const auto& file : cg->files) {
        BENCH_ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(file.path), st);
        file_size += file_info.size();
      }
    }

    total_file_size += file_size;
    iteration_count++;
  }

  // Report compression metrics
  if (iteration_count > 0) {
    int64_t avg_file_size = total_file_size / iteration_count;
    ReportCompressionRatio(st, raw_data_size, avg_file_size);
    st.counters["file_size_kb"] =
        ::benchmark::Counter(static_cast<double>(avg_file_size) / 1024.0, ::benchmark::Counter::kDefaults);
    st.counters["raw_size_kb"] =
        ::benchmark::Counter(static_cast<double>(raw_data_size) / 1024.0, ::benchmark::Counter::kDefaults);
  }

  st.SetLabel(format + "/" + data_config.name);
}

BENCHMARK_REGISTER_F(FormatWriteBenchmark, CompressionAnalysis)
    ->ArgsProduct({
        {0, 1},          // Format: parquet(0), vortex(1)
        {0, 1, 2, 3, 4}  // All DataConfigs
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);  // Few iterations since we're measuring file size

//=============================================================================
// Typical Benchmarks
// Run with: --benchmark_filter="Typical/"
//=============================================================================

BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->Name("Typical/FormatWrite_Parquet")
    ->Args({0, 1, 1})  // Parquet + Medium + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->Name("Typical/FormatWrite_Vortex")
    ->Args({1, 1, 1})  // Vortex + Medium + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatWriteBenchmark, CompressionAnalysis)
    ->Name("Typical/Compression_Parquet")
    ->Args({0, 1})  // Parquet + Medium
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);

BENCHMARK_REGISTER_F(FormatWriteBenchmark, CompressionAnalysis)
    ->Name("Typical/Compression_Vortex")
    ->Args({1, 1})  // Vortex + Medium
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);

}  // namespace benchmark
}  // namespace milvus_storage
