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

#pragma once

#include <benchmark/benchmark.h>

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <numeric>
#include <random>
#include <algorithm>
#include <set>
#include <thread>
#include <atomic>
#include <chrono>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <fstream>
#endif

#include <arrow/filesystem/filesystem.h>
#include <arrow/api.h>
#include <arrow/memory_pool.h>

#include "milvus-storage/properties.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/reader.h"
#include "test_env.h"

namespace milvus_storage {
namespace benchmark {

//=============================================================================
// Benchmark Assertion Macros
//=============================================================================

#define BENCH_ASSERT_STATUS_OK(status, st)             \
  do {                                                 \
    if (!(status).ok()) {                              \
      (st).SkipWithError((status).ToString().c_str()); \
      return;                                          \
    }                                                  \
  } while (false)

#define BENCH_ASSERT_AND_ASSIGN_IMPL(status_name, lhs, rexpr, st) \
  auto status_name = (rexpr);                                     \
  BENCH_ASSERT_STATUS_OK(status_name.status(), st);               \
  lhs = std::move(status_name).ValueOrDie();

#define BENCH_ASSERT_AND_ASSIGN(lhs, rexpr, st) \
  BENCH_ASSERT_AND_ASSIGN_IMPL(CONCAT(_tmp_value, __COUNTER__), lhs, rexpr, st)

#define BENCH_ASSERT_NOT_NULL(ptr, st)               \
  do {                                               \
    if ((ptr) == nullptr) {                          \
      (st).SkipWithError("Unexpected null pointer"); \
      return;                                        \
    }                                                \
  } while (false)

//=============================================================================
// Data Configuration Structures
//=============================================================================

// Data size configurations as per design doc
struct DataSizeConfig {
  std::string name;
  size_t num_rows;
  size_t vector_dim;
  size_t string_length;

  static DataSizeConfig Small() { return {"Small", 4096, 128, 128}; }
  static DataSizeConfig Medium() { return {"Medium", 40960, 128, 128}; }
  static DataSizeConfig Large() { return {"Large", 409600, 128, 128}; }
  static DataSizeConfig HighDim() { return {"HighDim", 4096, 768, 128}; }
  static DataSizeConfig LongString() { return {"LongString", 4096, 128, 1024}; }

  static std::vector<DataSizeConfig> All() { return {Small(), Medium(), Large(), HighDim(), LongString()}; }

  static DataSizeConfig FromIndex(size_t idx) {
    auto all = All();
    return idx < all.size() ? all[idx] : Small();
  }
};

// Memory configurations as per design doc
struct MemoryConfig {
  std::string name;
  size_t buffer_size;
  size_t batch_size;

  static MemoryConfig Low() { return {"Low", 16ULL * 1024 * 1024, 1024}; }
  static MemoryConfig Default() { return {"Default", 128ULL * 1024 * 1024, 16384}; }
  static MemoryConfig High() { return {"High", 256ULL * 1024 * 1024, 32768}; }

  static std::vector<MemoryConfig> All() { return {Low(), Default(), High()}; }

  static MemoryConfig FromIndex(size_t idx) {
    auto all = All();
    return idx < all.size() ? all[idx] : Default();
  }
};

//=============================================================================
// Index Distribution Generators for Take Benchmarks
//=============================================================================

// Generate sequential indices starting from a given start position
inline std::vector<int64_t> GenerateSequentialIndices(size_t count, int64_t start = 0) {
  std::vector<int64_t> indices(count);
  std::iota(indices.begin(), indices.end(), start);
  return indices;
}

// Generate uniformly distributed random indices
inline std::vector<int64_t> GenerateRandomIndices(size_t count, int64_t max_value, uint32_t seed = 42) {
  std::vector<int64_t> indices;
  std::set<int64_t> seen;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int64_t> dist(0, max_value - 1);

  while (indices.size() < count) {
    int64_t idx = dist(gen);
    if (seen.insert(idx).second) {
      indices.push_back(idx);
    }
  }
  std::sort(indices.begin(), indices.end());
  return indices;
}

// Generate clustered indices (multiple small contiguous clusters)
inline std::vector<int64_t> GenerateClusteredIndices(size_t count,
                                                     int64_t max_value,
                                                     size_t cluster_size = 5,
                                                     uint32_t seed = 42) {
  std::vector<int64_t> indices;
  std::set<int64_t> seen;
  size_t num_clusters = (count + cluster_size - 1) / cluster_size;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int64_t> dist(0, max_value - static_cast<int64_t>(cluster_size));

  while (indices.size() < count) {
    int64_t start = dist(gen);
    for (size_t j = 0; j < cluster_size && indices.size() < count; ++j) {
      int64_t idx = start + static_cast<int64_t>(j);
      if (idx < max_value && seen.insert(idx).second) {
        indices.push_back(idx);
      }
    }
  }
  std::sort(indices.begin(), indices.end());
  return indices;
}

// Distribution type enum
enum class IndexDistribution { Sequential = 0, Random = 1, Clustered = 2 };

inline std::vector<int64_t> GenerateIndices(IndexDistribution dist,
                                            size_t count,
                                            int64_t max_value,
                                            uint32_t seed = 42) {
  switch (dist) {
    case IndexDistribution::Sequential:
      return GenerateSequentialIndices(count, 0);
    case IndexDistribution::Random:
      return GenerateRandomIndices(count, max_value, seed);
    case IndexDistribution::Clustered:
      return GenerateClusteredIndices(count, max_value, 5, seed);
    default:
      return GenerateRandomIndices(count, max_value, seed);
  }
}

inline const char* IndexDistributionName(IndexDistribution dist) {
  switch (dist) {
    case IndexDistribution::Sequential:
      return "Sequential";
    case IndexDistribution::Random:
      return "Random";
    case IndexDistribution::Clustered:
      return "Clustered";
    default:
      return "Unknown";
  }
}

//=============================================================================
// Format Utilities
//=============================================================================

// Get list of available formats based on build configuration
inline std::vector<std::string> GetAvailableFormats() { return {LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX}; }

// Get format name by index
inline std::string GetFormatByIndex(size_t idx) {
  auto formats = GetAvailableFormats();
  assert(idx < formats.size() && "Format index out of range");
  return formats[idx];
}

// Check if a format is available
inline bool IsFormatAvailable(const std::string& format) {
  auto formats = GetAvailableFormats();
  return std::find(formats.begin(), formats.end(), format) != formats.end();
}

//=============================================================================
// Memory Tracking Utilities
//=============================================================================

class MemoryTracker {
  public:
  MemoryTracker() : pool_(arrow::default_memory_pool()), initial_allocated_(pool_->bytes_allocated()) {}

  void Reset() { initial_allocated_ = pool_->bytes_allocated(); }

  int64_t GetPeakAllocated() const { return pool_->max_memory() - initial_allocated_; }

  double GetPeakAllocatedMB() const { return static_cast<double>(GetPeakAllocated()) / (1024.0 * 1024.0); }

  void ReportToState(::benchmark::State& st, const std::string& prefix = "") const {
    std::string peak_name = prefix.empty() ? "peak_memory_mb" : prefix + "_peak_memory_mb";
    st.counters[peak_name] = ::benchmark::Counter(GetPeakAllocatedMB(), ::benchmark::Counter::kDefaults);
  }

  private:
  arrow::MemoryPool* pool_;
  int64_t initial_allocated_;
};

//=============================================================================
// Thread Count Tracking Utilities
//=============================================================================

// Get current thread count for this process
inline int GetCurrentThreadCount() {
#ifdef __APPLE__
  mach_port_t task = mach_task_self();
  thread_act_array_t threads;
  mach_msg_type_number_t thread_count;
  if (task_threads(task, &threads, &thread_count) == KERN_SUCCESS) {
    vm_deallocate(task, reinterpret_cast<vm_address_t>(threads), thread_count * sizeof(thread_act_t));
    return static_cast<int>(thread_count);
  }
  return -1;
#elif defined(__linux__)
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("Threads:", 0) == 0) {
      return std::stoi(line.substr(8));
    }
  }
  return -1;
#else
  return -1;
#endif
}

class ThreadTracker {
  public:
  ThreadTracker() : running_(false), peak_threads_(0) {}

  ~ThreadTracker() { Stop(); }

  void Start(std::chrono::milliseconds interval = std::chrono::milliseconds(1)) {
    if (running_.exchange(true)) {
      return;
    }
    peak_threads_ = GetCurrentThreadCount();
    sampler_thread_ = std::thread([this, interval]() {
      while (running_.load()) {
        int current = GetCurrentThreadCount();
        if (current > 0) {
          int expected = peak_threads_.load();
          while (current > expected && !peak_threads_.compare_exchange_weak(expected, current)) {
          }
        }
        std::this_thread::sleep_for(interval);
      }
    });
  }

  void Stop() {
    if (running_.exchange(false)) {
      if (sampler_thread_.joinable()) {
        sampler_thread_.join();
      }
    }
  }

  int GetPeakThreads() const { return peak_threads_.load(); }

  void ReportToState(::benchmark::State& st) const {
    st.counters["peak_threads"] =
        ::benchmark::Counter(static_cast<double>(GetPeakThreads()), ::benchmark::Counter::kDefaults);
  }

  private:
  std::atomic<bool> running_;
  std::atomic<int> peak_threads_;
  std::thread sampler_thread_;
};

//=============================================================================
// Benchmark Metrics Helpers
//=============================================================================

inline void ReportThroughput(::benchmark::State& st, int64_t bytes_processed, int64_t rows_processed) {
  st.counters["throughput_mb_s"] =
      ::benchmark::Counter(static_cast<double>(bytes_processed) / (1024.0 * 1024.0), ::benchmark::Counter::kIsRate);
  st.counters["rows_per_sec"] =
      ::benchmark::Counter(static_cast<double>(rows_processed), ::benchmark::Counter::kIsRate);
}

inline void ReportCompressionRatio(::benchmark::State& st, int64_t raw_size, int64_t compressed_size) {
  if (raw_size > 0) {
    st.counters["compression_ratio"] = ::benchmark::Counter(
        static_cast<double>(compressed_size) / static_cast<double>(raw_size), ::benchmark::Counter::kDefaults);
  }
}

//=============================================================================
// Format Benchmark Fixture Base Class
//=============================================================================

class FormatBenchFixture : public ::benchmark::Fixture {
  public:
  void SetUp(::benchmark::State& st) override {
    // Initialize properties from environment
    BENCH_ASSERT_STATUS_OK(InitTestProperties(properties_), st);

    // Get filesystem
    BENCH_ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_), st);

    // Setup base path
    base_path_ = GetTestBasePath("format_benchmark");
    BENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_), st);
    BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_), st);

    // Get available formats
    available_formats_ = GetAvailableFormats();

    // Initialize memory tracker
    memory_tracker_.Reset();
  }

  void TearDown(::benchmark::State& st) override {
    // Report memory metrics
    memory_tracker_.ReportToState(st);

    // Clean up test directory
    auto status = DeleteTestDir(fs_, base_path_);
    if (!status.ok()) {
      // Log but don't fail on cleanup errors
    }
  }

  protected:
  // Configure memory settings based on MemoryConfig
  void ConfigureMemory(const MemoryConfig& config) {
    api::SetValue(properties_, PROPERTY_WRITER_BUFFER_SIZE, std::to_string(config.buffer_size).c_str());
    api::SetValue(properties_, PROPERTY_READER_RECORD_BATCH_MAX_ROWS, std::to_string(config.batch_size).c_str());
    api::SetValue(properties_, PROPERTY_READER_RECORD_BATCH_MAX_SIZE, std::to_string(config.buffer_size).c_str());
  }

  // Create schema for test data
  arrow::Result<std::shared_ptr<arrow::Schema>> CreateSchema(std::array<bool, 4> needed_columns = {true, true, true,
                                                                                                   true}) {
    return CreateTestSchema(needed_columns);
  }

  // Create test data batch
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateBatch(std::shared_ptr<arrow::Schema> schema,
                                                                 const DataSizeConfig& config,
                                                                 bool random_data = true,
                                                                 int64_t start_offset = 0) {
    return CreateTestData(schema, start_offset, random_data, config.num_rows, config.vector_dim, config.string_length);
  }

  // Create policy for specific format
  arrow::Result<std::unique_ptr<api::ColumnGroupPolicy>> CreatePolicyForFormat(
      const std::string& format, const std::shared_ptr<arrow::Schema>& schema) {
    return CreateSinglePolicy(format, schema);
  }

  // Get unique path for this benchmark iteration
  std::string GetUniquePath(const std::string& suffix = "") const {
    static std::atomic<uint64_t> counter{0};
    std::string path = base_path_ + "/bench_" + std::to_string(counter++);
    if (!suffix.empty()) {
      path += "_" + suffix;
    }
    return path;
  }

  // Check if format is available and skip if not
  bool CheckFormatAvailable(::benchmark::State& st, const std::string& format) {
    if (!IsFormatAvailable(format)) {
      st.SkipWithError(("Format not available: " + format).c_str());
      return false;
    }
    return true;
  }

  // Calculate logical data size for a batch.
  // Computes size from type layout and num_rows, consistent regardless of slicing.
  static int64_t CalculateRawDataSize(const std::shared_ptr<arrow::RecordBatch>& batch) {
    int64_t size = 0;
    int64_t num_rows = batch->num_rows();
    for (int i = 0; i < batch->num_columns(); ++i) {
      auto type = batch->column(i)->type();
      if (type->id() == arrow::Type::LIST) {
        // list<T>: use offsets to get actual child element count
        auto list_array = std::static_pointer_cast<arrow::ListArray>(batch->column(i));
        int64_t total_values = list_array->value_offset(num_rows) - list_array->value_offset(0);
        size += total_values * list_array->value_type()->byte_width();
      } else if (arrow::is_fixed_width(*type)) {
        size += num_rows * type->byte_width();
      } else if (type->id() == arrow::Type::STRING || type->id() == arrow::Type::BINARY) {
        // Variable-width (string/binary): use offsets
        auto offsets = batch->column(i)->data()->buffers[1];
        auto offset_base = batch->column(i)->offset();
        auto off_ptr = reinterpret_cast<const int32_t*>(offsets->data());
        size += off_ptr[offset_base + num_rows] - off_ptr[offset_base];
      } else {
        throw std::runtime_error("CalculateRawDataSize: unsupported type " + type->ToString());
      }
    }
    return size;
  }

  protected:
  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
  std::vector<std::string> available_formats_;
  MemoryTracker memory_tracker_;
};

}  // namespace benchmark
}  // namespace milvus_storage
