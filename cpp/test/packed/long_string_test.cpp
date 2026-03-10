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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <random>

#include <parquet/properties.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/mockfs.h>
#include <arrow/io/api.h>
#include <arrow/api.h>
#include <arrow/array/builder_binary.h>
#include <arrow/type.h>
#include <arrow/record_batch.h>
#include <arrow/util/key_value_metadata.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/packed/writer.h"
#include "milvus-storage/packed/reader.h"
#include "milvus-storage/format/parquet/file_reader.h"

#include "test_env.h"

namespace milvus_storage {

class LongStringTest : public ::testing::Test {
 protected:
  static constexpr int kTotalRows = 7000;
  static constexpr int kRowsPerBatch = 100;
  static constexpr int kNumBatches = kTotalRows / kRowsPerBatch;

  void SetUp() override {
    api::Properties properties;
    ASSERT_STATUS_OK(milvus_storage::InitTestProperties(properties));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties));
    path_ = GetTestBasePath("long-string-test");

    storage_config_ = StorageConfig();
    writer_memory_ = 64 * 1024 * 1024;  // 64 MB write buffer
    reader_memory_ = 16 * 1024 * 1024;  // 16 MB read buffer

    schema_ = arrow::schema({
        arrow::field("str_256", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
        arrow::field("str_32k", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
        arrow::field("str_64k", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"300"})),
    });
    paths_ = {path_ + "/data/0.parquet"};

    // Only write data if file doesn't exist yet (cache for repeated runs)
    auto file_info = fs_->GetFileInfo(paths_[0]);
    if (!file_info.ok() || file_info->type() == arrow::fs::FileType::NotFound) {
      ASSERT_STATUS_OK(CreateTestDir(fs_, path_));
      PrepareData();
    } else {
      std::cout << "[PrepareData] File already exists, skipping write" << std::endl;
    }
  }

  void TearDown() override {
    // Don't delete files - keep them cached for repeated runs
  }

  std::string GenerateRandomString(size_t length, std::mt19937& gen) {
    static const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<> dist(0, sizeof(charset) - 2);

    std::string str(length, '\0');
    for (size_t i = 0; i < length; ++i) {
      str[i] = charset[dist(gen)];
    }
    return str;
  }

  // Build a record batch with random strings per row
  std::shared_ptr<arrow::RecordBatch> BuildBatch(std::mt19937& gen) {
    arrow::StringBuilder builder_256;
    arrow::StringBuilder builder_32k;
    arrow::StringBuilder builder_64k;

    for (int i = 0; i < kRowsPerBatch; ++i) {
      EXPECT_TRUE(builder_256.Append(GenerateRandomString(256, gen)).ok());
      EXPECT_TRUE(builder_32k.Append(GenerateRandomString(32 * 1024, gen)).ok());
      EXPECT_TRUE(builder_64k.Append(GenerateRandomString(64 * 1024, gen)).ok());
    }

    std::shared_ptr<arrow::Array> arr_256, arr_32k, arr_64k;
    EXPECT_TRUE(builder_256.Finish(&arr_256).ok());
    EXPECT_TRUE(builder_32k.Finish(&arr_32k).ok());
    EXPECT_TRUE(builder_64k.Finish(&arr_64k).ok());

    return arrow::RecordBatch::Make(schema_, kRowsPerBatch, {arr_256, arr_32k, arr_64k});
  }

  // Write data to parquet files in SetUp so tests only focus on reading
  void PrepareData() {
    std::mt19937 gen(42);
    auto column_groups = std::vector<std::vector<int>>{{0, 1, 2}};

    auto writer_result =
        PackedRecordBatchWriter::Make(fs_, paths_, schema_, storage_config_, column_groups, writer_memory_);
    ASSERT_TRUE(writer_result.ok()) << writer_result.status().ToString();
    auto writer = std::move(writer_result).ValueOrDie();

    std::cout << "[PrepareData] Writing " << kTotalRows << " rows (columns: 256B / 32KB / 64KB strings)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumBatches; ++i) {
      auto batch = BuildBatch(gen);
      ASSERT_TRUE(writer->Write(batch).ok());
      if ((i + 1) % 10 == 0) {
        std::cout << "[PrepareData] Progress: " << (i + 1) << "/" << kNumBatches << " batches" << std::endl;
      }
    }
    ASSERT_TRUE(writer->Close().ok());

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::high_resolution_clock::now() - start)
                       .count();
    std::cout << "[PrepareData] Write completed in " << elapsed << " ms" << std::endl;
  }

  size_t writer_memory_;
  size_t reader_memory_;
  ArrowFileSystemPtr fs_;
  std::string path_;
  std::vector<std::string> paths_;
  std::shared_ptr<arrow::Schema> schema_;
  StorageConfig storage_config_;
};

TEST_F(LongStringTest, ReadLongStrings) {
  std::cout << "[Read] Starting read of " << kTotalRows << " rows using FileRowGroupReader..." << std::endl;
  auto read_start = std::chrono::high_resolution_clock::now();

  ASSERT_AND_ASSIGN(auto reader, FileRowGroupReader::Make(fs_, paths_[0], schema_, reader_memory_));

  auto metadata = reader->file_metadata();
  int num_row_groups = metadata->num_row_groups();
  std::cout << "[Read] File has " << num_row_groups << " row groups" << std::endl;

  ASSERT_STATUS_OK(reader->SetRowGroupOffsetAndCount(0, num_row_groups));

  int64_t total_read_rows = 0;
  int rg_count = 0;
  while (true) {
    std::shared_ptr<arrow::Table> table;
    ASSERT_STATUS_OK(reader->ReadNextRowGroup(&table));
    if (!table) break;

    total_read_rows += table->num_rows();
    rg_count++;

    // Verify columns exist and have expected types
    ASSERT_EQ(table->num_columns(), 3);

    if (rg_count % 50 == 0) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - read_start)
                         .count();
      std::cout << "[Read] Row groups: " << rg_count << "/" << num_row_groups
                << ", rows: " << total_read_rows << ", elapsed: " << elapsed << " ms" << std::endl;
    }
  }
  ASSERT_STATUS_OK(reader->Close());

  auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - read_start)
                     .count();

  std::cout << "[Read] Completed in " << read_ms << " ms" << std::endl;
  std::cout << "[Read] Total rows: " << total_read_rows << ", row groups: " << rg_count << std::endl;

  double throughput_mb = (double)total_read_rows * (256 + 32 * 1024 + 64 * 1024) / 1024.0 / 1024.0;
  double throughput_mbs = (read_ms > 0) ? (throughput_mb / (read_ms / 1000.0)) : 0;
  std::cout << "[Read] Data volume: " << std::fixed << std::setprecision(1) << throughput_mb << " MB, "
            << "throughput: " << throughput_mbs << " MB/s" << std::endl;

  ASSERT_EQ(total_read_rows, kTotalRows);
}

TEST_F(LongStringTest, ConcurrentReadLongStrings) {
  const int num_threads = 8;

  std::cout << "[ConcurrentRead] " << num_threads << " threads, each reading " << kTotalRows << " rows..." << std::endl;

  // Barrier: all threads start reading at the same time
  std::atomic<int> ready_count{0};
  std::atomic<bool> go{false};

  std::vector<std::thread> threads;
  std::vector<int64_t> thread_rows(num_threads, 0);
  std::vector<int64_t> thread_ms(num_threads, 0);
  std::vector<std::string> thread_errors(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      // Each thread creates its own reader
      auto reader_result = FileRowGroupReader::Make(fs_, paths_[0], schema_, reader_memory_);
      if (!reader_result.ok()) {
        thread_errors[t] = reader_result.status().ToString();
        return;
      }
      auto reader = std::move(reader_result).ValueOrDie();

      auto metadata = reader->file_metadata();
      int num_rg = metadata->num_row_groups();
      auto status = reader->SetRowGroupOffsetAndCount(0, num_rg);
      if (!status.ok()) {
        thread_errors[t] = status.ToString();
        return;
      }

      // Signal ready and wait for go
      ready_count.fetch_add(1);
      while (!go.load()) {
        std::this_thread::yield();
      }

      auto start = std::chrono::high_resolution_clock::now();

      int64_t rows = 0;
      while (true) {
        std::shared_ptr<arrow::Table> table;
        auto s = reader->ReadNextRowGroup(&table);
        if (!s.ok()) {
          thread_errors[t] = s.ToString();
          break;
        }
        if (!table) break;
        rows += table->num_rows();
      }
      reader->Close();

      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - start)
                         .count();
      thread_rows[t] = rows;
      thread_ms[t] = elapsed;
    });
  }

  // Wait for all threads ready, then fire
  while (ready_count.load() < num_threads) {
    std::this_thread::yield();
  }

  std::cout << "[ConcurrentRead] All threads ready, starting..." << std::endl;
  auto wall_start = std::chrono::high_resolution_clock::now();
  go.store(true);

  for (auto& th : threads) {
    th.join();
  }

  auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - wall_start)
                     .count();

  // Print results
  int64_t total_rows = 0;
  for (int t = 0; t < num_threads; ++t) {
    if (!thread_errors[t].empty()) {
      std::cout << "[Thread " << t << "] ERROR: " << thread_errors[t] << std::endl;
    } else {
      std::cout << "[Thread " << t << "] rows: " << thread_rows[t] << ", time: " << thread_ms[t] << " ms"
                << std::endl;
      total_rows += thread_rows[t];
    }
  }

  double total_mb = (double)total_rows * (256 + 32 * 1024 + 64 * 1024) / 1024.0 / 1024.0;
  double throughput = (wall_ms > 0) ? (total_mb / (wall_ms / 1000.0)) : 0;
  std::cout << "[ConcurrentRead] Wall time: " << wall_ms << " ms" << std::endl;
  std::cout << "[ConcurrentRead] Total data: " << std::fixed << std::setprecision(1) << total_mb << " MB, "
            << "aggregate throughput: " << throughput << " MB/s" << std::endl;

  for (int t = 0; t < num_threads; ++t) {
    EXPECT_TRUE(thread_errors[t].empty()) << "Thread " << t << ": " << thread_errors[t];
    EXPECT_EQ(thread_rows[t], kTotalRows);
  }
}

TEST_F(LongStringTest, InMemoryConcurrentReadLongStrings) {
  const int num_threads = 8;
  const std::string mock_path = "0.parquet";

  // Step 1: Create MockFileSystem and write data directly to it
  auto mock_fs = std::make_shared<arrow::fs::internal::MockFileSystem>(
      std::chrono::system_clock::now());
  std::vector<std::string> mock_paths = {mock_path};
  auto column_groups = std::vector<std::vector<int>>{{0, 1, 2}};

  std::cout << "[InMemory] Writing " << kTotalRows << " rows to MockFS..." << std::endl;
  auto write_start = std::chrono::high_resolution_clock::now();

  auto writer_result =
      PackedRecordBatchWriter::Make(mock_fs, mock_paths, schema_, storage_config_, column_groups, writer_memory_);
  ASSERT_TRUE(writer_result.ok()) << writer_result.status().ToString();
  auto writer = std::move(writer_result).ValueOrDie();

  std::mt19937 gen(42);
  for (int i = 0; i < kNumBatches; ++i) {
    auto batch = BuildBatch(gen);
    ASSERT_TRUE(writer->Write(batch).ok());
    if ((i + 1) % 10 == 0) {
      std::cout << "[InMemory] Write progress: " << (i + 1) << "/" << kNumBatches << " batches" << std::endl;
    }
  }
  ASSERT_TRUE(writer->Close().ok());

  auto write_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::high_resolution_clock::now() - write_start)
                      .count();

  // Verify file in MockFS
  ASSERT_AND_ASSIGN(auto mock_info, mock_fs->GetFileInfo(mock_path));
  std::cout << "[InMemory] Write completed in " << write_ms << " ms, file size: "
            << mock_info.size() / 1024 / 1024 << " MB" << std::endl;

  // Step 3: Concurrent read from MockFS (pure CPU, no IO)
  std::cout << "[InMemory] " << num_threads << " threads reading from memory..." << std::endl;

  std::atomic<int> ready_count{0};
  std::atomic<bool> go{false};

  std::vector<std::thread> threads;
  std::vector<int64_t> thread_rows(num_threads, 0);
  std::vector<int64_t> thread_ms(num_threads, 0);
  std::vector<std::string> thread_errors(num_threads);

  const int read_iterations = 10;  // Read file multiple times to give profiler enough samples

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      ready_count.fetch_add(1);
      while (!go.load()) {
        std::this_thread::yield();
      }

      auto start = std::chrono::high_resolution_clock::now();

      int64_t rows = 0;
      for (int iter = 0; iter < read_iterations; ++iter) {
        auto reader_result = FileRowGroupReader::Make(mock_fs, mock_path, schema_, reader_memory_);
        if (!reader_result.ok()) {
          thread_errors[t] = reader_result.status().ToString();
          return;
        }
        auto reader = std::move(reader_result).ValueOrDie();

        auto metadata = reader->file_metadata();
        int num_rg = metadata->num_row_groups();
        auto status = reader->SetRowGroupOffsetAndCount(0, num_rg);
        if (!status.ok()) {
          thread_errors[t] = status.ToString();
          return;
        }

        while (true) {
          std::shared_ptr<arrow::Table> table;
          auto s = reader->ReadNextRowGroup(&table);
          if (!s.ok()) {
            thread_errors[t] = s.ToString();
            return;
          }
          if (!table) break;
          rows += table->num_rows();
        }
        reader->Close();
      }

      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - start)
                         .count();
      thread_rows[t] = rows;
      thread_ms[t] = elapsed;
    });
  }

  while (ready_count.load() < num_threads) {
    std::this_thread::yield();
  }

  std::cout << "[InMemory] All threads ready, starting..." << std::endl;
  auto wall_start = std::chrono::high_resolution_clock::now();
  go.store(true);

  for (auto& th : threads) {
    th.join();
  }

  auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - wall_start)
                     .count();

  int64_t total_rows = 0;
  for (int t = 0; t < num_threads; ++t) {
    if (!thread_errors[t].empty()) {
      std::cout << "[Thread " << t << "] ERROR: " << thread_errors[t] << std::endl;
    } else {
      std::cout << "[Thread " << t << "] rows: " << thread_rows[t] << ", time: " << thread_ms[t] << " ms"
                << std::endl;
      total_rows += thread_rows[t];
    }
  }

  double total_mb = (double)total_rows * (256 + 32 * 1024 + 64 * 1024) / 1024.0 / 1024.0;
  double throughput = (wall_ms > 0) ? (total_mb / (wall_ms / 1000.0)) : 0;
  std::cout << "[InMemory] Wall time: " << wall_ms << " ms" << std::endl;
  std::cout << "[InMemory] Total data: " << std::fixed << std::setprecision(1) << total_mb << " MB, "
            << "aggregate throughput: " << throughput << " MB/s" << std::endl;

  for (int t = 0; t < num_threads; ++t) {
    EXPECT_TRUE(thread_errors[t].empty()) << "Thread " << t << ": " << thread_errors[t];
    EXPECT_EQ(thread_rows[t], kTotalRows * read_iterations);
  }
}

}  // namespace milvus_storage
