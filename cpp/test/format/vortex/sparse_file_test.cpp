// Copyright 2025 Zilliz
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

// POC test: Verify that a vortex file with only footer + specific field segments
// filled in (sparse file) can be read correctly via the scan API.

#include <gtest/gtest.h>
#include <memory>
#include <cstring>
#include <vector>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/api.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/io/memory.h>

#include "test_env.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

#include <boost/filesystem/operations.hpp>

namespace milvus_storage {

using namespace vortex;

// A minimal in-memory filesystem that serves a single file from a buffer.
class SingleBufferFileSystem : public arrow::fs::FileSystem {
 public:
  SingleBufferFileSystem(std::string path, std::shared_ptr<arrow::Buffer> buffer)
      : arrow::fs::FileSystem(arrow::io::default_io_context()),
        path_(std::move(path)),
        buffer_(std::move(buffer)) {}

  std::string type_name() const override { return "mem"; }

  bool Equals(const FileSystem& other) const override { return false; }

  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override {
    if (path == path_) {
      arrow::fs::FileInfo info;
      info.set_path(path);
      info.set_type(arrow::fs::FileType::File);
      info.set_size(buffer_->size());
      return info;
    }
    return arrow::Status::IOError("File not found: ", path);
  }

  arrow::Result<std::vector<arrow::fs::FileInfo>> GetFileInfo(
      const arrow::fs::FileSelector& select) override {
    return arrow::Status::NotImplemented("GetFileInfo with selector");
  }

  arrow::Status CreateDir(const std::string&, bool) override {
    return arrow::Status::NotImplemented("CreateDir");
  }
  arrow::Status DeleteDir(const std::string&) override {
    return arrow::Status::NotImplemented("DeleteDir");
  }
  arrow::Status DeleteDirContents(const std::string&, bool) override {
    return arrow::Status::NotImplemented("DeleteDirContents");
  }
  arrow::Status DeleteRootDirContents() override {
    return arrow::Status::NotImplemented("DeleteRootDirContents");
  }
  arrow::Status DeleteFile(const std::string&) override {
    return arrow::Status::NotImplemented("DeleteFile");
  }
  arrow::Status Move(const std::string&, const std::string&) override {
    return arrow::Status::NotImplemented("Move");
  }
  arrow::Status CopyFile(const std::string&, const std::string&) override {
    return arrow::Status::NotImplemented("CopyFile");
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(
      const std::string& path) override {
    if (path == path_) {
      return std::make_shared<arrow::io::BufferReader>(buffer_);
    }
    return arrow::Status::IOError("File not found: ", path);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(
      const std::string& path) override {
    if (path == path_) {
      return std::make_shared<arrow::io::BufferReader>(buffer_);
    }
    return arrow::Status::IOError("File not found: ", path);
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string&, const std::shared_ptr<const arrow::KeyValueMetadata>&) override {
    return arrow::Status::NotImplemented("OpenOutputStream");
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenAppendStream(
      const std::string&, const std::shared_ptr<const arrow::KeyValueMetadata>&) override {
    return arrow::Status::NotImplemented("OpenAppendStream");
  }

 private:
  std::string path_;
  std::shared_ptr<arrow::Buffer> buffer_;
};

class VortexSparseFileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    schema_ = arrow::schema({
        arrow::field("int32", arrow::int32(), false,
                     arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
        arrow::field("int64", arrow::int64(), false,
                     arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
        arrow::field("float", arrow::float32(), false,
                     arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"300"})),
    });

    ASSERT_STATUS_OK(InitTestProperties(properties_));
    local_fs_ = std::make_shared<arrow::fs::LocalFileSystem>();
  }

  void TearDown() override {
    boost::filesystem::remove_all(test_file_);
  }

  std::shared_ptr<arrow::RecordBatch> MakeBatch(int32_t offset, int32_t count) {
    arrow::Int32Builder i32_builder;
    arrow::Int64Builder i64_builder;
    arrow::FloatBuilder f32_builder;

    for (int32_t i = offset; i < offset + count; i++) {
      EXPECT_TRUE(i32_builder.Append(i).ok());
      EXPECT_TRUE(i64_builder.Append(static_cast<int64_t>(i) * 10).ok());
      EXPECT_TRUE(f32_builder.Append(static_cast<float>(i) * 0.5f).ok());
    }

    std::shared_ptr<arrow::Array> i32_arr, i64_arr, f32_arr;
    EXPECT_TRUE(i32_builder.Finish(&i32_arr).ok());
    EXPECT_TRUE(i64_builder.Finish(&i64_arr).ok());
    EXPECT_TRUE(f32_builder.Finish(&f32_arr).ok());

    return arrow::RecordBatch::Make(schema_, count, {i32_arr, i64_arr, f32_arr});
  }

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::fs::LocalFileSystem> local_fs_;
  api::Properties properties_;
  const char* test_file_ = "sparse_test.vortex";
};

TEST_F(VortexSparseFileTest, SparseFileFieldScan) {
  // Step 1: Write a vortex file with multiple batches
  {
    VortexFileWriter writer(local_fs_, schema_, test_file_, properties_);
    // Write enough data to produce multiple segments
    for (int i = 0; i < 10; i++) {
      auto batch = MakeBatch(i * 1000, 1000);
      ASSERT_STATUS_OK(writer.Write(batch));
    }
    ASSERT_STATUS_OK(writer.Flush());
    ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
    ASSERT_EQ(10000, cgfile.end_index);
  }

  // Step 2: Read the full file into memory
  ASSERT_AND_ASSIGN(auto file_info, local_fs_->GetFileInfo(test_file_));
  int64_t file_size = file_info.size();
  ASSERT_GT(file_size, 0);

  ASSERT_AND_ASSIGN(auto input, local_fs_->OpenInputFile(test_file_));
  ASSERT_AND_ASSIGN(auto full_buffer, input->Read(file_size));
  std::cout << "Full file size: " << file_size << " bytes" << std::endl;

  // Step 3: Open with VortexFile bridge to get field byte ranges
  // Wrap local filesystem in FileSystemWrapper for FFI
  auto fs_holder = std::make_shared<FileSystemWrapper>(local_fs_);
  auto vx_file = VortexFile::Open((uint8_t*)fs_holder.get(), test_file_);
  std::cout << "Row count: " << vx_file.RowCount() << std::endl;

  // Get byte ranges for "int32" field (layout tree uses arrow field name, not field_id)
  auto ranges = vx_file.FieldByteRanges("int32", static_cast<uint64_t>(file_size));
  ASSERT_GE(ranges.size(), 2u);  // At least footer range
  ASSERT_EQ(ranges.size() % 2, 0u);  // Must be pairs

  std::cout << "Byte ranges for field 'int32':" << std::endl;
  std::cout << "  Footer: [" << ranges[0] << ", " << ranges[1] << ")" << std::endl;
  uint64_t total_field_bytes = ranges[1] - ranges[0];
  for (size_t i = 2; i < ranges.size(); i += 2) {
    std::cout << "  Segment: [" << ranges[i] << ", " << ranges[i + 1] << ")" << std::endl;
    total_field_bytes += ranges[i + 1] - ranges[i];
  }
  std::cout << "Total bytes needed: " << total_field_bytes
            << " / " << file_size << " ("
            << (100.0 * total_field_bytes / file_size) << "%)" << std::endl;

  // Step 4: Create sparse buffer — zero-filled, then copy only needed ranges
  auto sparse_data = std::make_unique<uint8_t[]>(file_size);
  std::memset(sparse_data.get(), 0, file_size);

  const uint8_t* src = full_buffer->data();
  for (size_t i = 0; i < ranges.size(); i += 2) {
    uint64_t start = ranges[i];
    uint64_t end = ranges[i + 1];
    ASSERT_LE(end, static_cast<uint64_t>(file_size));
    std::memcpy(sparse_data.get() + start, src + start, end - start);
  }

  auto sparse_buffer = arrow::Buffer::Wrap(sparse_data.get(), file_size);

  // Step 5: Create filesystem backed by the sparse buffer
  auto sparse_fs = std::make_shared<SingleBufferFileSystem>(
      std::string(test_file_), sparse_buffer);

  // Step 6: Read from the sparse file — scan only the int32 field
  // VortexFormatReader takes arrow::fs::FileSystem directly
  VortexFormatReader reader(sparse_fs, schema_, test_file_, properties_,
                            std::vector<std::string>{"int32"});
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto chunked_array, reader.blocking_read(0, 10000));

  // Step 7: Verify results
  // Convert to table for easier access
  ASSERT_GT(chunked_array->num_chunks(), 0);

  // Flatten all chunks
  int64_t total_rows = 0;
  for (int i = 0; i < chunked_array->num_chunks(); i++) {
    auto chunk = chunked_array->chunk(i);
    // Each chunk is a StructArray containing the projected columns
    auto struct_arr = std::dynamic_pointer_cast<arrow::StructArray>(chunk);
    ASSERT_NE(struct_arr, nullptr);

    auto int32_col = struct_arr->GetFieldByName("int32");
    ASSERT_NE(int32_col, nullptr);
    auto int32_arr = std::dynamic_pointer_cast<arrow::Int32Array>(int32_col);
    ASSERT_NE(int32_arr, nullptr);

    for (int64_t j = 0; j < int32_arr->length(); j++) {
      int32_t expected = static_cast<int32_t>(total_rows + j);
      ASSERT_EQ(int32_arr->Value(j), expected)
          << "Mismatch at row " << (total_rows + j);
    }
    total_rows += chunk->length();
  }

  ASSERT_EQ(total_rows, 10000);
  std::cout << "SUCCESS: Sparse file scan verified " << total_rows
            << " rows correctly!" << std::endl;
}

}  // namespace milvus_storage
