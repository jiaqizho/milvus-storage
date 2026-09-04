// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus-storage/filesystem/talon/talon_file_system.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>
#include <arrow/status.h>
#include <arrow/util/future.h>

#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#include "talon_bridge.h"

namespace milvus_storage::talon {
namespace {

class TalonInputFile final : public arrow::io::RandomAccessFile, public NonBlockingRandomAccessFile {
  public:
  TalonInputFile(TalonObjectReader reader, arrow::MemoryPool* const pool) : reader_(std::move(reader)), pool_(pool) {}

  arrow::Result<int64_t> GetSize() override { return GetSizeAsync().result(); }

  arrow::Future<int64_t> GetSizeAsync() override {
    if (closed_) {
      return arrow::Future<int64_t>::MakeFinished(
          arrow::Result<int64_t>(arrow::Status::Invalid("Operation on closed Talon input file")));
    }
    const int64_t known_size = reader_->KnownSize();
    if (known_size >= 0) {
      return arrow::Future<int64_t>::MakeFinished(known_size);
    }

    return reader_->StatAsync();
  }

  arrow::Future<int64_t> ReadAtAsyncInto(int64_t position, int64_t nbytes, uint8_t* out) override {
    if (position < 0) {
      return arrow::Future<int64_t>::MakeFinished(
          arrow::Result<int64_t>(arrow::Status::Invalid("Cannot read from negative position")));
    }
    if (nbytes < 0) {
      return arrow::Future<int64_t>::MakeFinished(
          arrow::Result<int64_t>(arrow::Status::Invalid("Cannot read a negative number of bytes")));
    }
    if (nbytes > 0 && out == nullptr) {
      return arrow::Future<int64_t>::MakeFinished(
          arrow::Result<int64_t>(arrow::Status::Invalid("Talon read destination is null")));
    }

    if (closed_) {
      return arrow::Future<int64_t>::MakeFinished(
          arrow::Result<int64_t>(arrow::Status::Invalid("Operation on closed Talon input file")));
    }
    if (nbytes == 0) {
      return arrow::Future<int64_t>::MakeFinished(0);
    }

    return reader_->ReadAtAsync(static_cast<uint64_t>(position), static_cast<uint64_t>(nbytes), out);
  }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    return ReadAtAsyncInto(position, nbytes, reinterpret_cast<uint8_t*>(out)).result();
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    if (nbytes < 0) {
      return arrow::Status::Invalid("Cannot read a negative number of bytes");
    }
    ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateResizableBuffer(nbytes, pool_));
    ARROW_ASSIGN_OR_RAISE(const int64_t bytes_read, ReadAt(position, nbytes, buffer->mutable_data()));
    ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read));
    return std::shared_ptr<arrow::Buffer>(std::move(buffer));
  }

  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& io_context,
                                                          int64_t position,
                                                          int64_t nbytes) override {
    if (nbytes < 0) {
      return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(arrow::Result<std::shared_ptr<arrow::Buffer>>(
          arrow::Status::Invalid("Cannot read a negative number of bytes")));
    }
    auto maybe_buffer = arrow::AllocateResizableBuffer(nbytes, io_context.pool());
    if (!maybe_buffer.ok()) {
      return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(
          arrow::Result<std::shared_ptr<arrow::Buffer>>(maybe_buffer.status()));
    }
    auto buffer = std::move(maybe_buffer).ValueOrDie();
    auto* const out = buffer->mutable_data();
    return ReadAtAsyncInto(position, nbytes, out)
        .Then([buffer = std::move(buffer),
               nbytes](const int64_t bytes_read) mutable -> arrow::Result<std::shared_ptr<arrow::Buffer>> {
          if (bytes_read > nbytes) {
            return arrow::Status::IOError("Talon returned more bytes than requested");
          }
          ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read));
          return std::shared_ptr<arrow::Buffer>(std::move(buffer));
        });
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    ARROW_ASSIGN_OR_RAISE(const int64_t bytes_read, ReadAt(pos_, nbytes, out));
    pos_ += bytes_read;
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
    pos_ += buffer->size();
    return buffer;
  }

  arrow::Status Seek(int64_t position) override {
    if (position < 0) {
      return arrow::Status::Invalid("Cannot seek to negative position");
    }
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed Talon input file");
    }
    const int64_t known_size = reader_->KnownSize();
    if (known_size >= 0 && position > known_size) {
      return arrow::Status::IOError("Cannot seek past end of Talon input file");
    }
    pos_ = position;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed Talon input file");
    }
    return pos_;
  }

  // Matches the lifecycle contract used by Arrow's S3 and Azure input files:
  // Close() is not safe to call concurrently with another operation on this
  // file. Reads whose submission completed before Close() was called own a
  // Rust reader clone and may complete after the C++ handle is released here.
  arrow::Status Close() override {
    reader_.reset();
    closed_ = true;
    return arrow::Status::OK();
  }

  bool closed() const override { return closed_; }

  private:
  std::optional<TalonObjectReader> reader_;
  arrow::MemoryPool* const pool_;
  int64_t pos_ = 0;
  bool closed_ = false;
};

class TalonFileSystem final : public arrow::fs::FileSystem,
                              public UploadConditional,
                              public UploadSizable,
                              public Observable {
  public:
  TalonFileSystem(ArrowFileSystemPtr origin_fs,
                  std::shared_ptr<TalonClient> client,
                  std::string bucket,
                  std::string cloud_provider,
                  std::string coordinator,
                  uint32_t block_size)
      : arrow::fs::FileSystem(origin_fs->io_context()),
        origin_fs_(std::move(origin_fs)),
        client_(std::move(client)),
        bucket_(std::move(bucket)),
        cloud_provider_(std::move(cloud_provider)),
        coordinator_(std::move(coordinator)),
        block_size_(block_size) {}

  std::string type_name() const override { return origin_fs_->type_name(); }

  arrow::Result<std::string> NormalizePath(std::string path) override {
    return origin_fs_->NormalizePath(std::move(path));
  }

  arrow::Result<std::string> PathFromUri(const std::string& uri) const override { return origin_fs_->PathFromUri(uri); }

  arrow::Result<std::string> MakeUri(std::string path) const override { return origin_fs_->MakeUri(std::move(path)); }

  bool Equals(const arrow::fs::FileSystem& other) const override {
    if (this == &other) {
      return true;
    }
    const auto* talon = dynamic_cast<const TalonFileSystem*>(&other);
    return talon != nullptr && bucket_ == talon->bucket_ && cloud_provider_ == talon->cloud_provider_ &&
           coordinator_ == talon->coordinator_ && block_size_ == talon->block_size_ &&
           origin_fs_->Equals(*talon->origin_fs_);
  }

  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override {
    return origin_fs_->GetFileInfo(path);
  }

  arrow::Result<arrow::fs::FileInfoVector> GetFileInfo(const arrow::fs::FileSelector& selector) override {
    return origin_fs_->GetFileInfo(selector);
  }

  arrow::fs::FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& selector) override {
    return origin_fs_->GetFileInfoGenerator(selector);
  }

  arrow::Status CreateDir(const std::string& path, bool recursive) override {
    return origin_fs_->CreateDir(path, recursive);
  }

  arrow::Status DeleteDir(const std::string& path) override { return origin_fs_->DeleteDir(path); }

  arrow::Status DeleteDirContents(const std::string& path, bool missing_dir_ok) override {
    return origin_fs_->DeleteDirContents(path, missing_dir_ok);
  }

  arrow::Status DeleteRootDirContents() override { return origin_fs_->DeleteRootDirContents(); }

  arrow::Status DeleteFile(const std::string& path) override { return origin_fs_->DeleteFile(path); }

  arrow::Status Move(const std::string& src, const std::string& dest) override { return origin_fs_->Move(src, dest); }

  arrow::Status CopyFile(const std::string& src, const std::string& dest) override {
    return origin_fs_->CopyFile(src, dest);
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const std::string& path) override {
    ARROW_ASSIGN_OR_RAISE(auto file, OpenTalon(path, arrow::fs::kNoSize));
    return std::static_pointer_cast<arrow::io::InputStream>(std::move(file));
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const arrow::fs::FileInfo& info) override {
    ARROW_ASSIGN_OR_RAISE(auto file, OpenTalon(info.path(), info.size()));
    return std::static_pointer_cast<arrow::io::InputStream>(std::move(file));
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    return OpenTalon(path, arrow::fs::kNoSize);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    return OpenTalon(info.path(), info.size());
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    return origin_fs_->OpenOutputStream(path, metadata);
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenAppendStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    return origin_fs_->OpenAppendStream(path, metadata);
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenConditionalOutputStream(
      const std::string& path, std::shared_ptr<arrow::KeyValueMetadata> metadata) override {
    const auto conditional = std::dynamic_pointer_cast<UploadConditional>(origin_fs_);
    if (conditional == nullptr) {
      return arrow::Status::NotImplemented("Talon cannot forward conditional output stream for path '", path,
                                           "': origin filesystem type '", origin_fs_->type_name(),
                                           "' does not implement UploadConditional");
    }
    return conditional->OpenConditionalOutputStream(path, std::move(metadata));
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStreamWithUploadSize(
      const std::string& path,
      const std::shared_ptr<const arrow::KeyValueMetadata>& metadata,
      int64_t part_size) override {
    const auto sizable = std::dynamic_pointer_cast<UploadSizable>(origin_fs_);
    if (sizable == nullptr) {
      return arrow::Status::NotImplemented("Talon cannot forward sized output stream for path '", path,
                                           "' with part size ", part_size, ": origin filesystem type '",
                                           origin_fs_->type_name(), "' does not implement UploadSizable");
    }
    return sizable->OpenOutputStreamWithUploadSize(path, metadata, part_size);
  }

  std::shared_ptr<FilesystemMetrics> GetMetrics() const override {
    const auto observable = std::dynamic_pointer_cast<Observable>(origin_fs_);
    return observable == nullptr ? nullptr : observable->GetMetrics();
  }

  private:
  arrow::Result<std::string> ObjectKey(const std::string& path) const {
    const std::string prefix = bucket_ + "/";
    if (!path.starts_with(prefix)) {
      return arrow::Status::Invalid("Talon path does not belong to configured bucket ", bucket_, ": ", path);
    }
    const std::string key = path.substr(prefix.size());
    if (key.empty()) {
      return arrow::Status::Invalid("Talon object key must be non-empty");
    }
    return key;
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenTalon(const std::string& path,
                                                                        int64_t known_size) const {
    ARROW_ASSIGN_OR_RAISE(auto key, ObjectKey(path));
    if (known_size < arrow::fs::kNoSize) {
      return arrow::Status::Invalid("Invalid known Talon object size: ", known_size);
    }
    ARROW_ASSIGN_OR_RAISE(auto reader, client_->OpenObject(cloud_provider_, bucket_, key, known_size));
    return std::make_shared<TalonInputFile>(std::move(reader), io_context().pool());
  }

  const ArrowFileSystemPtr origin_fs_;
  const std::shared_ptr<TalonClient> client_;
  const std::string bucket_;
  const std::string cloud_provider_;
  const std::string coordinator_;
  const uint32_t block_size_;
};

}  // namespace

namespace internal {

arrow::Result<ArrowFileSystemPtr> MakeTalonFileSystem(const ArrowFileSystemConfig& config,
                                                      ArrowFileSystemPtr origin_fs,
                                                      std::string bucket) {
  ARROW_ASSIGN_OR_RAISE(auto client, TalonClient::Make(config.talon_coordinator, config.talon_block_size));
  return std::make_shared<TalonFileSystem>(std::move(origin_fs), std::move(client), std::move(bucket),
                                           config.cloud_provider, config.talon_coordinator, config.talon_block_size);
}

}  // namespace internal

}  // namespace milvus_storage::talon
