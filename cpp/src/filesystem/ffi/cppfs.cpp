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
#include <cstdint>

#include "milvus-storage/filesystem/fs.h"
#include <arrow/buffer.h>

extern "C" {
    typedef void *FileSystemHandle;
    typedef void *FileSystemWriterHandle;
    typedef void *FileSystemConfig;

    FileSystemHandle fscpp_create(FileSystemConfig config);

    FileSystemWriterHandle fscpp_open_writer(FileSystemHandle ptr, const uint8_t* path_ptr, uint64_t path_len);

    // C-ABI: write data from pointer + size, return number of bytes written or negative error code
    uint64_t fscpp_write(FileSystemWriterHandle ptr, const uint8_t* data, uint64_t size);
    void fscpp_flush(FileSystemWriterHandle ptr);
    void fscpp_close(FileSystemWriterHandle ptr);

    uint64_t fscpp_head_object(FileSystemHandle ptr, const uint8_t* path_ptr, uint64_t path_len);
    void fscpp_get_object(FileSystemHandle ptr,
                              const uint8_t* path_ptr,
                              uint64_t path_len,
                              uint64_t start,
                              uint8_t* out_data,
                              uint64_t out_size);

    void fscpp_release(FileSystemHandle ptr);
};


FileSystemHandle fscpp_create(FileSystemConfig /*config*/) {
    try {
        auto conf = milvus_storage::ArrowFileSystemConfig();
        conf.storage_type = "local";

        milvus_storage::ArrowFileSystemSingleton::GetInstance().Init(conf);
        auto fsptr = milvus_storage::ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();
        auto raw_ptr = fsptr.get();

        return raw_ptr;
    } catch (const std::exception& e) {
        printf("Exception in fscpp_create: %s\n", e.what());
        return nullptr;
    }
}


FileSystemWriterHandle fscpp_open_writer(FileSystemHandle ptr, const uint8_t* path_ptr, uint64_t path_len)
{
    auto* fs = reinterpret_cast<arrow::fs::FileSystem*>(ptr);
    std::string path(reinterpret_cast<const char*>(path_ptr), path_len);
    auto output_stream_result = fs->OpenOutputStream(path);
    if (!output_stream_result.ok()) {
        return nullptr;
    }
    auto output_stream = output_stream_result.ValueOrDie();
    auto rc = reinterpret_cast<FileSystemWriterHandle>(output_stream.get());
    output_stream.reset();  
    return rc;
}

uint64_t fscpp_write(FileSystemWriterHandle ptr, const uint8_t* data, uint64_t size) {
    auto* output_stream = reinterpret_cast<arrow::io::OutputStream*>(ptr);
    auto write_status = output_stream->Write(data, size);
    if (!write_status.ok()) {
        return static_cast<uint64_t>(-1);
    }

    return size;
}

void fscpp_flush(FileSystemWriterHandle ptr) {
    auto* output_stream = reinterpret_cast<arrow::io::OutputStream*>(ptr);
    output_stream->Flush();
}

void fscpp_close(FileSystemWriterHandle ptr) {
    auto* output_stream = reinterpret_cast<arrow::io::OutputStream*>(ptr);
    output_stream->Close();
}

uint64_t fscpp_head_object(FileSystemHandle ptr, const uint8_t* path_ptr, uint64_t path_len) {
    auto* fs = reinterpret_cast<arrow::fs::FileSystem*>(ptr);
    std::string path(reinterpret_cast<const char*>(path_ptr), path_len);
    auto info_result = fs->GetFileInfo(path);
    if (!info_result.ok()) {
        return static_cast<uint64_t>(-1);
    }
    auto info = info_result.ValueOrDie();
    return static_cast<uint64_t>(info.size());
}

void fscpp_get_object(FileSystemHandle ptr,
                              const uint8_t* path_ptr,
                              uint64_t path_len,
                              uint64_t start,
                              uint64_t out_size,
                              uint8_t* out_data)
{
    auto* fs = reinterpret_cast<arrow::fs::FileSystem*>(ptr);
    std::string path(reinterpret_cast<const char*>(path_ptr), path_len);

    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> input_file_result;
    std::shared_ptr<arrow::io::RandomAccessFile> input_file;

    arrow::Result<std::shared_ptr<arrow::Buffer>> read_result;
    std::shared_ptr<arrow::Buffer> out_buffer = nullptr;
    int64_t read_size = 0;
    
    input_file_result = fs->OpenInputFile(path);
    if (!input_file_result.ok()) {
        goto failed;
    }

    input_file = input_file_result.ValueOrDie();
    read_result = input_file->ReadAt(start, out_size);
    if (!read_result.ok()) {
        input_file->Close();
        goto failed;
    }

    out_buffer = read_result.ValueOrDie();
    read_size = out_buffer->size();

    if (read_size != out_size) {
        goto failed;
    }

    // TBD: copy data
    memcpy(out_data, out_buffer->data(), out_size);

    // won't fail close here
    input_file->Close();
    return;

failed: 
    out_data = nullptr;
    out_size = 0;
    return;
}

void fscpp_release(FileSystemHandle ptr) {
    
}