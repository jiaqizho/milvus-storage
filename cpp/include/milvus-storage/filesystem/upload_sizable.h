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

#include <memory>

#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/util/key_value_metadata.h>

namespace milvus_storage {

/// \brief Interface for filesystems that support opening output streams with specified upload sizing
///
/// This interface allows filesystems to optimize multipart uploads by specifying
/// the multipart part size and initial upload buffer capacity upfront, which is
/// particularly useful for S3 and other cloud storage systems that support
/// multipart uploads.
class UploadSizable {
  public:
  virtual ~UploadSizable() = default;

  /// \brief Open an output stream with specified sizing for multipart uploads
  /// \param path The file path
  /// \param metadata Optional metadata
  /// \param upload_part_size The multipart upload part size
  /// \param upload_buffer_size The initial upload buffer capacity
  /// \return Output stream for writing
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStreamWithUploadSize(
      const std::string& path,
      const std::shared_ptr<const arrow::KeyValueMetadata>& metadata,
      int64_t upload_part_size,
      int64_t upload_buffer_size) = 0;
};

}  // namespace milvus_storage
