// Copyright 2026 Zilliz
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

#include "bridge_util.h"

#include <cerrno>
#include <charconv>
#include <string>

#include <arrow/util/io_util.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/ffi_c.h"

namespace milvus_storage {
namespace {

std::string JoinContextAndMessage(std::string_view context, std::string_view message) {
  if (context.empty()) {
    return std::string(message);
  }
  if (message.empty()) {
    return std::string(context);
  }
  std::string result;
  result.reserve(context.size() + 2 + message.size());
  result.append(context);
  result.append(": ");
  result.append(message);
  return result;
}

arrow::Status MakeExtendErrorWithContext(std::string_view context, const arrow::Status& status) {
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  auto full_message = JoinContextAndMessage(context, status.message());
  return MakeExtendError(detail->code(), full_message, full_message);
}

arrow::Status MakeIOErrorWithContext(std::string_view context, const arrow::Status& status) {
  auto result = arrow::Status::IOError(JoinContextAndMessage(context, status.message()));
  if (arrow::internal::ErrnoFromStatus(status) == ENOENT) {
    return result.WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
  }
  return result;
}

}  // namespace

arrow::Status MakeBridgeErrorStatus(std::string_view message) {
  auto marker_pos = message.find(kFFIErrorCodeMarker);
  if (marker_pos == std::string_view::npos) {
    return arrow::Status::IOError(message);
  }

  auto code_start = marker_pos + kFFIErrorCodeMarker.size();
  auto code_end = code_start;
  while (code_end < message.size() && message[code_end] >= '0' && message[code_end] <= '9') {
    ++code_end;
  }
  if (code_end == code_start) {
    return arrow::Status::IOError(message);
  }

  int ffi_err_code = 0;
  auto parse_result = std::from_chars(message.data() + code_start, message.data() + code_end, ffi_err_code);
  if (parse_result.ec != std::errc()) {
    return arrow::Status::IOError(message);
  }

  auto message_start = code_end;
  if (message_start < message.size() && message[message_start] == ';') {
    ++message_start;
  }
  if (message_start < message.size() && message[message_start] == ' ') {
    ++message_start;
  }
  std::string clean_message;
  clean_message.reserve(message.size());
  clean_message.append(message.substr(0, marker_pos));
  clean_message.append(message.substr(message_start));
  if (clean_message.empty()) {
    clean_message = "Unknown FFI error";
  }

  if (ffi_err_code == LOON_FILE_NOT_FOUND) {
    return arrow::Status::IOError(clean_message).WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
  }
  if (auto code = ExtendStatusCodeFromInt(ffi_err_code); code.has_value()) {
    return MakeExtendError(*code, clean_message, clean_message);
  }
  return arrow::Status::IOError(clean_message);
}

arrow::Status MakeBridgeErrorStatus(std::string_view context, std::string_view message) {
  return MakeBridgeErrorStatus(context, MakeBridgeErrorStatus(message));
}

arrow::Status MakeBridgeErrorStatus(std::string_view context, const arrow::Status& status) {
  if (status.ok()) {
    return arrow::Status::OK();
  }
  if (ExtendStatusDetail::UnwrapStatus(status)) {
    return MakeExtendErrorWithContext(context, status);
  }
  if (arrow::internal::ErrnoFromStatus(status) == ENOENT) {
    return MakeIOErrorWithContext(context, status);
  }
  auto parsed_status = MakeBridgeErrorStatus(status.message());
  if (ExtendStatusDetail::UnwrapStatus(parsed_status)) {
    return MakeExtendErrorWithContext(context, parsed_status);
  }
  return MakeIOErrorWithContext(context, parsed_status);
}

}  // namespace milvus_storage
