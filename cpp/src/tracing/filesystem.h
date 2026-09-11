// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arrow/io/interfaces.h>
#include <memory>
#include <string>
#include "milvus-storage/tracing.h"
namespace milvus_storage::tracing {
// I/O instrumentation shares the operation's provider/options snapshot. A
// suppressed I/O span still propagates its parent and contributes read totals.
TraceScope TraceIO(opentelemetry::nostd::string_view name, TraceScope::Attributes attributes = {}) noexcept;
void AccountRead(const ContextPtr& context, int64_t requested, int64_t returned) noexcept;

// The wrapper stores the backend name only, never a request context or path.
std::shared_ptr<arrow::io::RandomAccessFile> WrapFile(std::shared_ptr<arrow::io::RandomAccessFile> file,
                                                      std::string backend);
}  // namespace milvus_storage::tracing
