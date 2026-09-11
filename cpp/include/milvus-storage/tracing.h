// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <opentelemetry/trace/tracer_provider.h>

#include <optional>
#include <utility>
#include <folly/io/async/Request.h>
#include <arrow/status.h>
#include <arrow/util/tracing.h>
#include <opentelemetry/trace/tracer.h>

namespace milvus_storage::tracing {

struct TraceParent {
  std::array<uint8_t, 16> trace_id{};
  std::array<uint8_t, 8> span_id{};
  uint8_t trace_flags = 0;
  std::string tracestate;
  bool is_remote = false;
};

using ProviderPtr = opentelemetry::nostd::shared_ptr<opentelemetry::trace::TracerProvider>;

// Storage never installs a global provider, creates an exporter, or shuts down
// an injected provider. The host must use a matching OTel C++ ABI.
void SetTracerProvider(ProviderPtr provider);

struct TraceOptions {
  bool io_spans = true;
  // Includes the operation span. Excess children are aggregated on the operation.
  uint32_t max_spans_per_operation = 256;
};
// Changes apply to subsequent operations; active operations retain their snapshot.
void SetTraceOptions(const TraceOptions& options);

class TraceScope {
  public:
  ~TraceScope();
  TraceScope(const TraceScope&) = delete;
  TraceScope& operator=(const TraceScope&) = delete;
  TraceScope(TraceScope&&) = delete;
  TraceScope& operator=(TraceScope&&) = delete;

  private:
  struct Impl;
  explicit TraceScope(const TraceParent& parent);
  std::unique_ptr<Impl> impl_;
  friend TraceScope AttachParent(const TraceParent& parent);
};

// Only changes the Storage RequestContext. Does not create/end a parent span
// or modify OTel TLS. Invalid parents mask enclosing scopes. Destroy on the
// same execution flow in stack order; Folly fibers may suspend with this guard.
[[nodiscard]] TraceScope AttachParent(const TraceParent& parent);

// Storage instrumentation and execution-context propagation.
struct Context;
using ContextPtr = std::shared_ptr<const Context>;
ContextPtr Capture();
// Check the active flow without copying an owning context snapshot.
bool HasContext();
void StartCurrent();

class ContextScope {
  public:
  explicit ContextScope(ContextPtr context);

  private:
  std::optional<folly::ShallowCopyRequestContextScopeGuard> scope_;
};

// Spans are Arrow's actual thin wrapper. The separate context owns request
// propagation, the provider snapshot, and per-operation accounting.
using SpanPtr = std::shared_ptr<arrow::util::tracing::Span>;

// Start from Capture(). The context is updated even when tracing is disabled so
// asynchronous work retains that decision. A null span never owns its parent.
SpanPtr StartSpan(ContextPtr& context,
                  const char* name,
                  bool lazy = false,
                  bool io = false,
                  opentelemetry::trace::SpanContext link = opentelemetry::trace::SpanContext::GetInvalid(),
                  const char* operation = nullptr,
                  const char* format = nullptr,
                  bool create_span = true);
void EnsureStarted(const SpanPtr& span, const ContextPtr& context);
// A scope-only end leaves status unset. Explicit I/O/completion sites may pass
// their actual status. Ending an unstarted lazy scope without a result is inert.
void EndSpan(const SpanPtr& span, const ContextPtr& context, const std::optional<arrow::Status>& status = std::nullopt);
bool IsEnabled(const ContextPtr& context);
void AccountRead(const ContextPtr& context, int64_t requested, int64_t returned);
void SetAttribute(const SpanPtr& span, const ContextPtr& context, const char* key, int64_t value);
void SetAttribute(const SpanPtr& span, const ContextPtr& context, const char* key, const char* value);
opentelemetry::trace::SpanContext GetSpanContext(const ContextPtr& context);

// Restores only Storage-owned data, preserving other RequestContext keys.
template <typename F>
auto Bind(F&& fn) {
  return [context = Capture(), fn = std::forward<F>(fn)](auto&&... args) mutable -> decltype(auto) {
    ContextScope scope(context);
    StartCurrent();
    return fn(std::forward<decltype(args)>(args)...);
  };
}

}  // namespace milvus_storage::tracing
