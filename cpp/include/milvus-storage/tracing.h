// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <opentelemetry/common/attribute_value.h>
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

// Storage instrumentation and execution-context propagation.
struct Context;
using ContextPtr = std::shared_ptr<const Context>;
ContextPtr Capture();
// Check the active flow without copying an owning context snapshot.
bool HasContext();
void StartCurrent();

// Spans are Arrow's actual thin wrapper. The separate context owns request
// propagation, the provider snapshot, and per-operation accounting.
using SpanPtr = std::shared_ptr<arrow::util::tracing::Span>;

void EnsureStarted(const SpanPtr& span, const ContextPtr& context);
// A scope-only end leaves status unset. Explicit I/O/completion sites may pass
// their actual status. Ending an unstarted deferred scope without a result is inert.
void EndSpan(const SpanPtr& span, const ContextPtr& context, const std::optional<arrow::Status>& status = std::nullopt);
bool IsEnabled(const ContextPtr& context);
// Setting an attribute materializes a deferred span if it has not started yet.
void SetAttribute(const SpanPtr& span,
                  const ContextPtr& context,
                  opentelemetry::nostd::string_view key,
                  const opentelemetry::common::AttributeValue& value);
// Resolve the span identity (or inherited parent). This starts a deferred span
// when necessary to obtain its actual span ID; Capture() itself does not start it.
opentelemetry::trace::SpanContext GetSpanContext(const ContextPtr& context);

// Own a span for the current C++ scope. The destructor ends only that span,
// then restores the previous Folly context. It never invokes business code,
// catches exceptions, or interprets return values.
class TraceScope {
  public:
  using Attributes =
      std::initializer_list<std::pair<opentelemetry::nostd::string_view, opentelemetry::common::AttributeValue>>;
  using Links = std::initializer_list<std::pair<opentelemetry::trace::SpanContext, Attributes>>;
  explicit TraceScope(opentelemetry::nostd::string_view name,
                      Attributes attributes = {},
                      Links links = {},
                      opentelemetry::trace::SpanKind kind = opentelemetry::trace::SpanKind::kInternal)
      : TraceScope(name, attributes, links, kind, Mode::Scoped) {}
  // Start when captured work executes, rather than when its SemiFuture is built.
  // Names, attributes and links are copied; destroying unexecuted work emits nothing.
  static TraceScope Deferred(opentelemetry::nostd::string_view name,
                             Attributes attributes = {},
                             Links links = {},
                             opentelemetry::trace::SpanKind kind = opentelemetry::trace::SpanKind::kInternal);
  ~TraceScope() {
    if (span_)
      EndSpan(span_, context_);
  }
  TraceScope(const TraceScope&) = delete;
  TraceScope& operator=(const TraceScope&) = delete;
  TraceScope(TraceScope&&) = delete;
  TraceScope& operator=(TraceScope&&) = delete;

  const SpanPtr& span() const { return span_; }
  const ContextPtr& context() const { return context_; }
  // Call after an async completion callback has captured span()/context().
  // Only relinquishes span completion; context restoration still happens here.
  void ReleaseSpan() { span_.reset(); }
  // End this scope early and restore the previous context. Inner guards must
  // have exited first, exactly as for normal stack destruction.
  void Finish(const std::optional<arrow::Status>& status = std::nullopt);

  private:
  enum class Mode { Scoped, Deferred, IO };
  TraceScope(opentelemetry::nostd::string_view name,
             Attributes attributes,
             Links links,
             opentelemetry::trace::SpanKind kind,
             Mode mode)
      : context_(Capture()) {
    // Keep the default disabled path local to the caller: it needs no span,
    // configuration snapshot, attributes copy, or request-context guard.
    if (context_)
      Initialize(name, attributes, links, kind, mode);
  }
  void Initialize(opentelemetry::nostd::string_view name,
                  Attributes attributes,
                  Links links,
                  opentelemetry::trace::SpanKind kind,
                  Mode mode);
  explicit TraceScope(ContextPtr context);
  ContextPtr context_;
  SpanPtr span_;
  std::optional<folly::ShallowCopyRequestContextScopeGuard> scope_;
  friend TraceScope AttachContext(ContextPtr context);
  friend TraceScope TraceIO(opentelemetry::nostd::string_view name, Attributes attributes);
};

// Attach-only scopes never create or end the parent's span or modify OTel TLS.
// Captured empty contexts and invalid parents mask enclosing scopes. Destroy
// guards in stack order on the same execution flow (including Folly fibers).
[[nodiscard]] TraceScope AttachContext(ContextPtr context);
[[nodiscard]] TraceScope AttachParent(const TraceParent& parent);

// Restores only Storage-owned data, preserving other RequestContext keys.
template <typename F>
auto Bind(F&& fn) {
  return [context = Capture(), fn = std::forward<F>(fn)](auto&&... args) mutable -> decltype(auto) {
    auto scope = AttachContext(context);
    StartCurrent();
    return fn(std::forward<decltype(args)>(args)...);
  };
}

}  // namespace milvus_storage::tracing
