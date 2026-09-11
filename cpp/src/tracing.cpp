// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include "milvus-storage/tracing.h"
#include "milvus-storage/common/extend_status.h"
#include <arrow/util/tracing_internal.h>
#include <atomic>
#include <mutex>
#include <opentelemetry/trace/trace_state.h>

#ifndef ARROW_WITH_OPENTELEMETRY
#error "Storage tracing requires Arrow built with OpenTelemetry enabled"
#endif

namespace milvus_storage::tracing {
namespace ot = opentelemetry::trace;
namespace at = arrow::internal::tracing;
namespace {
struct Configuration {
  ProviderPtr provider;
  opentelemetry::nostd::shared_ptr<ot::Tracer> tracer;
  TraceOptions options;
};
std::mutex configuration_mutex;
std::shared_ptr<const Configuration> configuration = std::make_shared<Configuration>();
const folly::RequestToken storage_key("milvus-storage.tracing.v1");
// Contexts can only originate from AttachParent. Until the first attachment,
// avoid touching Folly's request-context TLS on the default disabled path.
// Never reset: old operations must retain context after provider replacement.
std::atomic<bool> contexts_seen{false};
struct Budget {
  std::atomic<uint64_t> used{0};
  std::atomic<int64_t> dropped{0};
  std::atomic<int64_t> reads{0}, requested_bytes{0}, returned_bytes{0};
};
struct SpanState {
  mutable std::mutex mutex;
  // Arrow owns the span representation; the SDK span starts only when work
  // begins. Aliasing shared pointers expose this Arrow span without another owner allocation.
  std::optional<arrow::util::tracing::Span> span;
  ContextPtr parent;
  std::shared_ptr<const Configuration> config;
  std::shared_ptr<Budget> budget;
  const char* name;
  const char* operation = nullptr;
  const char* format = nullptr;
  ot::SpanContext link = ot::SpanContext::GetInvalid();
  bool finished = false;
  bool root = false;
  // Publishes the immutable span pointer (or a disabled decision) once Start
  // completes. Keep this beside the other flags to reuse their padding.
  std::atomic<bool> started{false};
  void Start();
  ~SpanState() {
    if (span && span->valid() && !finished) {
      auto& arrow_span = *span;
      at::UnwrapSpan(arrow_span.details.get())->SetAttribute("storage.completion.unobserved", true);
      END_SPAN(arrow_span);
    }
  }
};
}  // namespace
struct Context {
  ot::SpanContext parent = ot::SpanContext::GetInvalid();
  std::shared_ptr<SpanState> operation;
  // Freeze a disabled operation without allocating SpanState, Budget or a mutex.
  bool disabled = false;
};
namespace {
ot::SpanContext Parent(const ContextPtr& context) {
  if (!context)
    return ot::SpanContext::GetInvalid();
  if (!context->operation)
    return context->parent;
  auto& op = context->operation;
  op->Start();
  return op->span && op->span->valid() ? at::UnwrapSpan(op->span->details.get())->GetContext() : Parent(op->parent);
}
void SpanState::Start() {
  if (started.load(std::memory_order_acquire))
    return;
  std::lock_guard<std::mutex> lock(mutex);
  if ((span && span->valid()) || finished) {
    started.store(true, std::memory_order_release);
    return;
  }
  auto parent_context = Parent(parent);
  if (!parent_context.IsValid() || !config->tracer) {
    started.store(true, std::memory_order_release);
    return;
  }
  ot::StartSpanOptions options;
  options.parent = parent_context;
  auto& ot_span = link.IsValid()
                      ? at::RewrapSpan(span->details.get(), config->tracer->StartSpan(name, {}, {{link, {}}}, options))
                      : at::RewrapSpan(span->details.get(), config->tracer->StartSpan(name, options));
  if (ot_span && ot_span->IsRecording()) {
    if (operation)
      ot_span->SetAttribute("storage.operation", operation);
    if (format)
      ot_span->SetAttribute("storage.format", format);
  }
  started.store(true, std::memory_order_release);
}
struct Data final : folly::RequestData {
  explicit Data(ContextPtr value) : context(std::move(value)) {}
  bool hasCallback() override { return false; }
  const ContextPtr context;
};
}  // namespace
ContextPtr Capture() {
  if (!contexts_seen.load(std::memory_order_relaxed))
    return nullptr;
  auto* data = static_cast<Data*>(folly::RequestContext::get()->getContextData(storage_key));
  return data ? data->context : nullptr;
}
bool HasContext() {
  if (!contexts_seen.load(std::memory_order_relaxed))
    return false;
  auto* data = static_cast<Data*>(folly::RequestContext::get()->getContextData(storage_key));
  return data && data->context;
}
void StartCurrent() {
  auto context = Capture();
  if (context && context->operation)
    context->operation->Start();
}
ContextScope::ContextScope(ContextPtr context) {
  // Common disabled path does not allocate a RequestContext. A captured empty
  // context must still mask unrelated context on a foreign completion thread.
  if (context || HasContext())
    scope_.emplace(storage_key, std::make_unique<Data>(std::move(context)));
}

struct TraceScope::Impl {
  explicit Impl(const TraceParent& parent)
      : scope(std::make_shared<Context>(Context{ot::SpanContext(ot::TraceId(parent.trace_id),
                                                                ot::SpanId(parent.span_id),
                                                                ot::TraceFlags(parent.trace_flags),
                                                                parent.is_remote,
                                                                ot::TraceState::FromHeader(parent.tracestate)),
                                                nullptr})) {}
  ContextScope scope;
};
TraceScope::TraceScope(const TraceParent& parent) {
  contexts_seen.store(true, std::memory_order_relaxed);
  impl_ = std::make_unique<Impl>(parent);
}
TraceScope::~TraceScope() = default;
TraceScope AttachParent(const TraceParent& parent) { return TraceScope(parent); }
void SetTracerProvider(ProviderPtr provider) {
  auto tracer = provider ? provider->GetTracer("milvus-storage", MILVUS_STORAGE_VERSION) : nullptr;
  std::lock_guard<std::mutex> lock(configuration_mutex);
  auto next = std::make_shared<Configuration>(*configuration);
  next->provider = std::move(provider);
  next->tracer = std::move(tracer);
  configuration = std::move(next);
}
void SetTraceOptions(const TraceOptions& options) {
  std::lock_guard<std::mutex> lock(configuration_mutex);
  auto next = std::make_shared<Configuration>(*configuration);
  next->options = options;
  configuration = std::move(next);
}
SpanPtr StartSpan(ContextPtr& context,
                  const char* name,
                  bool lazy,
                  bool io,
                  ot::SpanContext link,
                  const char* operation,
                  const char* format,
                  bool create_span) {
  if (!context || context->disabled || !create_span)
    return nullptr;
  const bool root = !context->operation;
  std::shared_ptr<const Configuration> config;
  std::shared_ptr<Budget> budget;
  if (!root) {
    config = context->operation->config;
    // Children retain their parent's fixed configuration even if the host
    // injects a provider later. No child state is needed for suppressed spans.
    if (!config->tracer || (io && !config->options.io_spans))
      return nullptr;
    budget = context->operation->budget;
    if (budget->used.fetch_add(1, std::memory_order_relaxed) >=
        std::max<uint32_t>(1, config->options.max_spans_per_operation)) {
      budget->dropped.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
  } else {
    if (!context->parent.IsValid())
      return nullptr;
    {
      std::lock_guard<std::mutex> lock(configuration_mutex);
      config = configuration;
    }
    if (!config->tracer || (io && !config->options.io_spans)) {
      context = std::make_shared<Context>(Context{context->parent, nullptr, true});
      return nullptr;
    }
    budget = std::make_shared<Budget>();
  }
  auto state = std::make_shared<SpanState>();
  state->parent = context;
  state->name = name;
  state->operation = operation;
  state->format = format;
  state->link = std::move(link);
  state->config = std::move(config);
  state->budget = std::move(budget);
  state->root = root;
  if (state->root)
    state->budget->used.store(1, std::memory_order_relaxed);
  state->span.emplace();
  SpanPtr span(state, &*state->span);
  context = std::make_shared<Context>(Context{ot::SpanContext::GetInvalid(), std::move(state)});
  if (!lazy)
    EnsureStarted(span, context);
  return span;
}
bool IsEnabled(const ContextPtr& context) { return context && context->operation; }
void AccountRead(const ContextPtr& context, int64_t requested, int64_t returned) {
  if (!context || !context->operation)
    return;
  auto budget = context->operation->budget;
  budget->reads.fetch_add(1, std::memory_order_relaxed);
  budget->requested_bytes.fetch_add(std::max<int64_t>(0, requested), std::memory_order_relaxed);
  budget->returned_bytes.fetch_add(std::max<int64_t>(0, returned), std::memory_order_relaxed);
}
void EnsureStarted(const SpanPtr& span, const ContextPtr& context) {
  if (span)
    context->operation->Start();
}
void EndSpan(const SpanPtr& owned_span, const ContextPtr& context, const std::optional<arrow::Status>& status) {
  if (!owned_span)
    return;
  auto op = context->operation;
  if (!status && !op->started.load(std::memory_order_acquire))
    return;
  op->Start();
  {
    std::lock_guard<std::mutex> lock(op->mutex);
    if (op->finished)
      return;
    op->finished = true;
  }
  if (!op->span || !op->span->valid())
    return;
  auto& arrow_span = *op->span;
  auto& span = at::UnwrapSpan(arrow_span.details.get());
  if (status && !status->ok()) {
    // Arrow's error marker exports Status::ToString(). Preserve Storage's
    // contract: return the original status, export only its classification.
    span->SetStatus(ot::StatusCode::kError);
    if (span->IsRecording()) {
      if (auto detail = ExtendStatusDetail::UnwrapStatus(*status)) {
        span->SetAttribute("error.type", detail->CodeAsString());
        span->SetAttribute("error.retryable", detail->retryable());
      } else {
        span->SetAttribute("error.type", status->CodeAsString());
      }
    }
  } else if (status) {
    MARK_SPAN(arrow_span, *status);
  }
  if (op->root) {
    span->SetAttribute("storage.spans.dropped", op->budget->dropped.load());
    span->SetAttribute("storage.io.reads", op->budget->reads.load());
    span->SetAttribute("storage.io.requested_bytes", op->budget->requested_bytes.load());
    span->SetAttribute("storage.io.returned_bytes", op->budget->returned_bytes.load());
  }
  END_SPAN(arrow_span);
}
void SetAttribute(const SpanPtr& span, const ContextPtr& context, const char* key, int64_t value) {
  EnsureStarted(span, context);
  if (span && span->valid())
    at::UnwrapSpan(span->details.get())->SetAttribute(key, value);
}
void SetAttribute(const SpanPtr& span, const ContextPtr& context, const char* key, const char* value) {
  EnsureStarted(span, context);
  if (span && span->valid())
    at::UnwrapSpan(span->details.get())->SetAttribute(key, value);
}
ot::SpanContext GetSpanContext(const ContextPtr& context) { return Parent(context); }
}  // namespace milvus_storage::tracing
