// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include "milvus-storage/tracing.h"
#include "milvus-storage/common/extend_status.h"
#include "tracing/filesystem.h"
#include <arrow/util/tracing_internal.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <type_traits>
#include <vector>
#include <opentelemetry/sdk/common/attribute_utils.h>
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
// OTel's SDK owns attribute values; this view borrows them only for the duration
// of each SDK callback. In particular, strings and arrays must survive deferred
// execution and be present when the sampler receives StartSpan's attributes.
class OwnedAttributes final : public opentelemetry::common::KeyValueIterable {
  public:
  explicit OwnedAttributes(TraceScope::Attributes attributes) : attributes_(attributes) {}

  bool ForEachKeyValue(
      opentelemetry::nostd::function_ref<bool(opentelemetry::nostd::string_view, opentelemetry::common::AttributeValue)>
          callback) const noexcept override {
    for (const auto& [key, attribute] : attributes_) {
      const bool keep_going = opentelemetry::nostd::visit(
          [&](const auto& value) {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, std::vector<bool>>) {
              // vector<bool> has no contiguous bool storage to expose as a span.
              auto values = std::make_unique<bool[]>(value.size());
              std::copy(value.begin(), value.end(), values.get());
              return callback(key, opentelemetry::nostd::span<const bool>(values.get(), value.size()));
            } else if constexpr (std::is_same_v<Value, std::vector<std::string>>) {
              std::vector<opentelemetry::nostd::string_view> values(value.begin(), value.end());
              return callback(key, opentelemetry::nostd::span<const opentelemetry::nostd::string_view>(values));
            } else if constexpr (std::is_arithmetic_v<Value>) {
              return callback(key, value);
            } else if constexpr (std::is_same_v<Value, std::string>) {
              return callback(key, opentelemetry::nostd::string_view(value));
            } else {
              return callback(key, opentelemetry::nostd::span<const typename Value::value_type>(value));
            }
          },
          attribute);
      if (!keep_going)
        return false;
    }
    return true;
  }

  size_t size() const noexcept override { return attributes_.size(); }

  private:
  const opentelemetry::sdk::common::AttributeMap attributes_;
};
struct DeferredSpan final : ot::SpanContextKeyValueIterable {
  DeferredSpan(opentelemetry::nostd::string_view span_name,
               TraceScope::Attributes span_attributes,
               TraceScope::Links span_links)
      : name(span_name), attributes(span_attributes) {
    links.reserve(span_links.size());
    for (const auto& [context, values] : span_links) links.emplace_back(context, values);
  }

  bool ForEachKeyValue(
      opentelemetry::nostd::function_ref<bool(ot::SpanContext, const opentelemetry::common::KeyValueIterable&)>
          callback) const noexcept override {
    for (const auto& [context, values] : links) {
      if (!callback(context, values))
        return false;
    }
    return true;
  }

  size_t size() const noexcept override { return links.size(); }

  const std::string name;
  const OwnedAttributes attributes;
  std::vector<std::pair<ot::SpanContext, OwnedAttributes>> links;
};
struct SpanState {
  mutable std::mutex mutex;
  // Arrow owns the span representation; the SDK span starts only when work
  // begins. Aliasing shared pointers expose this Arrow span without another owner allocation.
  std::optional<arrow::util::tracing::Span> span;
  ContextPtr parent;
  std::shared_ptr<const Configuration> config;
  std::shared_ptr<Budget> budget;
  // Immediate spans borrow creation arguments; only deferred spans need copies.
  std::unique_ptr<DeferredSpan> deferred;
  ot::SpanKind kind = ot::SpanKind::kInternal;
  bool finished = false;
  bool root = false;
  // Publishes the immutable span pointer (or a disabled decision) once Start
  // completes. Keep this beside the other flags to reuse their padding.
  std::atomic<bool> started{false};
  void Start(opentelemetry::nostd::string_view name = {},
             TraceScope::Attributes attributes = {},
             TraceScope::Links links = {});
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
void SpanState::Start(opentelemetry::nostd::string_view name,
                      TraceScope::Attributes attributes,
                      TraceScope::Links links) {
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
  options.kind = kind;
  at::RewrapSpan(span->details.get(),
                 deferred ? config->tracer->StartSpan(deferred->name, deferred->attributes, *deferred, options)
                          : config->tracer->StartSpan(name, attributes, links, options));
  deferred.reset();
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
TraceScope::TraceScope(ContextPtr context) : context_(std::move(context)) {
  // A captured empty context must mask unrelated context on a foreign thread.
  if (context_ || HasContext())
    scope_.emplace(storage_key, std::make_unique<Data>(context_));
}
TraceScope TraceScope::Deferred(opentelemetry::nostd::string_view name,
                                Attributes attributes,
                                Links links,
                                ot::SpanKind kind) {
  return TraceScope(name, attributes, links, kind, Mode::Deferred);
}
TraceScope TraceIO(opentelemetry::nostd::string_view name, TraceScope::Attributes attributes) {
  return TraceScope(name, attributes, {}, ot::SpanKind::kInternal, TraceScope::Mode::IO);
}
void TraceScope::Finish(const std::optional<arrow::Status>& status) {
  if (span_)
    EndSpan(span_, context_, status);
  span_.reset();
  scope_.reset();
}
TraceScope AttachContext(ContextPtr context) { return TraceScope(std::move(context)); }
TraceScope AttachParent(const TraceParent& parent) {
  contexts_seen.store(true, std::memory_order_relaxed);
  return AttachContext(std::make_shared<Context>(Context{
      ot::SpanContext(ot::TraceId(parent.trace_id), ot::SpanId(parent.span_id), ot::TraceFlags(parent.trace_flags),
                      parent.is_remote, ot::TraceState::FromHeader(parent.tracestate)),
      nullptr}));
}
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
void TraceScope::Initialize(
    opentelemetry::nostd::string_view name, Attributes attributes, Links links, ot::SpanKind kind, Mode mode) {
  const bool root = !context_->operation;
  if (!context_->disabled && (!root || context_->parent.IsValid())) {
    std::shared_ptr<const Configuration> config;
    std::shared_ptr<Budget> budget;
    if (root) {
      std::lock_guard<std::mutex> lock(configuration_mutex);
      config = configuration;
    } else {
      config = context_->operation->config;
      budget = context_->operation->budget;
    }
    bool create = config->tracer && (mode != Mode::IO || config->options.io_spans);
    if (!root && create &&
        budget->used.fetch_add(1, std::memory_order_relaxed) >=
            std::max<uint32_t>(1, config->options.max_spans_per_operation)) {
      budget->dropped.fetch_add(1, std::memory_order_relaxed);
      create = false;
    }
    if (root && !create) {
      // Retain the disabled decision even if the host replaces its provider.
      context_ = std::make_shared<Context>(Context{context_->parent, nullptr, true});
    } else if (create) {
      if (root) {
        budget = std::make_shared<Budget>();
        budget->used.store(1, std::memory_order_relaxed);
      }
      auto state = std::make_shared<SpanState>();
      state->parent = context_;
      state->kind = kind;
      state->config = std::move(config);
      state->budget = std::move(budget);
      state->root = root;
      state->span.emplace();
      span_ = SpanPtr(state, &*state->span);
      if (mode == Mode::Deferred)
        state->deferred = std::make_unique<DeferredSpan>(name, attributes, links);
      else
        state->Start(name, attributes, links);
      context_ = std::make_shared<Context>(Context{ot::SpanContext::GetInvalid(), std::move(state)});
    }
  }
  // Suppression keeps the inherited context active without owning its span.
  scope_.emplace(storage_key, std::make_unique<Data>(context_));
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
void SetAttribute(const SpanPtr& span,
                  const ContextPtr& context,
                  opentelemetry::nostd::string_view key,
                  const opentelemetry::common::AttributeValue& value) {
  EnsureStarted(span, context);
  if (span && span->valid())
    at::UnwrapSpan(span->details.get())->SetAttribute(key, value);
}
ot::SpanContext GetSpanContext(const ContextPtr& context) { return Parent(context); }
}  // namespace milvus_storage::tracing
