// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include "milvus-storage/tracing.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/fiu_local.h"
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
std::shared_ptr<const Configuration> configuration;
const folly::RequestToken storage_key("milvus-storage.tracing.v1");
// Contexts can only originate from AttachParent. Until the first attachment,
// avoid touching Folly's request-context TLS on the default disabled path.
// Never reset: old operations must retain context after provider replacement.
std::atomic<bool> contexts_seen{false};
// A failed attachment cannot allocate a replacement Folly context. Suppress
// capture conservatively until all failed guards exit, including across fibers.
std::atomic<uint32_t> failed_scopes{0};
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
  explicit OwnedAttributes(TraceScope::Attributes attributes) {
    // AttributeMap::SetAttribute is noexcept despite allocating. Use the SDK's
    // owning conversion directly so our allocation failures reach the caller's
    // tracing-only catch boundary instead of terminating inside that setter.
    for (const auto& [key, value] : attributes) {
      attributes_.insert_or_assign(
          std::string(key), opentelemetry::nostd::visit(opentelemetry::sdk::common::AttributeConverter{}, value));
    }
  }

  bool ForEachKeyValue(
      opentelemetry::nostd::function_ref<bool(opentelemetry::nostd::string_view, opentelemetry::common::AttributeValue)>
          callback) const noexcept override {
    try {
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
    } catch (...) {
      // SDK iteration callbacks are noexcept; our temporary array views must
      // not let allocation failures escape through that contract.
      return false;
    }
  }

  size_t size() const noexcept override { return attributes_.size(); }

  private:
  opentelemetry::sdk::common::AttributeMap attributes_;
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
    try {
      for (const auto& [context, values] : links) {
        if (!callback(context, values))
          return false;
      }
      return true;
    } catch (...) {
      return false;
    }
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
  std::atomic<bool> start_failed{false};
  bool Start(opentelemetry::nostd::string_view name = {},
             TraceScope::Attributes attributes = {},
             TraceScope::Links links = {}) noexcept;
  void End(const arrow::Status* status, ot::StatusCode code = ot::StatusCode::kUnset) noexcept;
  ~SpanState() noexcept {
    try {
      if (span && span->valid() && !finished) {
        auto& arrow_span = *span;
        at::UnwrapSpan(arrow_span.details.get())->SetAttribute("storage.completion.unobserved", true);
        END_SPAN(arrow_span);
      }
    } catch (...) {
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
ot::SpanContext Parent(const ContextPtr& context) noexcept {
  if (!context)
    return ot::SpanContext::GetInvalid();
  if (!context->operation)
    return context->parent;
  auto& op = context->operation;
  if (!op->Start())
    return ot::SpanContext::GetInvalid();
  try {
    return op->span && op->span->valid() ? at::UnwrapSpan(op->span->details.get())->GetContext() : Parent(op->parent);
  } catch (...) {
    return ot::SpanContext::GetInvalid();
  }
}
bool SpanState::Start(opentelemetry::nostd::string_view name,
                      TraceScope::Attributes attributes,
                      TraceScope::Links links) noexcept {
  if (start_failed.load(std::memory_order_acquire))
    return false;
  if (started.load(std::memory_order_acquire))
    return true;
  try {
    std::lock_guard<std::mutex> lock(mutex);
    if (start_failed.load(std::memory_order_acquire))
      return false;
    if (started.load(std::memory_order_relaxed))
      return true;
    auto parent_context = Parent(parent);
    if (parent_context.IsValid() && config->tracer) {
      ot::StartSpanOptions options;
      options.parent = parent_context;
      options.kind = kind;
      at::RewrapSpan(span->details.get(),
                     deferred ? config->tracer->StartSpan(deferred->name, deferred->attributes, *deferred, options)
                              : config->tracer->StartSpan(name, attributes, links, options));
    }
    deferred.reset();
    // Publish only after Arrow's span storage is complete. A lock failure in a
    // competing caller must never publish unfinished storage or retry borrowed
    // immediate creation arguments after their lifetime has ended.
    started.store(true, std::memory_order_release);
    return true;
  } catch (...) {
    start_failed.store(true, std::memory_order_release);
    return false;
  }
}
struct Data final : folly::RequestData {
  explicit Data(ContextPtr value) : context(std::move(value)) {}
  bool hasCallback() override { return false; }
  const ContextPtr context;
};
}  // namespace
ContextPtr Capture() noexcept {
  if (!contexts_seen.load(std::memory_order_relaxed) || failed_scopes.load(std::memory_order_acquire) != 0)
    return nullptr;
  try {
    const auto* request = folly::RequestContext::try_get();
    const auto* data = request ? static_cast<const Data*>(request->getContextData(storage_key)) : nullptr;
    return data ? data->context : nullptr;
  } catch (...) {
    return nullptr;
  }
}
bool HasContext() noexcept {
  if (!contexts_seen.load(std::memory_order_relaxed) || failed_scopes.load(std::memory_order_acquire) != 0)
    return false;
  try {
    const auto* request = folly::RequestContext::try_get();
    const auto* data = request ? static_cast<const Data*>(request->getContextData(storage_key)) : nullptr;
    return data && data->context;
  } catch (...) {
    return false;
  }
}
void StartCurrent() noexcept {
  auto context = Capture();
  if (context && context->operation)
    context->operation->Start();
}
TraceScope::TraceScope(ContextPtr context) noexcept : context_(std::move(context)) {
  // A captured empty context must mask unrelated context on a foreign thread.
  try {
    FIU_DO_ON(FIUKEY_TRACING_CONTEXT_ATTACH_FAIL, {
      Fail();
      return;
    });
    // Do not use HasContext here: global failure suppression may temporarily
    // hide a foreign parent which this guard must continue masking afterward.
    const auto* request = folly::RequestContext::try_get();
    const auto* data = request ? static_cast<const Data*>(request->getContextData(storage_key)) : nullptr;
    if (context_ || (data && data->context))
      scope_.emplace(storage_key, std::make_unique<Data>(context_));
  } catch (...) {
    Fail();
  }
}
TraceScope TraceScope::Deferred(opentelemetry::nostd::string_view name,
                                Attributes attributes,
                                Links links,
                                ot::SpanKind kind) noexcept {
  return TraceScope(name, attributes, links, kind, Mode::Deferred);
}
TraceScope TraceIO(opentelemetry::nostd::string_view name, TraceScope::Attributes attributes) noexcept {
  return TraceScope(name, attributes, {}, ot::SpanKind::kInternal, TraceScope::Mode::IO);
}
void TraceScope::Fail() noexcept {
  if (!failed_) {
    failed_scopes.fetch_add(1, std::memory_order_acq_rel);
    failed_ = true;
  }
  // During initialization context_ may still be the parent; dropping the owned
  // span directly avoids ever ending that parent's span by mistake.
  span_.reset();
  context_.reset();
  scope_.reset();
}
void TraceScope::Finish() noexcept { Finish(nullptr); }
void TraceScope::Finish(const arrow::Status& status) noexcept { Finish(&status); }
void TraceScope::Finish(const arrow::Status* status) noexcept {
  if (span_ && context_ && context_->operation)
    context_->operation->End(status);
  span_.reset();
  scope_.reset();
  if (failed_) {
    failed_ = false;
    failed_scopes.fetch_sub(1, std::memory_order_acq_rel);
  }
}
TraceScope AttachContext(ContextPtr context) noexcept { return TraceScope(std::move(context)); }
TraceScope AttachParent(const TraceParent& parent) noexcept {
  contexts_seen.store(true, std::memory_order_relaxed);
  try {
    FIU_RETURN_ON(FIUKEY_TRACING_CONTEXT_ATTACH_FAIL, TraceScope(TraceScope::Failed{}));
    return AttachContext(std::make_shared<Context>(Context{
        ot::SpanContext(ot::TraceId(parent.trace_id), ot::SpanId(parent.span_id), ot::TraceFlags(parent.trace_flags),
                        parent.is_remote, ot::TraceState::FromHeader(parent.tracestate)),
        nullptr}));
  } catch (...) {
    return TraceScope(TraceScope::Failed{});
  }
}
// Returning a non-OK Arrow Status itself requires allocation. As with Arrow's
// noexcept span allocation, persistent allocator exhaustion inside that
// dependency cannot be recovered by this tracing boundary.
arrow::Status SetTracerProvider(ProviderPtr provider) noexcept {
  try {
    FIU_RETURN_ON(FIUKEY_TRACING_CONFIGURATION_FAIL,
                  arrow::Status::UnknownError("tracing.configuration.fail: injected failure"));
    auto tracer = provider ? provider->GetTracer("milvus-storage", MILVUS_STORAGE_VERSION) : nullptr;
    std::lock_guard<std::mutex> lock(configuration_mutex);
    auto next = configuration ? std::make_shared<Configuration>(*configuration) : std::make_shared<Configuration>();
    next->provider = std::move(provider);
    next->tracer = std::move(tracer);
    configuration = std::move(next);
    return arrow::Status::OK();
  } catch (const std::exception& error) {
    return arrow::Status::UnknownError("Failed to set tracer provider: ", error.what());
  } catch (...) {
    return arrow::Status::UnknownError("Failed to set tracer provider: unknown exception");
  }
}
arrow::Status SetTraceOptions(const TraceOptions& options) noexcept {
  try {
    FIU_RETURN_ON(FIUKEY_TRACING_CONFIGURATION_FAIL,
                  arrow::Status::UnknownError("tracing.configuration.fail: injected failure"));
    std::lock_guard<std::mutex> lock(configuration_mutex);
    auto next = configuration ? std::make_shared<Configuration>(*configuration) : std::make_shared<Configuration>();
    next->options = options;
    configuration = std::move(next);
    return arrow::Status::OK();
  } catch (const std::exception& error) {
    return arrow::Status::UnknownError("Failed to set tracing options: ", error.what());
  } catch (...) {
    return arrow::Status::UnknownError("Failed to set tracing options: unknown exception");
  }
}
void TraceScope::Initialize(
    opentelemetry::nostd::string_view name, Attributes attributes, Links links, ot::SpanKind kind, Mode mode) noexcept {
  try {
    FIU_DO_ON(FIUKEY_TRACING_SCOPE_FAIL, {
      Fail();
      return;
    });
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
      bool create = config && config->tracer && (mode != Mode::IO || config->options.io_spans);
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
        else if (!state->Start(name, attributes, links)) {
          Fail();
          return;
        }
        context_ = std::make_shared<Context>(Context{ot::SpanContext::GetInvalid(), std::move(state)});
      }
    }
    // Suppression keeps the inherited context active without owning its span.
    FIU_DO_ON(FIUKEY_TRACING_CONTEXT_ATTACH_FAIL, {
      Fail();
      return;
    });
    scope_.emplace(storage_key, std::make_unique<Data>(context_));
  } catch (...) {
    Fail();
  }
}
bool IsEnabled(const ContextPtr& context) noexcept { return context && context->operation; }
void AccountRead(const ContextPtr& context, int64_t requested, int64_t returned) noexcept {
  if (!context || !context->operation)
    return;
  auto budget = context->operation->budget;
  budget->reads.fetch_add(1, std::memory_order_relaxed);
  budget->requested_bytes.fetch_add(std::max<int64_t>(0, requested), std::memory_order_relaxed);
  budget->returned_bytes.fetch_add(std::max<int64_t>(0, returned), std::memory_order_relaxed);
}
void EnsureStarted(const SpanPtr& span, const ContextPtr& context) noexcept {
  if (span && context && context->operation)
    context->operation->Start();
}
void SpanState::End(const arrow::Status* status, ot::StatusCode code) noexcept {
  if (!status && code == ot::StatusCode::kUnset && !started.load(std::memory_order_acquire))
    return;
  if (!Start())
    return;
  try {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (finished)
        return;
      finished = true;
    }
    if (!span || !span->valid())
      return;
    auto& arrow_span = *span;
    auto& ot_span = at::UnwrapSpan(arrow_span.details.get());
    try {
      if (status && !status->ok()) {
        // Arrow's error marker exports Status::ToString(). Preserve Storage's
        // contract: return the original status, export only its classification.
        ot_span->SetStatus(ot::StatusCode::kError);
        if (ot_span->IsRecording()) {
          if (auto detail = ExtendStatusDetail::UnwrapStatus(*status)) {
            ot_span->SetAttribute("error.type", detail->CodeAsString());
            ot_span->SetAttribute("error.retryable", detail->retryable());
          } else {
            ot_span->SetAttribute("error.type", status->CodeAsString());
          }
        }
      } else if (status) {
        MARK_SPAN(arrow_span, *status);
      } else if (code != ot::StatusCode::kUnset) {
        ot_span->SetStatus(code);
      }
      if (root) {
        ot_span->SetAttribute("storage.spans.dropped", budget->dropped.load());
        ot_span->SetAttribute("storage.io.reads", budget->reads.load());
        ot_span->SetAttribute("storage.io.requested_bytes", budget->requested_bytes.load());
        ot_span->SetAttribute("storage.io.returned_bytes", budget->returned_bytes.load());
      }
    } catch (...) {
      // Failed annotation must not prevent ending a span that was already begun.
    }
    END_SPAN(arrow_span);
  } catch (...) {
  }
}
void EndSpan(const SpanPtr& owned_span, const ContextPtr& context) noexcept {
  if (owned_span && context && context->operation)
    context->operation->End(nullptr);
}
void EndSpan(const SpanPtr& owned_span, const ContextPtr& context, const arrow::Status& status) noexcept {
  if (owned_span && context && context->operation)
    context->operation->End(&status);
}
void EndSpan(const SpanPtr& owned_span, const ContextPtr& context, ot::StatusCode status) noexcept {
  if (owned_span && context && context->operation)
    context->operation->End(nullptr, status);
}
void SetAttribute(const SpanPtr& span,
                  const ContextPtr& context,
                  opentelemetry::nostd::string_view key,
                  const opentelemetry::common::AttributeValue& value) noexcept {
  if (!span || !context || !context->operation || !context->operation->Start())
    return;
  try {
    if (span->valid())
      at::UnwrapSpan(span->details.get())->SetAttribute(key, value);
  } catch (...) {
  }
}
ot::SpanContext GetSpanContext(const ContextPtr& context) noexcept { return Parent(context); }
}  // namespace milvus_storage::tracing
