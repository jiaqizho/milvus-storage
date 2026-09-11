// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <folly/ScopeGuard.h>
#include <folly/futures/Future.h>
#include <future>
#include <stdexcept>
#include <folly/executors/ManualExecutor.h>
#include <arrow/io/memory.h>
#include <arrow/util/tracing_internal.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/fibers/Baton.h>
#include <folly/fibers/FiberManager.h>
#include <folly/fibers/SimpleLoopController.h>
#include <opentelemetry/exporters/memory/in_memory_span_exporter.h>
#include <opentelemetry/sdk/trace/simple_processor.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/sdk/trace/samplers/parent.h>
#include <opentelemetry/sdk/trace/samplers/always_on.h>
#include "milvus-storage/tracing.h"
#include "milvus-storage/common/fiu_local.h"
#include "tracing_bridge.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/writer.h"
#include "test_env.h"
#include "milvus-storage/common/extend_status.h"
#include "tracing/filesystem.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/format/parquet/folly_arrow_executor.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/format_reader_cache.h"

namespace milvus_storage::tracing {
namespace {
namespace sdk = opentelemetry::sdk::trace;
namespace ot = opentelemetry::trace;
using Memory = opentelemetry::exporter::memory::InMemorySpanData;
TraceParent Parent(uint8_t id) {
  TraceParent parent;
  parent.trace_id[0] = id;
  parent.span_id[0] = id;
  parent.trace_flags = 1;
  parent.is_remote = true;
  parent.tracestate = "vendor=value";
  return parent;
}
ot::TraceId ExpectedTrace(uint8_t id) {
  const auto parent = Parent(id);
  return ot::TraceId(opentelemetry::nostd::span<const uint8_t, 16>(parent.trace_id.data(), 16));
}
ot::SpanId ExpectedSpan(uint8_t id) {
  const auto parent = Parent(id);
  return ot::SpanId(opentelemetry::nostd::span<const uint8_t, 8>(parent.span_id.data(), 8));
}
ProviderPtr Provider(std::shared_ptr<Memory>& data, bool parent_based = false) {
  auto exporter = std::make_unique<opentelemetry::exporter::memory::InMemorySpanExporter>(4096);
  data = exporter->GetData();
  auto processor = std::make_unique<sdk::SimpleSpanProcessor>(std::move(exporter));
  std::unique_ptr<sdk::Sampler> sampler = std::make_unique<sdk::AlwaysOnSampler>();
  if (parent_based)
    sampler = std::make_unique<sdk::ParentBasedSampler>(std::move(sampler));
  return ProviderPtr(new sdk::TracerProvider(std::move(processor), opentelemetry::sdk::resource::Resource::Create({}),
                                             std::move(sampler)));
}
class StorageTracingTest : public testing::Test {
  protected:
  void SetUp() override { ASSERT_STATUS_OK(SetTracerProvider(Provider(data))); }
  void TearDown() override {
    ASSERT_STATUS_OK(SetTracerProvider(nullptr));
    ASSERT_STATUS_OK(SetTraceOptions({}));
  }
  std::shared_ptr<Memory> data;
};
arrow::Status Work(const char* name = "storage.read") {
  TraceScope scope(name);
  scope.Finish(arrow::Status::OK());
  return arrow::Status::OK();
}
class ControlledFile final : public arrow::io::BufferReader, public NonBlockingRandomAccessFile {
  public:
  ControlledFile() : arrow::io::BufferReader(arrow::Buffer::FromString("abcd")) {}
  arrow::Future<int64_t> pending = arrow::Future<int64_t>::Make();
  arrow::Future<int64_t> ReadAtAsyncInto(int64_t, int64_t, uint8_t*) override { return pending; }
  arrow::Future<int64_t> GetSizeAsync() override { return arrow::Future<int64_t>::MakeFinished(4); }
};
// Verify the tracing boundary itself. libc++ does not declare the implicit
// const-char* -> string_view conversion noexcept, even though it does not allocate.
static_assert(std::is_nothrow_constructible_v<TraceScope, opentelemetry::nostd::string_view>);
static_assert(std::is_nothrow_destructible_v<TraceScope>);
static_assert(noexcept(TraceScope::Deferred(std::declval<opentelemetry::nostd::string_view>())));
static_assert(noexcept(AttachContext(ContextPtr{})));
static_assert(noexcept(AttachParent(std::declval<const TraceParent&>())));
static_assert(std::is_nothrow_constructible_v<TraceParent, const ot::SpanContext&>);
static_assert(noexcept(Capture()));
static_assert(noexcept(SetTracerProvider(ProviderPtr{})));
static_assert(noexcept(SetTraceOptions(std::declval<const TraceOptions&>())));
static_assert(noexcept(Bind([]() noexcept {})()));
static_assert(noexcept(HasContext()));
static_assert(noexcept(StartCurrent()));
static_assert(noexcept(EnsureStarted(std::declval<const SpanPtr&>(), std::declval<const ContextPtr&>())));
static_assert(noexcept(EndSpan(std::declval<const SpanPtr&>(), std::declval<const ContextPtr&>())));
// Passing an existing error status must not copy/allocate before API entry.
static_assert(noexcept(
    EndSpan(std::declval<const SpanPtr&>(), std::declval<const ContextPtr&>(), std::declval<const arrow::Status&>())));
static_assert(
    noexcept(EndSpan(std::declval<const SpanPtr&>(), std::declval<const ContextPtr&>(), ot::StatusCode::kError)));
static_assert(noexcept(std::declval<TraceScope&>().Finish(std::declval<const arrow::Status&>())));
static_assert(noexcept(SetAttribute(std::declval<const SpanPtr&>(),
                                    std::declval<const ContextPtr&>(),
                                    std::declval<opentelemetry::nostd::string_view>(),
                                    std::declval<const opentelemetry::common::AttributeValue&>())));
static_assert(noexcept(GetSpanContext(std::declval<const ContextPtr&>())));

#ifdef BUILD_WITH_FIU
TEST_F(StorageTracingTest, FailedScopeDoesNotEscapeOrReplaceBusinessResult) {
  auto parent = AttachParent(Parent(1));
  auto before = Capture();
  const auto business_error = arrow::Status::Invalid("original business error");
  {
    ScopedFiuFault fault(FIUKEY_TRACING_SCOPE_FAIL);
    ASSERT_EQ(fault.enable_result(), 0);
    auto business_call = [&]() -> arrow::Status {
      TraceScope scope("failed");
      EXPECT_FALSE(scope.span());
      EXPECT_FALSE(Capture());
      EXPECT_TRUE(Work("not_misattributed").ok());
      return business_error;
    };
    EXPECT_EQ(business_call(), business_error);
  }
  EXPECT_EQ(Capture(), before);
  EXPECT_TRUE(data->GetSpans().empty());
  EXPECT_TRUE(Work("after_failure").ok());
  ASSERT_EQ(data->GetSpans().size(), 1);
}

TEST_F(StorageTracingTest, FailedAttachmentMasksForeignParentAndRestoresOtherRequestData) {
  struct OtherData : folly::RequestData {
    bool hasCallback() override { return false; }
  };
  auto outer = AttachParent(Parent(1));
  const auto before = Capture();
  const folly::RequestToken key("tracing-failure.other-data");
  auto other = std::make_unique<OtherData>();
  const auto* expected = other.get();
  folly::RequestContext::get()->setContextData(key, std::move(other));
  {
    ScopedFiuFault fault(FIUKEY_TRACING_CONTEXT_ATTACH_FAIL);
    ASSERT_EQ(fault.enable_result(), 0);
    auto failed = AttachParent(Parent(2));
    EXPECT_FALSE(Capture());
    EXPECT_FALSE(HasContext());
    EXPECT_EQ(folly::RequestContext::get()->getContextData(key), expected);
    EXPECT_TRUE(Work("not_under_foreign_parent").ok());
  }
  EXPECT_EQ(Capture(), before);
  EXPECT_EQ(folly::RequestContext::get()->getContextData(key), expected);
  EXPECT_TRUE(data->GetSpans().empty());
  EXPECT_TRUE(Work("restored").ok());
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetTraceId(), ExpectedTrace(1));
}

TEST_F(StorageTracingTest, EmptyAttachmentStillMasksWhenAnotherFailedScopeFinishes) {
  auto outer = AttachParent(Parent(1));
  const auto before = Capture();
  ScopedFiuFault fault(FIUKEY_TRACING_SCOPE_FAIL);
  ASSERT_EQ(fault.enable_result(), 0);
  TraceScope failed("failed");
  EXPECT_FALSE(Capture());
  {
    auto empty = AttachContext(nullptr);
    failed.Finish();
    EXPECT_FALSE(Capture());
    EXPECT_FALSE(HasContext());
    EXPECT_TRUE(Work("masked_after_recovery").ok());
  }
  EXPECT_EQ(Capture(), before);
  EXPECT_TRUE(data->GetSpans().empty());
  EXPECT_TRUE(Work("restored_after_failure").ok());
  EXPECT_EQ(data->GetSpans().size(), 1);
}

TEST_F(StorageTracingTest, ConfigurationFailureReturnsStatusAndPreservesPreviousSettings) {
  {
    ScopedFiuFault fault(FIUKEY_TRACING_CONFIGURATION_FAIL);
    ASSERT_EQ(fault.enable_result(), 0);
    EXPECT_FALSE(SetTracerProvider(nullptr).ok());
  }
  auto parent = AttachParent(Parent(1));
  EXPECT_TRUE(Work("old_provider").ok());
  EXPECT_EQ(data->GetSpans().size(), 1);
  ASSERT_STATUS_OK(SetTraceOptions({true, 1}));
  {
    ScopedFiuFault fault(FIUKEY_TRACING_CONFIGURATION_FAIL);
    ASSERT_EQ(fault.enable_result(), 0);
    EXPECT_FALSE(SetTraceOptions({true, 256}).ok());
  }
  {
    TraceScope scope("root");
    EXPECT_TRUE(Work("still_suppressed").ok());
  }
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetName(), "root");
}
#endif

TEST_F(StorageTracingTest, NativeAttributesReachSamplerAndPreserveTypes) {
  class AttributeSampler final : public sdk::Sampler {
 public:
    sdk::SamplingResult ShouldSample(const ot::SpanContext& parent,
                                     ot::TraceId,
                                     opentelemetry::nostd::string_view,
                                     ot::SpanKind,
                                     const opentelemetry::common::KeyValueIterable& attributes,
                                     const ot::SpanContextKeyValueIterable&) noexcept override {
      bool sample = false;
      attributes.ForEachKeyValue(
          [&](opentelemetry::nostd::string_view key, opentelemetry::common::AttributeValue value) noexcept {
            if (key == "test.sample")
              sample = opentelemetry::nostd::get<bool>(value);
            return true;
          });
      return {sample ? sdk::Decision::RECORD_AND_SAMPLE : sdk::Decision::DROP, nullptr, parent.trace_state()};
    }
    opentelemetry::nostd::string_view GetDescription() const noexcept override { return "attribute sampler"; }
  };
  auto exporter = std::make_unique<opentelemetry::exporter::memory::InMemorySpanExporter>(128);
  data = exporter->GetData();
  auto processor = std::make_unique<sdk::SimpleSpanProcessor>(std::move(exporter));
  ASSERT_STATUS_OK(SetTracerProvider(
      ProviderPtr(new sdk::TracerProvider(std::move(processor), opentelemetry::sdk::resource::Resource::Create({}),
                                          std::make_unique<AttributeSampler>()))));
  ASSERT_STATUS_OK(SetTraceOptions({false, 256}));
  auto parent = AttachParent(Parent(1));
  const int column_index = 3;
  const std::string column_name = "column-name";
  const int64_t records_to_read = 4096;
  {
    TraceScope scope("parquet::arrow::read_column",
                     {{"test.sample", true},
                      {"parquet.arrow.columnindex", column_index},
                      {"parquet.arrow.columnname", column_name},
                      {"parquet.arrow.physicaltype", "INT64"},
                      {"parquet.arrow.records_to_read", records_to_read},
                      {"test.ratio", 0.5}},
                     {}, ot::SpanKind::kClient);
  }
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetSpanKind(), ot::SpanKind::kClient);
  const auto& attributes = spans[0]->GetAttributes();
  EXPECT_EQ(opentelemetry::nostd::get<int32_t>(attributes.at("parquet.arrow.columnindex")), 3);
  EXPECT_EQ(opentelemetry::nostd::get<std::string>(attributes.at("parquet.arrow.columnname")), column_name);
  EXPECT_EQ(opentelemetry::nostd::get<int64_t>(attributes.at("parquet.arrow.records_to_read")), records_to_read);
  EXPECT_EQ(opentelemetry::nostd::get<double>(attributes.at("test.ratio")), 0.5);
}

TEST_F(StorageTracingTest, DeferredScopeOwnsNativeAttributesAndLinksUntilExecution) {
  auto parent = AttachParent(Parent(1));
  SpanPtr span;
  ContextPtr context;
  const ot::SpanContext link(ExpectedTrace(2), ExpectedSpan(2), ot::TraceFlags(1), false);
  {
    std::string name = "deferred-owned-span-name";
    std::string column = "deferred-owned-column-name";
    std::string text = "deferred-owned-array-element";
    std::string link_label = "deferred-owned-link-label";
    int64_t indices[] = {1, 7};
    bool flags[] = {true, false};
    opentelemetry::nostd::string_view texts[] = {text};
    auto scope =
        TraceScope::Deferred(name,
                             {{"column", column},
                              {"indices", opentelemetry::nostd::span<const int64_t>(indices)},
                              {"flags", opentelemetry::nostd::span<const bool>(flags)},
                              {"texts", opentelemetry::nostd::span<const opentelemetry::nostd::string_view>(texts)}},
                             {{link, {{"label", link_label}}}}, ot::SpanKind::kClient);
    span = scope.span();
    context = scope.context();
    scope.ReleaseSpan();
    name.assign("changed");
    column.assign("changed");
    text.assign("changed");
    link_label.assign("changed");
    indices[0] = 99;
    flags[0] = false;
  }
  EXPECT_TRUE(data->GetSpans().empty());
  {
    auto scope = AttachContext(context);
    StartCurrent();
    EndSpan(span, context, arrow::Status::OK());
  }
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetName(), "deferred-owned-span-name");
  EXPECT_EQ(spans[0]->GetSpanKind(), ot::SpanKind::kClient);
  const auto& attributes = spans[0]->GetAttributes();
  EXPECT_EQ(opentelemetry::nostd::get<std::string>(attributes.at("column")), "deferred-owned-column-name");
  EXPECT_EQ(opentelemetry::nostd::get<std::vector<int64_t>>(attributes.at("indices")), (std::vector<int64_t>{1, 7}));
  EXPECT_EQ(opentelemetry::nostd::get<std::vector<bool>>(attributes.at("flags")), (std::vector<bool>{true, false}));
  EXPECT_EQ(opentelemetry::nostd::get<std::vector<std::string>>(attributes.at("texts")),
            (std::vector<std::string>{"deferred-owned-array-element"}));
  ASSERT_EQ(spans[0]->GetLinks().size(), 1);
  EXPECT_EQ(spans[0]->GetLinks()[0].GetSpanContext().span_id(), ExpectedSpan(2));
  EXPECT_EQ(opentelemetry::nostd::get<std::string>(spans[0]->GetLinks()[0].GetAttributes().at("label")),
            "deferred-owned-link-label");
}

TEST_F(StorageTracingTest, ReleasedScopeLeavesSpanAliveForNativeCompletion) {
  auto parent = AttachParent(Parent(1));
  const auto before = Capture();
  SpanPtr span;
  ContextPtr context;
  {
    TraceScope scope("native");
    span = scope.span();
    context = scope.context();
    EXPECT_EQ(Capture(), context);
    scope.ReleaseSpan();
  }
  EXPECT_EQ(Capture(), before);
  EXPECT_TRUE(data->GetSpans().empty());
  EndSpan(span, context, arrow::Status::OK());
  EXPECT_EQ(data->GetSpans().size(), 1);
}

TEST_F(StorageTracingTest, ParentScopeNestedMaskAndRestore) {
  static_assert(!std::is_move_constructible_v<TraceScope>);
  static_assert(!std::is_copy_constructible_v<TraceScope>);
  EXPECT_TRUE(Work().ok());
  {
    auto outer = AttachParent(Parent(1));
    EXPECT_TRUE(Work("outer").ok());
    {
      auto inner = AttachParent(Parent(2));
      EXPECT_TRUE(Work("inner").ok());
    }
    {
      auto empty = AttachParent({});
      EXPECT_TRUE(Work("masked").ok());
    }
    EXPECT_TRUE(Work("restored").ok());
  }
  EXPECT_TRUE(Work().ok());
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 3);
  EXPECT_EQ(spans[0]->GetTraceId(), ExpectedTrace(1));
  EXPECT_EQ(spans[0]->GetParentSpanId(), ExpectedSpan(1));
  EXPECT_EQ(spans[1]->GetTraceId(), ExpectedTrace(2));
  EXPECT_EQ(spans[2]->GetTraceId(), spans[0]->GetTraceId());
  EXPECT_EQ(spans[0]->GetInstrumentationScope().GetName(), "milvus-storage");
  EXPECT_EQ(spans[0]->GetSpanContext().trace_state()->ToHeader(), "vendor=value");
}
TEST_F(StorageTracingTest, NullProviderDoesNotUseGlobalProvider) {
  ASSERT_STATUS_OK(SetTracerProvider(nullptr));
  auto scope = AttachParent(Parent(1));
  EXPECT_TRUE(Work().ok());
  EXPECT_TRUE(data->GetSpans().empty());
}
TEST_F(StorageTracingTest, ArrowSpanCanParentStorageOperations) {
  auto provider = Provider(data);
  ASSERT_STATUS_OK(SetTracerProvider(provider));
  arrow::util::tracing::Span parent_span;
  auto& span = arrow::internal::tracing::RewrapSpan(parent_span.details.get(),
                                                    provider->GetTracer("arrow-caller")->StartSpan("caller"));
  const auto context = span->GetContext();
  const ot::SpanContext upstream(context.trace_id(), context.span_id(), ot::TraceFlags(0), true,
                                 ot::TraceState::FromHeader("vendor=upstream"));
  const TraceParent parent{upstream};
  {
    auto scope = AttachParent(parent);
    const auto attached = GetSpanContext(Capture());
    EXPECT_FALSE(attached.IsSampled());
    EXPECT_TRUE(attached.IsRemote());
    EXPECT_EQ(attached.trace_state()->ToHeader(), "vendor=upstream");
    EXPECT_TRUE(Work().ok());
  }
  MARK_SPAN(parent_span, arrow::Status::OK());
  END_SPAN(parent_span);
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 2);
  EXPECT_EQ(spans[0]->GetTraceId(), spans[1]->GetTraceId());
  EXPECT_EQ(spans[0]->GetParentSpanId(), spans[1]->GetSpanId());
  EXPECT_EQ(spans[1]->GetName(), "caller");
}
TEST_F(StorageTracingTest, DisabledParquetReadDoesNotUseArrowGlobalProvider) {
  using namespace milvus_storage::api;
  struct RestoreGlobalProvider {
    ProviderPtr previous = ot::Provider::GetTracerProvider();
    ~RestoreGlobalProvider() { ot::Provider::SetTracerProvider(previous); }
  } restore;
  std::shared_ptr<Memory> global_spans;
  ot::Provider::SetTracerProvider(Provider(global_spans));
  ASSERT_STATUS_OK(SetTracerProvider(nullptr));

  Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  ASSERT_AND_ASSIGN(auto fs, GetFileSystem(properties));
  const auto path = GetTestBasePath("storage-tracing-arrow-global");
  ASSERT_STATUS_OK(CreateTestDir(fs, path));
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto batch, CreateTestData(schema));
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy("parquet", schema));
  auto writer = Writer::create(path, schema, std::move(policy), properties);
  ASSERT_STATUS_OK(writer->write(batch));
  ASSERT_AND_ASSIGN(auto groups, writer->close());
  auto reader = Reader::create(groups, schema, nullptr, properties);
  ASSERT_TRUE(reader->take({0, 3, 7}).ok());
  ASSERT_STATUS_OK(DeleteTestDir(fs, path));
  EXPECT_TRUE(global_spans->GetSpans().empty());
}
TEST_F(StorageTracingTest, NestedMetadataNamesDoNotSuppressDistinctOperations) {
  auto scope = AttachParent(Parent(1));
  TraceScope span_scope("storage.metadata.load");
  auto context = span_scope.context();
  auto span = span_scope.span();
  {
    auto nested_scope = AttachContext(context);
    EXPECT_TRUE(Work("storage.metadata.load").ok());
  }
  EndSpan(span, context, arrow::Status::OK());
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 2);
  EXPECT_EQ(spans[0]->GetParentSpanId(), spans[1]->GetSpanId());
  EXPECT_EQ(spans[1]->GetParentSpanId(), ExpectedSpan(1));
}
TEST_F(StorageTracingTest, NullProviderKeepsReadyFutureReady) {
  ASSERT_STATUS_OK(SetTracerProvider(nullptr));
  auto scope = AttachParent(Parent(1));
  auto file = WrapFile(std::make_shared<ControlledFile>(), "test");
  auto* async = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  auto future = async->GetSizeAsync();
  EXPECT_TRUE(future.is_finished());
  ASSERT_TRUE(future.result().ok());
  EXPECT_EQ(*future.result(), 4);
  EXPECT_TRUE(data->GetSpans().empty());
}

TEST_F(StorageTracingTest, UnsampledParentIsNotPromotedToRoot) {
  ASSERT_STATUS_OK(SetTracerProvider(Provider(data, true)));
  auto parent = Parent(1);
  parent.trace_flags = 0;
  auto scope = AttachParent(parent);
  TraceScope operation_scope("storage.read");
  auto operation_context = operation_scope.context();
  auto operation = operation_scope.span();
  EXPECT_EQ(tracing::GetSpanContext(operation_context).trace_id(), ExpectedTrace(1));
  EXPECT_FALSE(tracing::GetSpanContext(operation_context).IsSampled());
  tracing::EndSpan(operation, operation_context, arrow::Status::OK());
  EXPECT_TRUE(data->GetSpans().empty());
}
TEST_F(StorageTracingTest, UnsampledFlagStillRespectsHostAlwaysOnSampler) {
  auto parent = Parent(1);
  parent.trace_flags = 0;
  auto scope = AttachParent(parent);
  EXPECT_TRUE(Work().ok());
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_TRUE(spans[0]->GetSpanContext().IsSampled());
  EXPECT_EQ(spans[0]->GetParentSpanId(), ExpectedSpan(1));
}
TEST_F(StorageTracingTest, ContextPresenceMasksAndRestoresWithoutChangingOtherKeys) {
  struct OtherData : folly::RequestData {
    bool hasCallback() override { return false; }
  };
  folly::RequestContextScopeGuard request_scope;
  const folly::RequestToken key("storage-test.other-key");
  auto* original = new OtherData;
  folly::RequestContext::get()->setContextData(key, std::unique_ptr<OtherData>(original));
  EXPECT_FALSE(HasContext());
  {
    auto parent = AttachParent(Parent(1));
    EXPECT_TRUE(HasContext());
    {
      auto same = AttachContext(Capture());
      EXPECT_TRUE(HasContext());
      EXPECT_EQ(folly::RequestContext::get()->getContextData(key), original);
      folly::RequestContext::get()->clearContextData(key);
      folly::RequestContext::get()->setContextData(key, std::make_unique<OtherData>());
    }
    EXPECT_EQ(folly::RequestContext::get()->getContextData(key), original);
    {
      auto masked = AttachContext(nullptr);
      EXPECT_FALSE(HasContext());
    }
    EXPECT_TRUE(HasContext());
  }
  EXPECT_FALSE(HasContext());
  EXPECT_EQ(folly::RequestContext::get()->getContextData(key), original);
}
TEST_F(StorageTracingTest, OpaqueRustAttachmentOwnsSnapshotAndRestoresForeignParent) {
  namespace ffi = milvus_storage::rust_bridge::ffi;
  auto context = [&] {
    auto parent = AttachParent(Parent(1));
    return ffi::capture_trace_context();
  }();
  std::thread worker([&] {
    auto parent = AttachParent(Parent(2));
    {
      auto attachment = ffi::attach_trace_context(context);
      EXPECT_TRUE(Work("captured").ok());
    }
    {
      auto attachment = ffi::attach_trace_context(nullptr);
      EXPECT_FALSE(HasContext());
      EXPECT_TRUE(Work("masked").ok());
    }
    EXPECT_TRUE(Work("restored").ok());
  });
  worker.join();
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 2);
  EXPECT_EQ(spans[0]->GetTraceId(), ExpectedTrace(1));
  EXPECT_EQ(spans[1]->GetTraceId(), ExpectedTrace(2));
}
TEST_F(StorageTracingTest, LazyFutureCapturesParentAndProviderBeforeConsumption) {
  auto future = [&] {
    auto parent = AttachParent(Parent(1));
    auto span_scope = TraceScope::Deferred("storage.read");
    auto context = span_scope.context();
    auto span = span_scope.span();
    auto scope = AttachContext(context);
    auto future = folly::makeSemiFuture().deferValue([span, context](folly::Unit) {
      auto scope = AttachContext(context);
      StartCurrent();
      SCOPE_EXIT { EndSpan(span, context); };
      return Work("child");
    });
    span_scope.ReleaseSpan();
    return future;
  }();
  EXPECT_TRUE(data->GetSpans().empty());
  std::shared_ptr<Memory> replacement;
  ASSERT_STATUS_OK(SetTracerProvider(Provider(replacement)));
  folly::CPUThreadPoolExecutor executor(1);
  {
    auto scope = AttachParent(Parent(2));
    EXPECT_TRUE(std::move(future).via(&executor).get().ok());
  }
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 2);
  for (auto& span : spans) EXPECT_EQ(span->GetTraceId(), ExpectedTrace(1));
  EXPECT_EQ(spans[0]->GetParentSpanId(), spans[1]->GetSpanId());
  EXPECT_TRUE(replacement->GetSpans().empty());
}

TEST_F(StorageTracingTest, UnconsumedFutureDoesNotCreateWorkSpans) {
  auto parent = AttachParent(Parent(1));
  {
    auto span_scope = TraceScope::Deferred("storage.read");
    auto context = span_scope.context();
    auto span = span_scope.span();
    auto scope = AttachContext(context);
    auto future = folly::makeSemiFuture().deferValue([span, context](folly::Unit) {
      auto scope = AttachContext(context);
      StartCurrent();
      SCOPE_EXIT { EndSpan(span, context); };
      return Work("storage.fs.read");
    });
    span_scope.ReleaseSpan();
  }
  EXPECT_TRUE(data->GetSpans().empty());
}

TEST_F(StorageTracingTest, NoParentKeepsReadyFutureReady) {
  auto file = WrapFile(std::make_shared<ControlledFile>(), "test");
  auto* async = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  auto future = async->GetSizeAsync();
  EXPECT_TRUE(future.is_finished());
  ASSERT_TRUE(future.result().ok());
  EXPECT_EQ(*future.result(), 4);
  EXPECT_TRUE(data->GetSpans().empty());
}

TEST_F(StorageTracingTest, EmptyCapturedContextMasksLaterParent) {
  auto task = Bind([] { return Work("must_remain_untraced"); });
  auto parent = AttachParent(Parent(1));
  EXPECT_TRUE(task().ok());
  EXPECT_TRUE(Work("restored_parent").ok());
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetName(), "restored_parent");
  EXPECT_EQ(spans[0]->GetTraceId(), ExpectedTrace(1));
}
TEST_F(StorageTracingTest, NativeCompletionDoesNotScheduleConsumerExecutor) {
  folly::ManualExecutor executor;
  folly::Promise<arrow::Status> promise;
  SpanPtr completion;
  ContextPtr completion_context;
  auto future =
      [&] {
        auto parent = AttachParent(Parent(1));
        TraceScope scope("storage.format.read");
        completion_context = scope.context();
        completion = scope.span();
        auto future = promise.getSemiFuture();
        scope.ReleaseSpan();
        return future;
      }()
          .via(&executor);
  std::thread worker([&] {
    EndSpan(completion, completion_context, arrow::Status::OK());
    promise.setValue(arrow::Status::OK());
  });
  worker.join();
  EXPECT_TRUE(future.isReady());
  EXPECT_EQ(executor.drain(), 0);
  ASSERT_EQ(data->GetSpans().size(), 1);
}

TEST_F(StorageTracingTest, SameThreadFibersRestoreIndependentNestedContexts) {
  folly::fibers::FiberManager manager(std::make_unique<folly::fibers::SimpleLoopController>());
  folly::fibers::Baton a_ready, b_ready;
  manager.addTask([&] {
    auto scope = AttachParent(Parent(1));
    EXPECT_TRUE(Work("a.before").ok());
    a_ready.post();
    b_ready.wait();
    {
      auto nested = AttachParent(Parent(3));
      EXPECT_TRUE(Work("a.nested").ok());
    }
    EXPECT_TRUE(Work("a.after").ok());
  });
  manager.addTask([&] {
    a_ready.wait();
    auto scope = AttachParent(Parent(2));
    EXPECT_TRUE(Work("b").ok());
    b_ready.post();
  });
  manager.loopUntilNoReady();
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 4);
  for (auto& span : spans) {
    auto expected = span->GetName() == "b" ? 2 : span->GetName() == "a.nested" ? 3 : 1;
    EXPECT_EQ(span->GetTraceId(), ExpectedTrace(expected));
  }
  EXPECT_TRUE(Work("outside").ok());
  EXPECT_TRUE(data->GetSpans().empty());
}
TEST_F(StorageTracingTest, BudgetSuppressesChildrenWithoutEndingParent) {
  ASSERT_STATUS_OK(SetTraceOptions({true, 2}));
  auto parent = AttachParent(Parent(1));
  TraceScope span_scope("storage.read");
  auto context = span_scope.context();
  auto span = span_scope.span();
  {
    auto scope = AttachContext(context);
    for (int i = 0; i < 10; ++i) EXPECT_TRUE(Work("child").ok());
  }
  EndSpan(span, context, arrow::Status::Invalid("sensitive path must not be exported"));
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 2);
  EXPECT_EQ(spans[1]->GetName(), "storage.read");
  EXPECT_EQ(spans[1]->GetStatus(), ot::StatusCode::kError);
  EXPECT_EQ(opentelemetry::nostd::get<int64_t>(spans[1]->GetAttributes().at("storage.spans.dropped")), 9);
  EXPECT_TRUE(spans[1]->GetDescription().empty());
}

TEST_F(StorageTracingTest, FinishRacesExportExactlyOnce) {
  auto scope = AttachParent(Parent(1));
  TraceScope operation_scope("storage.read");
  auto operation_context = operation_scope.context();
  auto operation = operation_scope.span();
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i)
    threads.emplace_back([operation, operation_context] {
      tracing::EndSpan(operation, operation_context, arrow::Status::IOError("private"));
    });
  for (auto& thread : threads) thread.join();
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetStatus(), ot::StatusCode::kError);
}
TEST_F(StorageTracingTest, ConcurrentLazyStartPublishesOneCompleteSpan) {
  auto scope = AttachParent(Parent(1));
  auto operation_scope = TraceScope::Deferred("storage.read");
  auto operation_context = operation_scope.context();
  auto operation = operation_scope.span();
  std::promise<void> start;
  auto ready = start.get_future().share();
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([operation, ready, operation_context] {
      ready.wait();
      for (int j = 0; j < 50; ++j) {
        tracing::EnsureStarted(operation, operation_context);
        EXPECT_EQ(tracing::GetSpanContext(operation_context).trace_id(), ExpectedTrace(1));
        tracing::SetAttribute(operation, operation_context, "published", int64_t{1});
      }
    });
  }
  start.set_value();
  for (auto& thread : threads) thread.join();
  tracing::EndSpan(operation, operation_context, arrow::Status::OK());
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetParentSpanId(), ExpectedSpan(1));
  EXPECT_EQ(opentelemetry::nostd::get<int64_t>(spans[0]->GetAttributes().at("published")), 1);
}

TEST_F(StorageTracingTest, DisabledIOFuturesDoNotRetainTracingContexts) {
  class PendingFile final : public arrow::io::BufferReader {
 public:
    PendingFile() : arrow::io::BufferReader(arrow::Buffer::FromString("abcd")) {}
    arrow::Future<std::shared_ptr<arrow::Buffer>> pending = arrow::Future<std::shared_ptr<arrow::Buffer>>::Make();
    arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext&, int64_t, int64_t) override {
      return pending;
    }
    std::vector<arrow::Future<std::shared_ptr<arrow::Buffer>>> ReadManyAsync(
        const arrow::io::IOContext&, const std::vector<arrow::io::ReadRange>& ranges) override {
      return std::vector<arrow::Future<std::shared_ptr<arrow::Buffer>>>(ranges.size(), pending);
    }
  };
  ASSERT_STATUS_OK(SetTracerProvider(nullptr));
  auto raw = std::make_shared<PendingFile>();
  auto file = WrapFile(raw, "test");
  std::weak_ptr<const Context> captured;
  std::vector<arrow::Future<std::shared_ptr<arrow::Buffer>>> futures;
  {
    auto parent = AttachParent(Parent(1));
    TraceScope operation_scope("storage.read");
    auto operation_context = operation_scope.context();
    auto operation = operation_scope.span();
    auto scope = AttachContext(operation_context);
    captured = operation_context;
    futures = file->ReadManyAsync(file->io_context(), {{0, 1}, {1, 1}});
    futures.push_back(file->ReadAsync(file->io_context(), 0, 2));
  }
  // Only tracing callbacks could retain this context: the source future has no
  // context of its own. Disabled reads return it without completion observers.
  EXPECT_TRUE(captured.expired());
  ASSERT_STATUS_OK(SetTracerProvider(Provider(data)));
  auto foreign = AttachParent(Parent(2));
  raw->pending.MarkFinished(arrow::Buffer::FromString("ab"));
  for (const auto& future : futures) {
    ASSERT_TRUE(future.result().ok());
    EXPECT_EQ((*future.result())->size(), 2);
  }
  EXPECT_TRUE(data->GetSpans().empty());
}
TEST_F(StorageTracingTest, SuppressedIOSpansStillAggregateReads) {
  auto file = WrapFile(std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString("abcd")), "test");
  auto parent = AttachParent(Parent(1));
  for (const auto options : {TraceOptions{false, 256}, TraceOptions{true, 1}}) {
    ASSERT_STATUS_OK(SetTraceOptions(options));
    {
      TraceScope span_scope("storage.read");
      auto context = span_scope.context();
      auto span = span_scope.span();
      auto scope = AttachContext(context);
      SCOPE_EXIT { EndSpan(span, context); };
      ASSERT_TRUE(file->ReadAt(0, 2).ok());
      ASSERT_TRUE(file->ReadAsync(file->io_context(), 0, 2).status().ok());
      for (const auto& future : file->ReadManyAsync(file->io_context(), {{0, 1}, {1, 1}})) {
        ASSERT_TRUE(future.status().ok());
      }
    }
    auto spans = data->GetSpans();
    ASSERT_EQ(spans.size(), 1);
    const auto& attrs = spans[0]->GetAttributes();
    EXPECT_EQ(opentelemetry::nostd::get<int64_t>(attrs.at("storage.io.reads")), 3);
    EXPECT_EQ(opentelemetry::nostd::get<int64_t>(attrs.at("storage.io.requested_bytes")), 6);
    EXPECT_EQ(opentelemetry::nostd::get<int64_t>(attrs.at("storage.io.returned_bytes")), 6);
  }
}

TEST_F(StorageTracingTest, NativeIOFinishesAfterDroppedFutureAndPreservesCapability) {
  auto raw = std::make_shared<ControlledFile>();
  auto file = WrapFile(raw, "test");
  auto* async = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async, nullptr);
  uint8_t out[4];
  {
    auto scope = AttachParent(Parent(1));
    auto unused = async->ReadAtAsyncInto(0, 4, out);
  }
  EXPECT_TRUE(data->GetSpans().empty());
  std::thread callback([&] {
    auto scope = AttachParent(Parent(2));
    raw->pending.MarkFinished(2);
  });
  callback.join();
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetTraceId(), ExpectedTrace(1));
  EXPECT_EQ(opentelemetry::nostd::get<int64_t>(spans[0]->GetAttributes().at("storage.returned_bytes")), 2);
  auto sync = WrapFile(std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString("abcd")), "test");
  EXPECT_EQ(dynamic_cast<NonBlockingRandomAccessFile*>(sync.get()), nullptr);
}
TEST_F(StorageTracingTest, IOOverloadsErrorAndDisabledSwitch) {
  auto file = WrapFile(std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString("abcd")), "test");
  auto scope = AttachParent(Parent(1));
  char out[4];
  EXPECT_TRUE(file->ReadAt(0, 2, out).ok());
  EXPECT_TRUE(file->ReadAt(0, 2).ok());
  EXPECT_TRUE(file->ReadAsync(file->io_context(), 0, 2).result().ok());
  auto futures = file->ReadManyAsync(file->io_context(), {{0, 1}, {1, 1}});
  for (auto& future : futures) EXPECT_TRUE(future.result().ok());
  EXPECT_FALSE(file->ReadAt(-1, 2).ok());
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 5);
  EXPECT_EQ(spans.back()->GetStatus(), ot::StatusCode::kError);
  ASSERT_STATUS_OK(SetTraceOptions({false, 256}));
  {
    TraceScope span_scope("storage.read");
    auto context = span_scope.context();
    auto span = span_scope.span();
    auto scope = AttachContext(context);
    SCOPE_EXIT { EndSpan(span, context); };
    EXPECT_TRUE(file->ReadAt(0, 2).ok());
  }
  spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetName(), "storage.read");
}

TEST_F(StorageTracingTest, ArrowExecutorUsesOperationContextOnForeignSubmissionThread) {
  folly::CPUThreadPoolExecutor pool(1);
  auto executor = [&] {
    auto scope = AttachParent(Parent(1));
    return parquet::MakeFollyArrowExecutor(folly::getKeepAliveToken(&pool)).ValueOrDie();
  }();
  std::promise<void> done;
  std::thread callback([&] {
    auto scope = AttachParent(Parent(2));
    EXPECT_TRUE(executor
                    ->Spawn([&] {
                      EXPECT_TRUE(Work().ok());
                      done.set_value();
                    })
                    .ok());
  });
  done.get_future().get();
  callback.join();
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetTraceId(), ExpectedTrace(1));
}
TEST_F(StorageTracingTest, MetadataFollowerLinksLeaderAndCacheDoesNotRetainParent) {
  using Cache = FormatReaderMetadataCache<parquet::ParquetFormatReader>;
  auto cache = Cache::Make();
  folly::CPUThreadPoolExecutor executor(2);
  folly::Promise<Cache::MetadataResult> promise;
  std::promise<void> entered;
  auto leader =
      [&] {
        auto scope = AttachParent(Parent(1));
        return cache->get_or_open_async("key", [&] {
          entered.set_value();
          return promise.getSemiFuture();
        });
      }()
          .via(&executor);
  entered.get_future().get();
  // Drive the follower synchronously until it has subscribed to the active flight.
  folly::ManualExecutor follower_executor;
  auto follower =
      [&] {
        auto scope = AttachParent(Parent(2));
        return cache->get_or_open_async("key", []() -> folly::SemiFuture<Cache::MetadataResult> {
          ADD_FAILURE() << "follower must not load";
          return folly::makeSemiFuture(Cache::MetadataResult(arrow::Status::Invalid("unexpected loader")));
        });
      }()
          .via(&follower_executor);
  follower_executor.run();
  auto metadata = std::make_shared<Cache::Trait::Metadata>();
  promise.setValue(Cache::MetadataResult(metadata));
  EXPECT_TRUE(std::move(leader).get().ok());
  follower_executor.drain();
  EXPECT_TRUE(std::move(follower).get().ok());
  {
    auto scope = AttachParent(Parent(3));
    EXPECT_TRUE(
        cache->get_or_open("key", []() -> Cache::MetadataResult { return arrow::Status::Invalid("must be a hit"); })
            .ok());
  }
  auto spans = data->GetSpans();
  const sdk::SpanData* load = nullptr;
  const sdk::SpanData* wait = nullptr;
  int loads = 0;
  for (auto& span : spans) {
    if (span->GetName() == "storage.metadata.load") {
      load = span.get();
      ++loads;
    }
    if (span->GetName() == "storage.metadata.wait")
      wait = span.get();
  }
  ASSERT_EQ(loads, 1);
  ASSERT_NE(wait, nullptr);
  EXPECT_EQ(load->GetTraceId(), ExpectedTrace(1));
  EXPECT_EQ(wait->GetTraceId(), ExpectedTrace(2));
  ASSERT_EQ(wait->GetLinks().size(), 1);
  EXPECT_EQ(wait->GetLinks()[0].GetSpanContext().span_id(), load->GetSpanId());
}
TEST_F(StorageTracingTest, MetadataFlightSupportsMixedTracedAndUntracedCallers) {
  using Cache = FormatReaderMetadataCache<parquet::ParquetFormatReader>;
  for (bool trace_leader : {false, true}) {
    SCOPED_TRACE(trace_leader);
    auto cache = Cache::Make();
    folly::ManualExecutor executor;
    folly::Promise<Cache::MetadataResult> promise;
    auto load = [&] { return cache->get_or_open_async("key", [&] { return promise.getSemiFuture(); }); };
    auto leader =
        [&] {
          if (trace_leader) {
            auto parent = AttachParent(Parent(1));
            return load();
          }
          return load();
        }()
            .via(&executor);
    executor.run();
    auto follow = [&] {
      return cache->get_or_open_async("key", []() -> folly::SemiFuture<Cache::MetadataResult> {
        ADD_FAILURE() << "follower must share the load";
        return folly::makeSemiFuture(Cache::MetadataResult(arrow::Status::Invalid("unexpected loader")));
      });
    };
    auto follower =
        [&] {
          if (!trace_leader) {
            auto parent = AttachParent(Parent(2));
            return follow();
          }
          return follow();
        }()
            .via(&executor);
    executor.run();
    auto metadata = std::make_shared<Cache::Trait::Metadata>();
    promise.setValue(Cache::MetadataResult(metadata));
    executor.drain();
    EXPECT_EQ(std::move(leader).get().ValueOrDie(), metadata);
    EXPECT_EQ(std::move(follower).get().ValueOrDie(), metadata);
    auto spans = data->GetSpans();
    int loads = 0, waits = 0;
    for (const auto& span : spans) {
      loads += span->GetName() == "storage.metadata.load";
      if (span->GetName() == "storage.metadata.wait") {
        ++waits;
        EXPECT_TRUE(span->GetLinks().empty());
        EXPECT_EQ(span->GetTraceId(), ExpectedTrace(2));
      }
    }
    EXPECT_EQ(loads, trace_leader ? 1 : 0);
    EXPECT_EQ(waits, trace_leader ? 0 : 1);
  }
}
TEST_F(StorageTracingTest, SharedReaderConcurrentParquetAndVortexReadsKeepTheirOwnTrace) {
  using namespace milvus_storage::api;
  for (const std::string format : {"parquet", "vortex"}) {
    Properties properties;
    ASSERT_STATUS_OK(InitTestProperties(properties));
    // Force both operations to perform physical reads. With shared Vortex
    // metadata, a small file may already be satisfied by the leader's reads.
    ASSERT_EQ(SetValue(properties, PROPERTY_READER_METADATA_CACHE_ENABLE, "false"), std::nullopt);
    ASSERT_AND_ASSIGN(auto fs, GetFileSystem(properties));
    auto path = GetTestBasePath("storage-tracing-" + format);
    ASSERT_STATUS_OK(CreateTestDir(fs, path));
    ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
    ASSERT_AND_ASSIGN(auto batch, CreateTestData(schema));
    ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, schema));
    auto writer = Writer::create(path, schema, std::move(policy), properties);
    ASSERT_STATUS_OK(writer->write(batch));
    ASSERT_AND_ASSIGN(auto groups, writer->close());
    auto reader = Reader::create(groups, schema, nullptr, properties);
    folly::CPUThreadPoolExecutor executor(2);
    auto make_read = [&](uint8_t id) {
      auto scope = AttachParent(Parent(id));
      return reader->take_async({0, 3, 7}, 2);
    };
    auto first = make_read(1).via(&executor);
    auto second = make_read(2).via(&executor);
    ASSERT_TRUE(std::move(first).get().ok());
    ASSERT_TRUE(std::move(second).get().ok());
    auto spans = data->GetSpans();
    int roots[2] = {0, 0};
    int reads[2] = {0, 0};
    for (auto& span : spans) {
      const int index = span->GetTraceId() == ExpectedTrace(1) ? 0 : 1;
      EXPECT_EQ(span->GetTraceId(), ExpectedTrace(index + 1));
      if (span->GetParentSpanId() == ExpectedSpan(index + 1))
        ++roots[index];
      if (span->GetName() == "storage.fs.read")
        ++reads[index];
      EXPECT_NE(span->GetStatus(), ot::StatusCode::kError);
    }
    EXPECT_EQ(roots[0], 1) << format;
    EXPECT_EQ(roots[1], 1) << format;
    EXPECT_GT(reads[0], 0) << format;
    EXPECT_GT(reads[1], 0) << format;
    ASSERT_STATUS_OK(DeleteTestDir(fs, path));
  }
}
TEST_F(StorageTracingTest, CachedReaderMetadataIOBelongsToSingleLeaderSpan) {
  using namespace milvus_storage::api;
  folly::CPUThreadPoolExecutor executor(2);
  for (const std::string format : {"parquet", "vortex"}) {
    Properties properties;
    ASSERT_STATUS_OK(InitTestProperties(properties));
    ASSERT_EQ(SetValue(properties, PROPERTY_READER_METADATA_CACHE_ENABLE, "true"), std::nullopt);
    ASSERT_AND_ASSIGN(auto fs, GetFileSystem(properties));
    auto path = GetTestBasePath("storage-tracing-cache-parent-" + format);
    ASSERT_STATUS_OK(CreateTestDir(fs, path));
    ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
    ASSERT_AND_ASSIGN(auto batch, CreateTestData(schema));
    ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, schema));
    auto writer = Writer::create(path, schema, std::move(policy), properties);
    ASSERT_STATUS_OK(writer->write(batch));
    ASSERT_AND_ASSIGN(auto groups, writer->close());
    for (bool async : {false, true}) {
      for (bool take : {false, true}) {
        SCOPED_TRACE(testing::Message() << format << " async=" << async << " take=" << take);
        // Each Reader owns a fresh cache. Exercise both column-group reader
        // implementations and both loader entrypoints on an actual cache miss.
        auto reader = Reader::create(groups, schema, nullptr, properties);
        {
          auto scope = AttachParent(Parent(1));
          if (take) {
            auto result = async ? reader->take_async({0, 3, 7}).via(&executor).get() : reader->take({0, 3, 7});
            ASSERT_TRUE(result.ok()) << result.status();
          } else {
            auto result = async ? reader->get_chunk_reader_async(0).via(&executor).get() : reader->get_chunk_reader(0);
            ASSERT_TRUE(result.ok()) << result.status();
          }
        }
        auto spans = data->GetSpans();
        const sdk::SpanData* load = nullptr;
        size_t loads = 0;
        for (const auto& span : spans) {
          EXPECT_EQ(span->GetTraceId(), ExpectedTrace(1));
          if (span->GetName() == "storage.metadata.load") {
            load = span.get();
            ++loads;
          }
        }
        ASSERT_EQ(loads, 1);
        bool has_metadata_io = false;
        for (const auto& span : spans) {
          if (span->GetName() == "storage.fs.read" && span->GetParentSpanId() == load->GetSpanId())
            has_metadata_io = true;
        }
        EXPECT_TRUE(has_metadata_io);
      }
    }
    ASSERT_STATUS_OK(DeleteTestDir(fs, path));
  }
}
TEST_F(StorageTracingTest, UnconsumedReaderFuturesDoNotCreateWorkSpans) {
  using namespace milvus_storage::api;
  Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  ASSERT_AND_ASSIGN(auto fs, GetFileSystem(properties));
  auto path = GetTestBasePath("storage-tracing-lazy-reader");
  ASSERT_STATUS_OK(CreateTestDir(fs, path));
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto batch, CreateTestData(schema));
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy("parquet", schema));
  auto writer = Writer::create(path, schema, std::move(policy), properties);
  ASSERT_STATUS_OK(writer->write(batch));
  ASSERT_AND_ASSIGN(auto groups, writer->close());
  auto reader = Reader::create(groups, schema, nullptr, properties);
  // Initialize outside tracing so chunk planning can run without any I/O.
  ASSERT_AND_ASSIGN(auto chunk_reader, reader->get_chunk_reader(0));
  {
    auto scope = AttachParent(Parent(1));
    auto take = reader->take_async({0, 3, 7});
    auto chunks = chunk_reader->get_chunks_async({0});
    auto open = reader->get_chunk_reader_async(0);
  }
  EXPECT_TRUE(data->GetSpans().empty());
  // The same deferred read still starts under the entry parent when consumed
  // later, even if the consumer has activated another request.
  auto future = [&] {
    auto scope = AttachParent(Parent(1));
    return reader->take_async({0, 3, 7});
  }();
  folly::CPUThreadPoolExecutor executor(1);
  {
    auto scope = AttachParent(Parent(2));
    auto result = std::move(future).via(&executor).get();
    ASSERT_TRUE(result.ok()) << result.status();
  }
  auto spans = data->GetSpans();
  ASSERT_FALSE(spans.empty());
  for (const auto& span : spans) EXPECT_EQ(span->GetTraceId(), ExpectedTrace(1));
  ASSERT_STATUS_OK(DeleteTestDir(fs, path));
}
TEST_F(StorageTracingTest, DisabledProviderSnapshotRemainsDisabledAfterInjection) {
  ASSERT_STATUS_OK(SetTracerProvider(nullptr));
  auto future = [&] {
    auto parent = AttachParent(Parent(1));
    auto span_scope = TraceScope::Deferred("storage.read");
    auto context = span_scope.context();
    auto span = span_scope.span();
    auto scope = AttachContext(context);
    auto future = folly::makeSemiFuture().deferValue([span, context](folly::Unit) {
      auto scope = AttachContext(context);
      StartCurrent();
      SCOPE_EXIT { EndSpan(span, context); };
      return Work("child");
    });
    span_scope.ReleaseSpan();
    return future;
  }();
  ASSERT_STATUS_OK(SetTracerProvider(Provider(data)));
  EXPECT_TRUE(std::move(future).get().ok());
  EXPECT_TRUE(data->GetSpans().empty());
}

TEST_F(StorageTracingTest, PreservesStorageErrorClassificationWithoutExportingMessages) {
  auto scope = AttachParent(Parent(1));
  auto error = MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "secret object path");
  TraceScope span_scope("storage.read");
  auto context = span_scope.context();
  auto span = span_scope.span();
  EndSpan(span, context, error);
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  const auto& attrs = spans[0]->GetAttributes();
  EXPECT_EQ(opentelemetry::nostd::get<std::string>(attrs.at("error.type")),
            ExtendStatusDetail::UnwrapStatus(error)->CodeAsString());
  EXPECT_TRUE(opentelemetry::nostd::get<bool>(attrs.at("error.retryable")));
  EXPECT_TRUE(spans[0]->GetDescription().empty());
}

TEST_F(StorageTracingTest, InlineIOFailureIsObserved) {
  auto raw = std::make_shared<ControlledFile>();
  raw->pending = arrow::Future<int64_t>::MakeFinished(arrow::Status::IOError("private"));
  auto file = WrapFile(raw, "test");
  auto scope = AttachParent(Parent(1));
  uint8_t out[4];
  auto* async = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  EXPECT_FALSE(async->ReadAtAsyncInto(0, 4, out).result().ok());
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetStatus(), ot::StatusCode::kError);
}
TEST_F(StorageTracingTest, ScopeCleanupDoesNotInterceptExceptionsOrInventStatus) {
  auto parent = AttachParent(Parent(1));
  auto throwing = []() -> arrow::Status {
    TraceScope scope("storage.read");
    throw std::runtime_error("private path");
  };
  EXPECT_THROW((void)throwing(), std::runtime_error);
  auto spans = data->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  EXPECT_EQ(spans[0]->GetStatus(), ot::StatusCode::kUnset);
  EXPECT_TRUE(spans[0]->GetDescription().empty());
  EXPECT_EQ(spans[0]->GetAttributes().count("storage.completion.unobserved"), 0);
}

TEST_F(StorageTracingTest, ScopeCleanupDoesNotStartUnconsumedLazySpan) {
  auto parent = AttachParent(Parent(1));
  {
    auto span_scope = TraceScope::Deferred("storage.read");
    auto context = span_scope.context();
    auto span = span_scope.span();
    EndSpan(span, context);
  }
  EXPECT_TRUE(data->GetSpans().empty());
}

class ThrowingFile final : public arrow::io::RandomAccessFile, public NonBlockingRandomAccessFile {
  public:
  arrow::Status Close() override { return arrow::Status::OK(); }
  bool closed() const override { return false; }
  arrow::Result<int64_t> Tell() const override { return 0; }
  arrow::Status Seek(int64_t) override { return arrow::Status::OK(); }
  arrow::Result<int64_t> GetSize() override { return 4; }
  arrow::Result<int64_t> Read(int64_t, void*) override { return arrow::Status::NotImplemented("test"); }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t) override { return arrow::Status::NotImplemented("test"); }
  arrow::Result<int64_t> ReadAt(int64_t, int64_t, void*) override { throw std::runtime_error("private"); }
  arrow::Future<int64_t> ReadAtAsyncInto(int64_t, int64_t, uint8_t*) override { throw 42; }
  arrow::Future<int64_t> GetSizeAsync() override { throw std::runtime_error("private"); }
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext&) override {
    throw std::runtime_error("private");
  }
  std::vector<arrow::Future<std::shared_ptr<arrow::Buffer>>> ReadManyAsync(
      const arrow::io::IOContext&, const std::vector<arrow::io::ReadRange>&) override {
    throw 42;
  }
};

TEST_F(StorageTracingTest, FileInstrumentationPreservesThrownExceptions) {
  for (bool enabled : {false, true}) {
    ASSERT_STATUS_OK(SetTracerProvider(enabled ? Provider(data) : nullptr));
    auto file = WrapFile(std::make_shared<ThrowingFile>(), "test");
    auto scope = AttachParent(Parent(1));
    uint8_t out[4];
    EXPECT_THROW((void)file->ReadAt(0, 4, out), std::runtime_error);
    auto* async = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
    ASSERT_NE(async, nullptr);
    EXPECT_THROW((void)async->ReadAtAsyncInto(0, 4, out), int);
    EXPECT_THROW((void)async->GetSizeAsync(), std::runtime_error);
    EXPECT_THROW((void)file->ReadMetadataAsync(file->io_context()), std::runtime_error);
    EXPECT_THROW((void)file->ReadManyAsync(file->io_context(), {{0, 1}, {1, 2}}), int);
  }
}

}  // namespace
}  // namespace milvus_storage::tracing
