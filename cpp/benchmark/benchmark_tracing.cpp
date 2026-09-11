// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include <benchmark/benchmark.h>
#include <opentelemetry/exporters/memory/in_memory_span_exporter.h>
#include <opentelemetry/sdk/trace/simple_processor.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/sdk/trace/samplers/parent.h>
#include <opentelemetry/sdk/trace/samplers/always_on.h>
#include "milvus-storage/tracing.h"

namespace milvus_storage::tracing {
namespace {
// Modes: no parent/provider, parent with null provider, unsampled, sampled.
// Measures only Storage's runtime overhead, including one request attachment
// and four child spans. No I/O, exporter transport, or engine work is included.
void BM_StorageTracing(benchmark::State& state) {
  namespace sdk = opentelemetry::sdk::trace;
  const auto mode = state.range(0);
  {
    const auto configured = SetTracerProvider(nullptr);
    if (!configured.ok()) {
      state.SkipWithError(configured.message().c_str());
      return;
    }
  }
  if (mode >= 2) {
    auto exporter = std::make_unique<opentelemetry::exporter::memory::InMemorySpanExporter>(1);
    auto processor = std::make_unique<sdk::SimpleSpanProcessor>(std::move(exporter));
    auto sampler = std::make_unique<sdk::ParentBasedSampler>(std::make_unique<sdk::AlwaysOnSampler>());
    {
      const auto configured = SetTracerProvider(ProviderPtr(new sdk::TracerProvider(
          std::move(processor), opentelemetry::sdk::resource::Resource::Create({}), std::move(sampler))));
      if (!configured.ok()) {
        state.SkipWithError(configured.message().c_str());
        return;
      }
    }
  }
  TraceParent parent;
  parent.trace_id[0] = 1;
  parent.span_id[0] = 1;
  parent.trace_flags = mode == 3 ? 1 : 0;
  const auto iteration = [] {
    TraceScope scope("storage.read");
    for (int i = 0; i < 4; ++i) {
      TraceScope child("storage.read_task");
      benchmark::DoNotOptimize(arrow::Status::OK());
    }
  };
  for (auto _ : state) {
    if (mode == 0) {
      iteration();
    } else {
      auto parent_scope = AttachParent(parent);
      iteration();
    }
  }
  {
    const auto configured = SetTracerProvider(nullptr);
    if (!configured.ok()) {
      state.SkipWithError(configured.message().c_str());
      return;
    }
  }
}
BENCHMARK(BM_StorageTracing)->DenseRange(0, 3)->UseRealTime();
}  // namespace
}  // namespace milvus_storage::tracing
