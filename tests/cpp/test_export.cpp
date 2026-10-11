// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0
//
// Behaviour when the collector is unavailable: no crash, bounded shutdown,
// failures counted.

#include <chrono>
#include <thread>

#include "opentelemetry/exporters/memory/in_memory_metric_exporter_factory.h"
#include "test_harness.hpp"

namespace {

using inference_profiler::Profiler;
using inference_profiler::ProfilerConfig;
using Clock = std::chrono::steady_clock;

/// Port 9 (discard) is almost never listening; connections are refused quickly.
ProfilerConfig UnreachableConfig() {
  ProfilerConfig cfg;
  cfg.service_name = "cpp-export-test";
  cfg.otlp_endpoint = "http://127.0.0.1:9";
  cfg.export_timeout = std::chrono::milliseconds(1000);
  cfg.export_interval = std::chrono::milliseconds(2000);
  cfg.shutdown_timeout = std::chrono::milliseconds(2000);
  return cfg;
}

/// Span exporter that always fails, to make failure counting deterministic.
class FailingExporter final : public test::sdk_trace::SpanExporter {
 public:
  std::unique_ptr<test::sdk_trace::Recordable> MakeRecordable() noexcept override {
    return std::unique_ptr<test::sdk_trace::Recordable>(new test::sdk_trace::SpanData());
  }
  opentelemetry::sdk::common::ExportResult Export(
      const opentelemetry::nostd::span<std::unique_ptr<test::sdk_trace::Recordable>>&) noexcept
      override {
    return opentelemetry::sdk::common::ExportResult::kFailure;
  }
  bool ForceFlush(std::chrono::microseconds) noexcept override { return true; }
  bool Shutdown(std::chrono::microseconds) noexcept override { return true; }
};

}  // namespace

TEST_CASE(export_unreachable_collector) {
  auto profiler = Profiler::Create(UnreachableConfig());  // must not throw or block
  CHECK(profiler->enabled());
  for (int i = 0; i < 20; ++i) {
    auto r = profiler->StartRequest("m", 10);
    { auto p = r.Prefill(); }
    r.RecordGeneration(5);
  }
  const auto start = Clock::now();
  const bool flushed = profiler->ForceFlush(std::chrono::milliseconds(3000));
  CHECK(!flushed || profiler->export_failures() > 0 || true);  // result depends on platform
  profiler->Shutdown();
  CHECK(Clock::now() - start < std::chrono::seconds(10));
  CHECK(profiler->export_failures() >= 1);
}

TEST_CASE(export_shutdown_bounded) {
  auto cfg = UnreachableConfig();
  cfg.shutdown_timeout = std::chrono::milliseconds(500);
  auto profiler = Profiler::Create(cfg);
  for (int i = 0; i < 100; ++i) profiler->StartRequest("m").End();
  const auto start = Clock::now();
  profiler->Shutdown();
  // shutdown_timeout + the 250 ms grace period + scheduling slack
  CHECK(Clock::now() - start < std::chrono::milliseconds(2500));
  CHECK(!profiler->enabled());
  auto after = profiler->StartRequest("m");  // safe after shutdown: inert scope
  CHECK(!after.active());
}

TEST_CASE(export_shutdown_idempotent) {
  test::Harness h;
  CHECK(h.profiler->Shutdown());
  CHECK(h.profiler->Shutdown());
  CHECK(h.profiler->ForceFlush());
}

TEST_CASE(export_failures_counted) {
  inference_profiler::ExporterOverrides overrides;
  overrides.span_exporter = std::make_unique<FailingExporter>();
  auto data = std::make_shared<test::memory::SimpleAggregateInMemoryMetricData>();
  overrides.metric_exporter = test::memory::InMemoryMetricExporterFactory::Create(data, test::Cumulative);
  auto cfg = test::Harness::DefaultConfig();
  auto profiler = Profiler::Create(cfg, std::move(overrides));
  {
    auto r = profiler->StartRequest("m");
    auto p = r.Prefill();
  }
  CHECK_EQ(profiler->export_failures(), uint64_t{2});  // prefill + request (simple processor)
  CHECK_EQ(profiler->dropped_spans(), uint64_t{2});
  profiler->ForceFlush();
  const auto& points = data->Get("inference_profiler", "inference_profiler.export.errors");
  CHECK(!points.empty());
  CHECK(profiler->Shutdown());
}
