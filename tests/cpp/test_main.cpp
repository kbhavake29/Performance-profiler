// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0

#include <variant>

#include "opentelemetry/exporters/memory/in_memory_metric_exporter_factory.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter.h"
#include "opentelemetry/sdk/metrics/data/point_data.h"
#include "test_harness.hpp"

namespace test {

inference_profiler::ProfilerConfig Harness::DefaultConfig() {
  inference_profiler::ProfilerConfig cfg;
  cfg.service_name = "cpp-test";
  cfg.runtime_version = "test-1.0";
  cfg.strict = true;
  cfg.export_interval = std::chrono::hours(1);  // metrics are flushed explicitly
  cfg.export_timeout = std::chrono::seconds(5);
  return cfg;
}

Harness::Harness(inference_profiler::ProfilerConfig cfg) {
  auto span_exporter = std::make_unique<memory::InMemorySpanExporter>(1000);
  spans = span_exporter->GetData();
  metrics = std::make_shared<memory::SimpleAggregateInMemoryMetricData>();
  inference_profiler::ExporterOverrides overrides;
  overrides.span_exporter = std::move(span_exporter);
  overrides.metric_exporter = memory::InMemoryMetricExporterFactory::Create(metrics, Cumulative);
  profiler = inference_profiler::Profiler::Create(cfg, std::move(overrides));
}

std::vector<std::unique_ptr<sdk_trace::SpanData>>& Harness::Spans() {
  for (auto& s : spans->GetSpans()) finished.push_back(std::move(s));
  return finished;
}

const sdk_trace::SpanData* Harness::Find(const std::string& name) {
  for (auto& s : Spans()) {
    if (std::string(s->GetName()) == name) return s.get();
  }
  return nullptr;
}

int Harness::Count(const std::string& name) {
  int n = 0;
  for (auto& s : Spans()) n += std::string(s->GetName()) == name ? 1 : 0;
  return n;
}

namespace {
double PointValue(const sdk_metrics::PointType& point) {
  if (auto* sum = opentelemetry::nostd::get_if<sdk_metrics::SumPointData>(&point)) {
    if (auto* i = opentelemetry::nostd::get_if<int64_t>(&sum->value_)) return static_cast<double>(*i);
    return opentelemetry::nostd::get<double>(sum->value_);
  }
  if (auto* hist = opentelemetry::nostd::get_if<sdk_metrics::HistogramPointData>(&point)) {
    return static_cast<double>(hist->count_);
  }
  if (auto* gauge = opentelemetry::nostd::get_if<sdk_metrics::LastValuePointData>(&point)) {
    if (auto* i = opentelemetry::nostd::get_if<int64_t>(&gauge->value_)) return static_cast<double>(*i);
    return opentelemetry::nostd::get<double>(gauge->value_);
  }
  return 0.0;
}
}  // namespace

double Harness::MetricTotal(const std::string& name,
                            const std::map<std::string, std::string>& attrs) {
  // SimpleAggregateInMemoryMetricData keeps the first point it sees per series,
  // so clear it and re-export the cumulative state on every query.
  metrics->Clear();
  profiler->ForceFlush(std::chrono::seconds(5));
  double total = 0.0;
  const auto& points = metrics->Get("inference_profiler", name);
  for (const auto& [point_attrs, point] : points) {
    bool match = true;
    for (const auto& [k, v] : attrs) {
      auto it = point_attrs.find(k);
      if (it == point_attrs.end() || opentelemetry::nostd::get<std::string>(it->second) != v) {
        match = false;
        break;
      }
    }
    if (match) total += PointValue(point);
  }
  return total;
}

}  // namespace test

int main(int argc, char** argv) {
  auto& registry = test::Registry();
  std::vector<std::string> selected;
  for (int i = 1; i < argc; ++i) selected.emplace_back(argv[i]);
  if (selected.empty()) {
    for (auto& [name, _] : registry) selected.push_back(name);
  }
  int failures = 0;
  for (const auto& name : selected) {
    auto it = registry.find(name);
    if (it == registry.end()) {
      std::cerr << "unknown test case: " << name << "\n";
      return 2;
    }
    try {
      it->second();
      std::cout << "[PASS] " << name << "\n";
    } catch (const std::exception& e) {
      ++failures;
      std::cout << "[FAIL] " << name << ": " << e.what() << "\n";
    }
  }
  std::cout << (selected.size() - failures) << "/" << selected.size() << " passed\n";
  return failures == 0 ? 0 : 1;
}
