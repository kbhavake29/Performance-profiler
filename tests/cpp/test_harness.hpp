// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0
//
// Tiny dependency-free test harness: each TEST registers a named case that the
// test binary runs when given its name (CTest runs one case per process).
#pragma once

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "inference_profiler/profiler.hpp"
#include "inference_profiler/testing.hpp"
#include "opentelemetry/exporters/memory/in_memory_metric_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/sdk/metrics/instruments.h"
#include "opentelemetry/sdk/trace/span_data.h"

namespace test {

struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline std::map<std::string, std::function<void()>>& Registry() {
  static std::map<std::string, std::function<void()>> registry;
  return registry;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { Registry()[name] = std::move(fn); }
};

#define TEST_CASE(name)                                     \
  static void test_##name();                                \
  static ::test::Registrar registrar_##name(#name, test_##name); \
  static void test_##name()

#define CHECK(cond)                                                                       \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::ostringstream os_;                                                             \
      os_ << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ") failed";                  \
      throw ::test::Failure(os_.str());                                                   \
    }                                                                                     \
  } while (0)

#define CHECK_EQ(a, b)                                                                    \
  do {                                                                                    \
    const auto& va_ = (a);                                                                \
    const auto& vb_ = (b);                                                                \
    if (!(va_ == vb_)) {                                                                  \
      std::ostringstream os_;                                                             \
      os_ << __FILE__ << ":" << __LINE__ << ": CHECK_EQ(" #a ", " #b ") failed: " << va_  \
          << " != " << vb_;                                                               \
      throw ::test::Failure(os_.str());                                                   \
    }                                                                                     \
  } while (0)

#define CHECK_THROWS(expr, Type)                                                         \
  do {                                                                                    \
    bool thrown_ = false;                                                                 \
    try {                                                                                 \
      (void)(expr);                                                                       \
    } catch (const Type&) {                                                               \
      thrown_ = true;                                                                     \
    }                                                                                     \
    if (!thrown_) throw ::test::Failure(std::string(__FILE__) + ": expected " #Type);     \
  } while (0)

namespace memory = opentelemetry::exporter::memory;
namespace sdk_trace = opentelemetry::sdk::trace;
namespace sdk_metrics = opentelemetry::sdk::metrics;

/// Cumulative temporality so UpDownCounter/Counter points hold running totals.
inline sdk_metrics::AggregationTemporality Cumulative(sdk_metrics::InstrumentType) {
  return sdk_metrics::AggregationTemporality::kCumulative;
}

/// A profiler wired to in-memory exporters.
struct Harness {
  std::shared_ptr<memory::InMemorySpanData> spans;
  std::shared_ptr<memory::SimpleAggregateInMemoryMetricData> metrics;
  std::shared_ptr<inference_profiler::Profiler> profiler;
  std::vector<std::unique_ptr<sdk_trace::SpanData>> finished;

  explicit Harness(inference_profiler::ProfilerConfig cfg = DefaultConfig());
  static inference_profiler::ProfilerConfig DefaultConfig();

  /// Moves newly exported spans into `finished` and returns it.
  std::vector<std::unique_ptr<sdk_trace::SpanData>>& Spans();
  const sdk_trace::SpanData* Find(const std::string& name);
  int Count(const std::string& name);
  /// Flushes metrics and returns the sum (counters) or count (histograms) for a metric,
  /// filtered by attributes.
  double MetricTotal(const std::string& name, const std::map<std::string, std::string>& attrs = {});
};

}  // namespace test
