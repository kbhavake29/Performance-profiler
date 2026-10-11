// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0
//
// Private implementation types shared by profiler.cpp, span.cpp and metrics.cpp.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "inference_profiler/config.hpp"
#include "inference_profiler/metrics.hpp"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/async_instruments.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/sync_instruments.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/unique_ptr.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#include "opentelemetry/sdk/trace/tracer_provider.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/tracer.h"

namespace inference_profiler {
namespace detail {

namespace otel_metrics = opentelemetry::metrics;
namespace otel_trace = opentelemetry::trace;
namespace nostd = opentelemetry::nostd;

using Clock = std::chrono::steady_clock;
using Attrs = std::map<std::string, std::string>;

/// Counters updated from exporter wrappers. Kept outside the instruments so the
/// metric exporter never records into the provider it is exporting for.
struct ExportStats {
  std::atomic<uint64_t> span_export_failures{0};
  std::atomic<uint64_t> metric_export_failures{0};
  std::atomic<uint64_t> dropped_spans{0};
  std::atomic<uint64_t> instrumentation_errors{0};
  /// Set once instruments exist; read by the span exporter wrapper.
  std::atomic<otel_metrics::Histogram<double>*> export_duration{nullptr};
};

struct Instruments {
  nostd::unique_ptr<otel_metrics::Histogram<double>> request_duration;
  nostd::unique_ptr<otel_metrics::Histogram<double>> queue_duration;
  nostd::unique_ptr<otel_metrics::Histogram<double>> prefill_duration;
  nostd::unique_ptr<otel_metrics::Histogram<double>> time_to_first_token;
  nostd::unique_ptr<otel_metrics::Histogram<double>> inter_token_latency;
  nostd::unique_ptr<otel_metrics::Histogram<double>> generation_duration;
  nostd::unique_ptr<otel_metrics::Histogram<double>> phase_duration;
  nostd::unique_ptr<otel_metrics::Histogram<double>> generation_throughput;
  nostd::unique_ptr<otel_metrics::Histogram<double>> cache_operation_duration;
  nostd::unique_ptr<otel_metrics::Histogram<double>> export_duration;
  nostd::unique_ptr<otel_metrics::Counter<uint64_t>> requests;
  nostd::unique_ptr<otel_metrics::Counter<uint64_t>> tokens;
  nostd::unique_ptr<otel_metrics::Counter<uint64_t>> cache_hits;
  nostd::unique_ptr<otel_metrics::Counter<uint64_t>> cache_misses;
  nostd::unique_ptr<otel_metrics::Counter<uint64_t>> cache_insertions;
  nostd::unique_ptr<otel_metrics::Counter<uint64_t>> cache_evictions;
  nostd::unique_ptr<otel_metrics::UpDownCounter<int64_t>> requests_in_flight;
  nostd::unique_ptr<otel_metrics::UpDownCounter<int64_t>> requests_active;
  nostd::unique_ptr<otel_metrics::UpDownCounter<int64_t>> queue_depth;
  nostd::shared_ptr<otel_metrics::ObservableInstrument> export_errors;
  nostd::shared_ptr<otel_metrics::ObservableInstrument> telemetry_dropped;
  nostd::shared_ptr<otel_metrics::ObservableInstrument> instrumentation_errors;
  nostd::shared_ptr<otel_metrics::ObservableInstrument> enabled;
  nostd::shared_ptr<otel_metrics::ObservableInstrument> process_memory;

  static Instruments Create(otel_metrics::Meter& meter, ExportStats* stats,
                            bool report_process_memory);
};

/// Registers explicit-bucket histogram views matching the Python SDK.
void RegisterHistogramViews(opentelemetry::sdk::metrics::MeterProvider& provider,
                            const std::string& scope_name, const std::string& scope_version);

struct ProfilerImpl {
  ProfilerConfig config;
  std::shared_ptr<opentelemetry::sdk::trace::TracerProvider> tracer_provider;
  std::shared_ptr<opentelemetry::sdk::metrics::MeterProvider> meter_provider;
  nostd::shared_ptr<otel_trace::Tracer> tracer;
  nostd::shared_ptr<otel_metrics::Meter> meter;
  Instruments instruments;
  std::shared_ptr<ExportStats> stats = std::make_shared<ExportStats>();
  std::unique_ptr<CardinalityLimiter> limiter;
  std::atomic<bool> is_shutdown{false};
  std::mutex shutdown_mutex;

  /// Non-strict: log once, count, continue. Strict: rethrow.
  void HandleError(const char* where, const std::exception& error);
  Attrs MetricAttrs(const std::string& model, const std::string& runtime_version);
};

struct RequestState {
  std::shared_ptr<ProfilerImpl> profiler;
  nostd::shared_ptr<otel_trace::Span> span;
  Attrs metric_attrs;
  Clock::time_point start;
  std::chrono::system_clock::time_point start_system;
  std::optional<Clock::time_point> first_token;
  std::optional<Clock::time_point> last_token;
  int64_t token_count = 0;
  bool ttft_recorded = false;
  std::mutex mutex;  // guards the fields below; phases may end on other threads
  double queue_seconds = 0.0;
  std::optional<double> generation_seconds;
  bool queued = false;
  std::string status = attr::kStatusOk;
  std::string error_type;
  std::atomic<bool> ended{false};
  int uncaught_at_start = 0;

  void EnterQueue();
  void LeaveQueueLocked();
  void PhaseFinished(const std::string& name, double seconds);
  void RecordTtft(double seconds);
};

struct PhaseState {
  std::shared_ptr<RequestState> request;
  nostd::shared_ptr<otel_trace::Span> span;
  std::string name;
  std::string cache_operation;  // "", "lookup", "insert", "evict"
  int64_t entries = 1;
  std::optional<bool> hit;
  Clock::time_point start;
  int uncaught_at_start = 0;
  bool failed = false;
};

/// Creates a phase span parented to `parent` and updates queue gauges.
std::unique_ptr<PhaseState> StartPhase(const std::shared_ptr<RequestState>& request,
                                       const nostd::shared_ptr<otel_trace::Span>& parent,
                                       const std::string& name, const std::string& span_name);

nostd::string_view Sv(std::string_view s);

inline opentelemetry::context::Context EmptyContext() { return opentelemetry::context::Context{}; }

}  // namespace detail
}  // namespace inference_profiler
