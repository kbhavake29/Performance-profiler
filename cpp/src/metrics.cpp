// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0

#include "inference_profiler/metrics.hpp"

#include <vector>

#include "internal.hpp"
#include "opentelemetry/metrics/observer_result.h"
#include "opentelemetry/nostd/variant.h"
#include "opentelemetry/sdk/metrics/aggregation/aggregation_config.h"
#include "opentelemetry/sdk/metrics/instruments.h"
#include "opentelemetry/sdk/metrics/view/instrument_selector_factory.h"
#include "opentelemetry/sdk/metrics/view/meter_selector_factory.h"
#include "opentelemetry/sdk/metrics/view/view_factory.h"

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>

#include <fstream>
#endif

namespace inference_profiler {

std::string CardinalityLimiter::Limit(const std::string& key, const std::string& value) {
  {
    std::shared_lock<std::shared_mutex> read(mutex_);
    auto it = seen_.find(key);
    if (it != seen_.end() && it->second.count(value) != 0) return value;
  }
  std::unique_lock<std::shared_mutex> write(mutex_);
  auto& values = seen_[key];
  if (values.count(value) != 0) return value;
  if (values.size() >= max_values_) {
    overflowed_.insert(key);
    return attr::kOtherValue;
  }
  values.insert(value);
  return value;
}

bool CardinalityLimiter::Overflowed(const std::string& key) const {
  std::shared_lock<std::shared_mutex> read(mutex_);
  return overflowed_.count(key) != 0;
}

long long ProcessRssBytes() noexcept {
#if defined(__APPLE__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                &count) == KERN_SUCCESS) {
    return static_cast<long long>(info.resident_size);
  }
  return 0;
#elif defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  long long pages_total = 0;
  long long pages_resident = 0;
  if (statm >> pages_total >> pages_resident) {
    return pages_resident * static_cast<long long>(sysconf(_SC_PAGESIZE));
  }
  return 0;
#else
  return 0;
#endif
}

namespace detail {
namespace {

namespace sdk_metrics = opentelemetry::sdk::metrics;

const std::vector<double> kLatencyBuckets = {0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025,
                                             0.05,   0.075, 0.1,    0.25,  0.5,    0.75, 1.0,
                                             2.5,    5.0,   7.5,    10.0,  30.0,   60.0};
const std::vector<double> kInterTokenBuckets = {0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02,
                                                0.03,  0.05,   0.075, 0.1,    0.2,  0.5,   1.0};
const std::vector<double> kThroughputBuckets = {1,   5,   10,  20,  30,  50,  75,
                                                100, 150, 200, 300, 500, 1000};

void ObserveCounter(otel_metrics::ObserverResult result, int64_t value, const Attrs& attrs) {
  if (nostd::holds_alternative<nostd::shared_ptr<otel_metrics::ObserverResultT<int64_t>>>(
          result)) {
    nostd::get<nostd::shared_ptr<otel_metrics::ObserverResultT<int64_t>>>(result)->Observe(
        value, attrs);
  }
}

void ObserveExportErrors(otel_metrics::ObserverResult result, void* state) {
  auto* stats = static_cast<ExportStats*>(state);
  ObserveCounter(result, static_cast<int64_t>(stats->span_export_failures.load()),
                 {{attr::kSignal, "traces"}});
  ObserveCounter(result, static_cast<int64_t>(stats->metric_export_failures.load()),
                 {{attr::kSignal, "metrics"}});
}

void ObserveDropped(otel_metrics::ObserverResult result, void* state) {
  auto* stats = static_cast<ExportStats*>(state);
  ObserveCounter(result, static_cast<int64_t>(stats->dropped_spans.load()),
                 {{attr::kSignal, "traces"}, {"inference_profiler.reason", "export_failed"}});
}

void ObserveInstrumentationErrors(otel_metrics::ObserverResult result, void* state) {
  auto* stats = static_cast<ExportStats*>(state);
  ObserveCounter(result, static_cast<int64_t>(stats->instrumentation_errors.load()), {});
}

void ObserveEnabled(otel_metrics::ObserverResult result, void* /*state*/) {
  ObserveCounter(result, 1, {});
}

void ObserveRss(otel_metrics::ObserverResult result, void* /*state*/) {
  const long long rss = ProcessRssBytes();
  if (rss > 0) ObserveCounter(result, rss, {});
}

}  // namespace

Instruments Instruments::Create(otel_metrics::Meter& meter, ExportStats* stats,
                                bool report_process_memory) {
  Instruments i;
  i.request_duration = meter.CreateDoubleHistogram(
      metric_names::kRequestDuration, "End-to-end inference request latency, including queueing.",
      "s");
  i.queue_duration = meter.CreateDoubleHistogram(
      metric_names::kQueueDuration, "Time a request waited before execution started.", "s");
  i.prefill_duration = meter.CreateDoubleHistogram(metric_names::kPrefillDuration,
                                                   "Prompt processing (prefill) duration.", "s");
  i.time_to_first_token = meter.CreateDoubleHistogram(
      metric_names::kTimeToFirstToken, "Time from request arrival to the first generated token.",
      "s");
  i.inter_token_latency = meter.CreateDoubleHistogram(
      metric_names::kInterTokenLatency, "Delay between consecutive generated tokens.", "s");
  i.generation_duration = meter.CreateDoubleHistogram(
      metric_names::kGenerationDuration, "Token generation (decode) duration.", "s");
  i.phase_duration = meter.CreateDoubleHistogram(metric_names::kPhaseDuration,
                                                 "Duration of custom request phases.", "s");
  i.generation_throughput = meter.CreateDoubleHistogram(
      metric_names::kGenerationThroughput, "Per-request decode throughput.", "{token}/s");
  i.cache_operation_duration = meter.CreateDoubleHistogram(
      metric_names::kCacheOperationDuration, "Cache lookup/insert/evict duration.", "s");
  i.export_duration = meter.CreateDoubleHistogram(metric_names::kExportDuration,
                                                  "Duration of telemetry export calls.", "s");
  i.requests = meter.CreateUInt64Counter(metric_names::kRequests, "Finished requests by status.",
                                         "{request}");
  i.tokens = meter.CreateUInt64Counter(metric_names::kTokens,
                                       "Prompt and generated tokens processed.", "{token}");
  i.cache_hits = meter.CreateUInt64Counter(metric_names::kCacheHits,
                                           "Cache lookups that found an entry.", "{lookup}");
  i.cache_misses =
      meter.CreateUInt64Counter(metric_names::kCacheMisses, "Cache lookups that missed.", "{lookup}");
  i.cache_insertions = meter.CreateUInt64Counter(metric_names::kCacheInsertions,
                                                 "Entries inserted into the cache.", "{entry}");
  i.cache_evictions = meter.CreateUInt64Counter(metric_names::kCacheEvictions,
                                                "Entries evicted from the cache.", "{entry}");
  i.requests_in_flight = meter.CreateInt64UpDownCounter(
      metric_names::kRequestsInFlight,
      "Admitted requests that have not finished (queued + executing).", "{request}");
  i.requests_active = meter.CreateInt64UpDownCounter(
      metric_names::kRequestsActive, "Requests currently executing (excludes queued).",
      "{request}");
  i.queue_depth = meter.CreateInt64UpDownCounter(
      metric_names::kQueueDepth, "Requests currently waiting in the queue phase.", "{request}");

  i.export_errors = meter.CreateInt64ObservableCounter(
      metric_names::kExportErrors, "Failed telemetry export attempts.", "{error}");
  i.export_errors->AddCallback(ObserveExportErrors, stats);
  i.telemetry_dropped = meter.CreateInt64ObservableCounter(
      metric_names::kTelemetryDropped, "Spans dropped because export failed.", "{item}");
  i.telemetry_dropped->AddCallback(ObserveDropped, stats);
  i.instrumentation_errors = meter.CreateInt64ObservableCounter(
      metric_names::kInstrumentationErrors, "Internal profiler errors swallowed in non-strict mode.",
      "{error}");
  i.instrumentation_errors->AddCallback(ObserveInstrumentationErrors, stats);
  i.enabled = meter.CreateInt64ObservableGauge(
      metric_names::kProfilerEnabled,
      "1 when the inference profiler SDK is enabled in this process.", "");
  i.enabled->AddCallback(ObserveEnabled, nullptr);
  if (report_process_memory) {
    i.process_memory = meter.CreateInt64ObservableGauge(
        metric_names::kProcessMemoryUsage,
        "Resident set size of the instrumented process (real measurement).", "By");
    i.process_memory->AddCallback(ObserveRss, nullptr);
  }
  stats->export_duration.store(i.export_duration.get());
  return i;
}

void RegisterHistogramViews(sdk_metrics::MeterProvider& provider, const std::string& scope_name,
                            const std::string& scope_version) {
  struct Spec {
    const char* name;
    const char* unit;
    const std::vector<double>* buckets;
  };
  const Spec specs[] = {
      {metric_names::kRequestDuration, "s", &kLatencyBuckets},
      {metric_names::kQueueDuration, "s", &kLatencyBuckets},
      {metric_names::kPrefillDuration, "s", &kLatencyBuckets},
      {metric_names::kTimeToFirstToken, "s", &kLatencyBuckets},
      {metric_names::kInterTokenLatency, "s", &kInterTokenBuckets},
      {metric_names::kGenerationDuration, "s", &kLatencyBuckets},
      {metric_names::kPhaseDuration, "s", &kLatencyBuckets},
      {metric_names::kGenerationThroughput, "{token}/s", &kThroughputBuckets},
      {metric_names::kCacheOperationDuration, "s", &kLatencyBuckets},
      {metric_names::kExportDuration, "s", &kLatencyBuckets},
  };
  for (const auto& spec : specs) {
    auto config = std::make_shared<sdk_metrics::HistogramAggregationConfig>();
    config->boundaries_ = *spec.buckets;
    provider.AddView(
        sdk_metrics::InstrumentSelectorFactory::Create(sdk_metrics::InstrumentType::kHistogram,
                                                       spec.name, spec.unit),
        sdk_metrics::MeterSelectorFactory::Create(scope_name, scope_version, ""),
        sdk_metrics::ViewFactory::Create(spec.name, "", sdk_metrics::AggregationType::kHistogram,
                                         config));
  }
}

}  // namespace detail
}  // namespace inference_profiler
