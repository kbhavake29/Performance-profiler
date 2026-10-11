// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0

#include "inference_profiler/profiler.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <utility>

#include "inference_profiler/testing.hpp"
#include "internal.hpp"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_options.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_options.h"
#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_factory.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_options.h"
#include "opentelemetry/sdk/metrics/meter_provider_factory.h"
#include "opentelemetry/sdk/metrics/view/view_registry.h"
#include "opentelemetry/sdk/resource/resource.h"
#include "opentelemetry/sdk/trace/batch_span_processor_factory.h"
#include "opentelemetry/sdk/trace/batch_span_processor_options.h"
#include "opentelemetry/sdk/trace/samplers/parent.h"
#include "opentelemetry/sdk/trace/samplers/trace_id_ratio.h"
#include "opentelemetry/sdk/trace/simple_processor_factory.h"
#include "opentelemetry/trace/context.h"
#include "opentelemetry/trace/propagation/http_trace_context.h"
#include "opentelemetry/trace/provider.h"

namespace inference_profiler {

constexpr const char* kScopeName = "inference_profiler";
constexpr const char* kVersion = INFERENCE_PROFILER_VERSION;

// ============================================================== configuration

namespace {

std::string Trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

bool ParseBool(const std::string& var, const std::string& raw) {
  std::string v = raw;
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return std::tolower(c); });
  if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
  if (v == "0" || v == "false" || v == "no" || v == "off") return false;
  throw ConfigError("invalid value for " + var + ": '" + raw + "' (expected true/false)");
}

double ParseDouble(const std::string& var, const std::string& raw) {
  try {
    std::size_t used = 0;
    double value = std::stod(raw, &used);
    if (used != raw.size() || !std::isfinite(value)) throw std::invalid_argument("trailing");
    return value;
  } catch (const std::exception&) {
    throw ConfigError("invalid value for " + var + ": '" + raw + "' (expected a number)");
  }
}

std::chrono::milliseconds ParseMs(const std::string& var, const std::string& raw) {
  return std::chrono::milliseconds(static_cast<long long>(ParseDouble(var, raw)));
}

bool HasControlChars(const std::string& s) {
  return std::any_of(s.begin(), s.end(),
                     [](unsigned char c) { return c < 0x20 || c == 0x7f; });
}

void CheckName(std::vector<std::string>& errors, const char* label, const std::string& value) {
  if (Trim(value).empty()) {
    errors.push_back(std::string(label) + " must be a non-empty string");
  } else if (value.size() > 128) {
    errors.push_back(std::string(label) + " must be at most 128 characters");
  } else if (HasControlChars(value)) {
    errors.push_back(std::string(label) + " must not contain control characters");
  }
}

std::string BaseUrl(const std::string& endpoint) {
  std::string base = endpoint;
  while (!base.empty() && base.back() == '/') base.pop_back();
  return base;
}

}  // namespace

ProfilerConfig ProfilerConfig::FromEnv() {
  return FromEnv([](const char* name) { return std::getenv(name); });
}

ProfilerConfig ProfilerConfig::FromEnv(const EnvLookup& lookup) {
  ProfilerConfig cfg;
  auto get = [&](const char* name) -> std::optional<std::string> {
    const char* raw = lookup(name);
    if (raw == nullptr) return std::nullopt;
    std::string value = Trim(raw);
    if (value.empty()) return std::nullopt;
    return value;
  };
  // OTEL_RESOURCE_ATTRIBUTES first so that OTEL_SERVICE_NAME wins over service.name.
  if (auto v = get("OTEL_RESOURCE_ATTRIBUTES")) {
    std::stringstream ss(*v);
    std::string item;
    while (std::getline(ss, item, ',')) {
      auto eq = item.find('=');
      if (eq == std::string::npos) continue;
      std::string key = Trim(item.substr(0, eq));
      std::string value = Trim(item.substr(eq + 1));
      if (key.empty()) continue;
      if (key == "service.name") {
        cfg.service_name = value;
      } else {
        cfg.resource_attributes[key] = value;
      }
    }
  }
  if (auto v = get("OTEL_SERVICE_NAME")) cfg.service_name = *v;
  if (auto v = get("OTEL_EXPORTER_OTLP_ENDPOINT")) cfg.otlp_endpoint = *v;
  if (auto v = get("INFERENCE_PROFILER_ENABLED")) {
    cfg.enabled = ParseBool("INFERENCE_PROFILER_ENABLED", *v);
  }
  if (auto v = get("OTEL_SDK_DISABLED"); v && ParseBool("OTEL_SDK_DISABLED", *v)) {
    cfg.enabled = false;
  }
  if (auto v = get("INFERENCE_PROFILER_SAMPLE_RATE")) {
    cfg.sample_rate = ParseDouble("INFERENCE_PROFILER_SAMPLE_RATE", *v);
  }
  if (auto v = get("INFERENCE_PROFILER_EXPORT_INTERVAL")) {
    cfg.export_interval = ParseMs("INFERENCE_PROFILER_EXPORT_INTERVAL", *v);
  }
  if (auto v = get("INFERENCE_PROFILER_EXPORT_TIMEOUT")) {
    cfg.export_timeout = ParseMs("INFERENCE_PROFILER_EXPORT_TIMEOUT", *v);
  }
  if (auto v = get("INFERENCE_PROFILER_SHUTDOWN_TIMEOUT")) {
    cfg.shutdown_timeout = ParseMs("INFERENCE_PROFILER_SHUTDOWN_TIMEOUT", *v);
  }
  if (auto v = get("INFERENCE_PROFILER_MEMORY_INTERVAL")) {
    // The native SDK samples RSS at metric export time; 0 disables it.
    cfg.report_process_memory = ParseMs("INFERENCE_PROFILER_MEMORY_INTERVAL", *v).count() > 0;
  }
  if (auto v = get("INFERENCE_PROFILER_RUNTIME_VERSION")) cfg.runtime_version = *v;
  if (auto v = get("INFERENCE_PROFILER_ENVIRONMENT")) cfg.deployment_environment = *v;
  if (auto v = get("INFERENCE_PROFILER_STRICT")) {
    cfg.strict = ParseBool("INFERENCE_PROFILER_STRICT", *v);
  }
  cfg.ValidateOrThrow();
  return cfg;
}

std::vector<std::string> ProfilerConfig::Validate() const {
  std::vector<std::string> errors;
  CheckName(errors, "service_name", service_name);
  CheckName(errors, "runtime_version", runtime_version);
  CheckName(errors, "deployment_environment", deployment_environment);
  const bool http = otlp_endpoint.rfind("http://", 0) == 0;
  const bool https = otlp_endpoint.rfind("https://", 0) == 0;
  const std::string rest = otlp_endpoint.substr(http ? 7 : (https ? 8 : 0));
  if (!http && !https) {
    errors.push_back("otlp_endpoint must be an http(s) URL such as http://localhost:4318");
  } else if (rest.empty() || rest[0] == '/' || rest[0] == ':') {
    errors.push_back("otlp_endpoint must include a host");
  } else if (rest.substr(0, rest.find('/')).find('@') != std::string::npos) {
    errors.push_back("otlp_endpoint must not embed credentials; use OTEL_EXPORTER_OTLP_HEADERS");
  } else if (rest.find_first_of("?#") != std::string::npos || HasControlChars(rest)) {
    errors.push_back("otlp_endpoint must not contain a query, fragment or control characters");
  }
  if (!(sample_rate >= 0.0 && sample_rate <= 1.0)) {
    errors.push_back("sample_rate must be between 0 and 1");
  }
  if (export_interval < std::chrono::milliseconds(100) ||
      export_interval > std::chrono::hours(1)) {
    errors.push_back("export_interval must be between 100 ms and 1 h");
  }
  if (export_timeout < std::chrono::milliseconds(100) ||
      export_timeout > std::chrono::minutes(10)) {
    errors.push_back("export_timeout must be between 100 ms and 10 min");
  }
  if (shutdown_timeout < std::chrono::milliseconds(0) ||
      shutdown_timeout > std::chrono::minutes(10)) {
    errors.push_back("shutdown_timeout must be between 0 and 10 min");
  }
  if (max_span_queue_size < 1 || max_span_queue_size > 1000000) {
    errors.push_back("max_span_queue_size must be between 1 and 1000000");
  }
  if (max_attribute_values < 1 || max_attribute_values > 10000) {
    errors.push_back("max_attribute_values must be between 1 and 10000");
  }
  for (const auto& [key, value] : resource_attributes) {
    if (Trim(key).empty() || HasControlChars(key) || HasControlChars(value)) {
      errors.push_back("resource attribute '" + key + "' is invalid");
    }
  }
  return errors;
}

void ProfilerConfig::ValidateOrThrow() const {
  auto errors = Validate();
  if (errors.empty()) return;
  std::string message = "invalid profiler configuration:";
  for (const auto& e : errors) message += " " + e + ";";
  throw ConfigError(message);
}

std::string ProfilerConfig::TracesUrl() const { return BaseUrl(otlp_endpoint) + "/v1/traces"; }
std::string ProfilerConfig::MetricsUrl() const { return BaseUrl(otlp_endpoint) + "/v1/metrics"; }

// ============================================================ exporter wrappers

namespace detail {
namespace {

namespace sdk_trace = opentelemetry::sdk::trace;
namespace sdk_metrics = opentelemetry::sdk::metrics;
namespace sdk_common = opentelemetry::sdk::common;

/// Counts failures and measures export latency; converts exceptions to kFailure.
class CountingSpanExporter final : public sdk_trace::SpanExporter {
 public:
  CountingSpanExporter(std::unique_ptr<sdk_trace::SpanExporter> delegate,
                       std::shared_ptr<ExportStats> stats)
      : delegate_(std::move(delegate)), stats_(std::move(stats)) {}

  std::unique_ptr<sdk_trace::Recordable> MakeRecordable() noexcept override {
    return delegate_->MakeRecordable();
  }

  sdk_common::ExportResult Export(
      const nostd::span<std::unique_ptr<sdk_trace::Recordable>>& spans) noexcept override {
    const auto count = spans.size();
    const auto start = Clock::now();
    sdk_common::ExportResult result = sdk_common::ExportResult::kFailure;
    try {
      result = delegate_->Export(spans);
    } catch (...) {
      result = sdk_common::ExportResult::kFailure;
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    const bool ok = result == sdk_common::ExportResult::kSuccess;
    if (!ok) {
      stats_->span_export_failures.fetch_add(1);
      stats_->dropped_spans.fetch_add(count);
    }
    if (auto* hist = stats_->export_duration.load()) {
      hist->Record(seconds, Attrs{{attr::kSignal, "traces"}}, EmptyContext());
    }
    return result;
  }

  bool ForceFlush(std::chrono::microseconds timeout) noexcept override {
    return delegate_->ForceFlush(timeout);
  }
  bool Shutdown(std::chrono::microseconds timeout) noexcept override {
    return delegate_->Shutdown(timeout);
  }

 private:
  std::unique_ptr<sdk_trace::SpanExporter> delegate_;
  std::shared_ptr<ExportStats> stats_;
};

class CountingMetricExporter final : public sdk_metrics::PushMetricExporter {
 public:
  CountingMetricExporter(std::unique_ptr<sdk_metrics::PushMetricExporter> delegate,
                         std::shared_ptr<ExportStats> stats)
      : delegate_(std::move(delegate)), stats_(std::move(stats)) {}

  sdk_common::ExportResult Export(const sdk_metrics::ResourceMetrics& data) noexcept override {
    sdk_common::ExportResult result = sdk_common::ExportResult::kFailure;
    try {
      result = delegate_->Export(data);
    } catch (...) {
      result = sdk_common::ExportResult::kFailure;
    }
    if (result != sdk_common::ExportResult::kSuccess) stats_->metric_export_failures.fetch_add(1);
    return result;
  }
  sdk_metrics::AggregationTemporality GetAggregationTemporality(
      sdk_metrics::InstrumentType type) const noexcept override {
    return delegate_->GetAggregationTemporality(type);
  }
  bool ForceFlush(std::chrono::microseconds timeout) noexcept override {
    return delegate_->ForceFlush(timeout);
  }
  bool Shutdown(std::chrono::microseconds timeout) noexcept override {
    return delegate_->Shutdown(timeout);
  }

 private:
  std::unique_ptr<sdk_metrics::PushMetricExporter> delegate_;
  std::shared_ptr<ExportStats> stats_;
};

/// Minimal header carrier for W3C trace-context propagation.
class MapCarrier final : public opentelemetry::context::propagation::TextMapCarrier {
 public:
  std::map<std::string, std::string> headers;
  nostd::string_view Get(nostd::string_view key) const noexcept override {
    auto it = headers.find(std::string(key));
    return it == headers.end() ? nostd::string_view{} : nostd::string_view(it->second);
  }
  void Set(nostd::string_view key, nostd::string_view value) noexcept override {
    headers[std::string(key)] = std::string(value);
  }
};

std::string RandomHex(std::size_t bytes) {
  thread_local std::mt19937_64 rng{std::random_device{}()};
  static const char* digits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes * 2);
  for (std::size_t i = 0; i < bytes; ++i) {
    const auto b = static_cast<unsigned>(rng() & 0xffU);
    out.push_back(digits[b >> 4U]);
    out.push_back(digits[b & 0xfU]);
  }
  return out;
}

}  // namespace

void ProfilerImpl::HandleError(const char* where, const std::exception& error) {
  stats->instrumentation_errors.fetch_add(1);
  if (config.strict) throw;
  static std::once_flag logged;
  std::call_once(logged, [&] {
    std::cerr << "inference_profiler: error in " << where << ": " << error.what()
              << " (further errors are counted in inference_profiler.instrumentation.errors)\n";
  });
}

Attrs ProfilerImpl::MetricAttrs(const std::string& model, const std::string& runtime_version) {
  return Attrs{{attr::kModel, limiter->Limit(attr::kModel, model)},
               {attr::kRuntimeVersion, limiter->Limit(attr::kRuntimeVersion, runtime_version)}};
}

}  // namespace detail

// ===================================================================== profiler

std::shared_ptr<Profiler> Profiler::Create(ProfilerConfig config) {
  return Create(std::move(config), ExporterOverrides{});
}

std::shared_ptr<Profiler> Profiler::Create(ProfilerConfig config, ExporterOverrides overrides) {
  namespace sdk_trace = opentelemetry::sdk::trace;
  namespace sdk_metrics = opentelemetry::sdk::metrics;
  namespace otlp = opentelemetry::exporter::otlp;

  config.ValidateOrThrow();
  auto impl = std::make_shared<detail::ProfilerImpl>();
  impl->config = config;
  impl->limiter = std::make_unique<CardinalityLimiter>(config.max_attribute_values);
  std::shared_ptr<Profiler> profiler(new Profiler(impl));
  if (!config.enabled) return profiler;

  opentelemetry::sdk::resource::ResourceAttributes attrs{
      {"service.name", config.service_name},
      {"deployment.environment.name", config.deployment_environment},
      {attr::kRuntimeVersion, config.runtime_version},
      {"telemetry.distro.name", "inference-profiler-cpp"},
      {"telemetry.distro.version", kVersion},
  };
  for (const auto& [key, value] : config.resource_attributes) attrs.SetAttribute(key, value);
  auto resource = opentelemetry::sdk::resource::Resource::Create(attrs);

  // ---- metrics
  std::unique_ptr<sdk_metrics::PushMetricExporter> metric_exporter =
      std::move(overrides.metric_exporter);
  if (!metric_exporter) {
    otlp::OtlpHttpMetricExporterOptions opts;
    opts.url = config.MetricsUrl();
    opts.timeout = config.export_timeout;
    opts.aggregation_temporality = otlp::PreferredAggregationTemporality::kCumulative;
    metric_exporter = otlp::OtlpHttpMetricExporterFactory::Create(opts);
  }
  sdk_metrics::PeriodicExportingMetricReaderOptions reader_opts;
  reader_opts.export_interval_millis = config.export_interval;
  // The SDK requires timeout < interval.
  reader_opts.export_timeout_millis =
      std::min(config.export_timeout, config.export_interval - std::chrono::milliseconds(1));
  std::shared_ptr<sdk_metrics::MetricReader> reader =
      sdk_metrics::PeriodicExportingMetricReaderFactory::Create(
          std::make_unique<detail::CountingMetricExporter>(std::move(metric_exporter),
                                                           impl->stats),
          reader_opts);
  impl->meter_provider = std::shared_ptr<sdk_metrics::MeterProvider>(
      sdk_metrics::MeterProviderFactory::Create(std::make_unique<sdk_metrics::ViewRegistry>(),
                                                resource));
  detail::RegisterHistogramViews(*impl->meter_provider, kScopeName, kVersion);
  impl->meter_provider->AddMetricReader(reader);
  impl->meter = impl->meter_provider->GetMeter(kScopeName, kVersion);
  impl->instruments = detail::Instruments::Create(*impl->meter, impl->stats.get(),
                                                  config.report_process_memory);

  // ---- traces
  std::unique_ptr<sdk_trace::SpanExporter> span_exporter = std::move(overrides.span_exporter);
  const bool injected = span_exporter != nullptr;
  if (!span_exporter) {
    otlp::OtlpHttpExporterOptions opts;
    opts.url = config.TracesUrl();
    opts.timeout = config.export_timeout;
    span_exporter = otlp::OtlpHttpExporterFactory::Create(opts);
  }
  auto counting =
      std::make_unique<detail::CountingSpanExporter>(std::move(span_exporter), impl->stats);
  std::unique_ptr<sdk_trace::SpanProcessor> processor;
  if (injected && !overrides.batch_spans) {
    processor = sdk_trace::SimpleSpanProcessorFactory::Create(std::move(counting));
  } else {
    sdk_trace::BatchSpanProcessorOptions bsp;
    bsp.max_queue_size = config.max_span_queue_size;
    bsp.max_export_batch_size = std::min<std::size_t>(512, config.max_span_queue_size);
    bsp.schedule_delay_millis = std::min(config.export_interval, std::chrono::milliseconds(5000));
    processor = sdk_trace::BatchSpanProcessorFactory::Create(std::move(counting), bsp);
  }
  auto sampler = std::make_unique<sdk_trace::ParentBasedSampler>(
      std::make_shared<sdk_trace::TraceIdRatioBasedSampler>(config.sample_rate));
  impl->tracer_provider =
      std::make_shared<sdk_trace::TracerProvider>(std::move(processor), resource,
                                                  std::move(sampler));
  impl->tracer = impl->tracer_provider->GetTracer(kScopeName, kVersion);

  if (config.set_global_providers) {
    opentelemetry::trace::Provider::SetTracerProvider(
        opentelemetry::nostd::shared_ptr<opentelemetry::trace::TracerProvider>(impl->tracer_provider));
    opentelemetry::metrics::Provider::SetMeterProvider(
        opentelemetry::nostd::shared_ptr<opentelemetry::metrics::MeterProvider>(impl->meter_provider));
  }
  return profiler;
}

Profiler::Profiler(std::shared_ptr<detail::ProfilerImpl> impl) : impl_(std::move(impl)) {}

Profiler::~Profiler() { Shutdown(); }

bool Profiler::enabled() const noexcept {
  return impl_->config.enabled && !impl_->is_shutdown.load();
}
const ProfilerConfig& Profiler::config() const noexcept { return impl_->config; }
uint64_t Profiler::export_failures() const noexcept {
  return impl_->stats->span_export_failures.load() + impl_->stats->metric_export_failures.load();
}
uint64_t Profiler::dropped_spans() const noexcept { return impl_->stats->dropped_spans.load(); }
uint64_t Profiler::instrumentation_errors() const noexcept {
  return impl_->stats->instrumentation_errors.load();
}

bool Profiler::ForceFlush(std::optional<std::chrono::milliseconds> timeout) {
  if (!enabled()) return true;
  const auto t = timeout.value_or(impl_->config.export_timeout);
  bool ok = impl_->tracer_provider->ForceFlush(t);
  ok = impl_->meter_provider->ForceFlush(t) && ok;
  return ok;
}

bool Profiler::Shutdown(std::optional<std::chrono::milliseconds> timeout) {
  std::lock_guard<std::mutex> lock(impl_->shutdown_mutex);
  if (impl_->is_shutdown.exchange(true) || !impl_->config.enabled) return true;
  const auto t = timeout.value_or(impl_->config.shutdown_timeout);
  auto impl = impl_;
  // Run on a helper thread so a hung exporter cannot block the caller past `t`.
  auto done = std::make_shared<std::promise<bool>>();
  auto result = done->get_future();
  std::thread worker([impl, done, t] {
    bool ok = impl->tracer_provider->Shutdown(t);
    ok = impl->meter_provider->Shutdown(t) && ok;
    done->set_value(ok);
  });
  if (result.wait_for(t + std::chrono::milliseconds(250)) == std::future_status::ready) {
    worker.join();
    return result.get();
  }
  worker.detach();  // keeps `impl` alive until it finishes
  std::cerr << "inference_profiler: shutdown exceeded " << t.count()
            << " ms; abandoning pending telemetry\n";
  return false;
}

RequestScope Profiler::StartRequest(std::string model, std::optional<int64_t> prompt_tokens) {
  RequestOptions opts;
  opts.model = std::move(model);
  opts.prompt_tokens = prompt_tokens;
  return StartRequest(opts);
}

RequestScope Profiler::StartRequest(const RequestOptions& options) {
  namespace otel_trace = opentelemetry::trace;
  if (!enabled()) return RequestScope{};
  try {
    if (options.model.empty()) throw std::invalid_argument("RequestOptions::model is required");
    if (options.prompt_tokens && *options.prompt_tokens < 0) {
      throw std::invalid_argument("prompt_tokens must be >= 0");
    }
    auto state = std::make_shared<detail::RequestState>();
    state->profiler = impl_;
    const std::string runtime =
        options.runtime_version.empty() ? impl_->config.runtime_version : options.runtime_version;
    state->metric_attrs = impl_->MetricAttrs(options.model, runtime);
    state->uncaught_at_start = std::uncaught_exceptions();

    otel_trace::StartSpanOptions span_opts;
    span_opts.kind = otel_trace::SpanKind::kServer;
    state->start_system = std::chrono::system_clock::now();
    state->start = detail::Clock::now();
    span_opts.start_system_time = opentelemetry::common::SystemTimestamp(state->start_system);
    span_opts.start_steady_time = opentelemetry::common::SteadyTimestamp(state->start);
    if (!options.traceparent.empty()) {
      detail::MapCarrier carrier;
      carrier.headers["traceparent"] = options.traceparent;
      otel_trace::propagation::HttpTraceContext propagator;
      auto empty = opentelemetry::context::Context{};
      auto ctx = propagator.Extract(carrier, empty);
      auto parent = otel_trace::GetSpan(ctx)->GetContext();
      if (parent.IsValid()) span_opts.parent = parent;
    }
    state->span = impl_->tracer->StartSpan("inference.request", span_opts);
    auto& span = *state->span;
    span.SetAttribute(attr::kModel, options.model);
    span.SetAttribute(attr::kRuntimeVersion, runtime);
    span.SetAttribute(attr::kRequestId,
                      options.request_id.empty() ? detail::RandomHex(16) : options.request_id);
    if (options.prompt_tokens) span.SetAttribute(attr::kInputTokens, *options.prompt_tokens);
    for (const auto& [key, value] : options.attributes) span.SetAttribute(key, value);

    auto& inst = impl_->instruments;
    inst.requests_in_flight->Add(1, state->metric_attrs);
    inst.requests_active->Add(1, state->metric_attrs);
    if (options.prompt_tokens && *options.prompt_tokens > 0) {
      auto attrs = state->metric_attrs;
      attrs[attr::kTokenType] = "input";
      inst.tokens->Add(static_cast<uint64_t>(*options.prompt_tokens), attrs);
    }
    return RequestScope(state);
  } catch (const std::exception& e) {
    impl_->HandleError("StartRequest", e);
    return RequestScope{};
  }
}

// The traceparent helpers live here because they need the propagator.
std::string RequestScope::TraceParent() const {
  if (!state_ || !state_->span->GetContext().IsValid()) return {};
  detail::MapCarrier carrier;
  opentelemetry::context::Context ctx;
  ctx = opentelemetry::trace::SetSpan(ctx, state_->span);
  opentelemetry::trace::propagation::HttpTraceContext().Inject(carrier, ctx);
  return carrier.headers["traceparent"];
}

}  // namespace inference_profiler
