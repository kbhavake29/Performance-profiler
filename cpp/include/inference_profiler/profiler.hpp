// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0
//
// Native instrumentation for LLM inference servers.
//
//   auto profiler = inference_profiler::Profiler::Create(
//       inference_profiler::ProfilerConfig::FromEnv());
//   {
//     inference_profiler::RequestOptions opts;
//     opts.model = "demo-model";
//     opts.prompt_tokens = 512;
//     auto request = profiler->StartRequest(opts);
//     { auto prefill = request.Prefill(); /* run prefill */ }
//     { auto gen = request.Generation(); for (...) request.RecordToken(); }
//     request.RecordGeneration(128);
//   }  // request span ends here
//   profiler->Shutdown();
#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include "inference_profiler/config.hpp"
#include "inference_profiler/metrics.hpp"
#include "inference_profiler/span.hpp"

namespace inference_profiler {

namespace detail {
struct ProfilerImpl;
}

/// Per-request metadata. Never put prompt or generated text here.
struct RequestOptions {
  std::string model;
  /// Span-only attribute (never a metric label). Generated if empty.
  std::string request_id;
  std::optional<int64_t> prompt_tokens;
  /// Overrides ProfilerConfig::runtime_version for this request.
  std::string runtime_version;
  /// Incoming W3C traceparent header to continue a remote trace.
  std::string traceparent;
  /// Extra span attributes (low-cardinality values recommended).
  std::map<std::string, std::string> attributes;
};

/// Exporter injection for tests and custom pipelines; see testing.hpp.
struct ExporterOverrides;

class Profiler {
 public:
  /// Validates `config` (throws ConfigError) and builds the OTLP pipeline.
  /// Exporters connect lazily: an unreachable collector does not fail creation.
  static std::shared_ptr<Profiler> Create(ProfilerConfig config);
  static std::shared_ptr<Profiler> Create(ProfilerConfig config, ExporterOverrides overrides);

  ~Profiler();
  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  RequestScope StartRequest(const RequestOptions& options);
  RequestScope StartRequest(std::string model, std::optional<int64_t> prompt_tokens = std::nullopt);

  /// Exports pending telemetry. Returns false on timeout or failure.
  bool ForceFlush(std::optional<std::chrono::milliseconds> timeout = std::nullopt);
  /// Flushes and stops exporters within the timeout (default: config.shutdown_timeout).
  /// Idempotent. Returns false if the deadline was exceeded.
  bool Shutdown(std::optional<std::chrono::milliseconds> timeout = std::nullopt);

  bool enabled() const noexcept;
  const ProfilerConfig& config() const noexcept;
  /// Failed export calls (traces + metrics) since creation.
  uint64_t export_failures() const noexcept;
  /// Spans dropped because their export failed.
  uint64_t dropped_spans() const noexcept;
  /// Internal instrumentation errors swallowed in non-strict mode.
  uint64_t instrumentation_errors() const noexcept;

 private:
  explicit Profiler(std::shared_ptr<detail::ProfilerImpl> impl);
  std::shared_ptr<detail::ProfilerImpl> impl_;
};

}  // namespace inference_profiler
