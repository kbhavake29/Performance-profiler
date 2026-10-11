// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace inference_profiler {

/// Thrown for missing, malformed or unsafe configuration values.
class ConfigError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

/// Profiler configuration. Mirrors the Python `ProfilerConfig`.
///
/// Precedence: values assigned in code after `FromEnv()` > environment variables > defaults.
/// A default-constructed config never reads the environment.
struct ProfilerConfig {
  std::string service_name = "inference-service";
  /// Base OTLP/HTTP endpoint; `/v1/traces` and `/v1/metrics` are appended.
  std::string otlp_endpoint = "http://localhost:4318";
  bool enabled = true;
  /// Fraction of request traces sampled (parent-based). Metrics are never sampled.
  double sample_rate = 1.0;
  std::chrono::milliseconds export_interval{5000};
  std::chrono::milliseconds export_timeout{10000};
  std::chrono::milliseconds shutdown_timeout{5000};
  std::string runtime_version = "unknown";
  std::string deployment_environment = "local";
  /// Rethrow internal instrumentation errors instead of logging them.
  bool strict = false;
  /// Bounded queue of spans waiting for export.
  std::size_t max_span_queue_size = 2048;
  /// Distinct values per metric attribute before values collapse to "_other".
  std::size_t max_attribute_values = 64;
  /// Register providers as the OpenTelemetry globals (off by default).
  bool set_global_providers = false;
  /// Report process RSS as `process.memory.usage`.
  bool report_process_memory = true;
  std::map<std::string, std::string> resource_attributes;

  using EnvLookup = std::function<const char*(const char*)>;

  /// Defaults overlaid with OTEL_* / INFERENCE_PROFILER_* environment variables.
  /// Throws ConfigError naming the offending variable.
  static ProfilerConfig FromEnv();
  /// Same as FromEnv() with an injectable lookup (for tests).
  static ProfilerConfig FromEnv(const EnvLookup& lookup);

  /// Returns a list of problems; empty when the configuration is valid.
  std::vector<std::string> Validate() const;
  /// Throws ConfigError listing every problem.
  void ValidateOrThrow() const;

  std::string TracesUrl() const;
  std::string MetricsUrl() const;
};

}  // namespace inference_profiler
