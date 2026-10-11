// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0
//
// Metric names and attribute keys shared with the Python SDK. Keep in sync with
// src/inference_profiler/metrics.py and docs/metrics.md.
#pragma once

#include <cstddef>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>

namespace inference_profiler {
namespace metric_names {

inline constexpr const char* kRequestDuration = "inference.request.duration";
inline constexpr const char* kQueueDuration = "inference.queue.duration";
inline constexpr const char* kPrefillDuration = "inference.prefill.duration";
inline constexpr const char* kTimeToFirstToken = "inference.time_to_first_token";
inline constexpr const char* kInterTokenLatency = "inference.inter_token.latency";
inline constexpr const char* kGenerationDuration = "inference.generation.duration";
inline constexpr const char* kPhaseDuration = "inference.phase.duration";
inline constexpr const char* kGenerationThroughput = "inference.generation.throughput";
inline constexpr const char* kRequests = "inference.requests";
inline constexpr const char* kTokens = "inference.tokens";
inline constexpr const char* kRequestsInFlight = "inference.requests.in_flight";
inline constexpr const char* kRequestsActive = "inference.requests.active";
inline constexpr const char* kQueueDepth = "inference.queue.depth";
inline constexpr const char* kCacheHits = "inference.cache.hits";
inline constexpr const char* kCacheMisses = "inference.cache.misses";
inline constexpr const char* kCacheInsertions = "inference.cache.insertions";
inline constexpr const char* kCacheEvictions = "inference.cache.evictions";
inline constexpr const char* kCacheOperationDuration = "inference.cache.operation.duration";
inline constexpr const char* kProcessMemoryUsage = "process.memory.usage";
inline constexpr const char* kExportErrors = "inference_profiler.export.errors";
inline constexpr const char* kExportDuration = "inference_profiler.export.duration";
inline constexpr const char* kTelemetryDropped = "inference_profiler.telemetry.dropped";
inline constexpr const char* kProfilerEnabled = "inference_profiler.enabled";
inline constexpr const char* kInstrumentationErrors = "inference_profiler.instrumentation.errors";

}  // namespace metric_names

namespace attr {

inline constexpr const char* kModel = "gen_ai.request.model";
inline constexpr const char* kRuntimeVersion = "inference.runtime.version";
inline constexpr const char* kStatus = "inference.status";
inline constexpr const char* kPhase = "inference.phase";
inline constexpr const char* kTokenType = "gen_ai.token.type";
inline constexpr const char* kCacheOperation = "inference.cache.operation";
inline constexpr const char* kSignal = "inference_profiler.signal";
inline constexpr const char* kRequestId = "inference.request.id";
inline constexpr const char* kInputTokens = "gen_ai.usage.input_tokens";
inline constexpr const char* kOutputTokens = "gen_ai.usage.output_tokens";
inline constexpr const char* kErrorType = "error.type";

inline constexpr const char* kStatusOk = "ok";
inline constexpr const char* kStatusError = "error";
inline constexpr const char* kStatusCancelled = "cancelled";
inline constexpr const char* kOtherValue = "_other";

}  // namespace attr

/// Caps distinct values per attribute key to keep metric cardinality bounded.
/// Admitted values are looked up under a shared lock; only new values take the
/// exclusive lock.
class CardinalityLimiter {
 public:
  explicit CardinalityLimiter(std::size_t max_values) : max_values_(max_values) {}

  std::string Limit(const std::string& key, const std::string& value);
  bool Overflowed(const std::string& key) const;

 private:
  std::size_t max_values_;
  mutable std::shared_mutex mutex_;
  std::map<std::string, std::set<std::string>> seen_;
  std::set<std::string> overflowed_;
};

/// Resident set size of this process in bytes, or 0 if unsupported on this platform.
long long ProcessRssBytes() noexcept;

}  // namespace inference_profiler
