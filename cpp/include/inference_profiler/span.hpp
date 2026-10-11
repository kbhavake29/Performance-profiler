// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0
//
// RAII request and phase scopes. A scope ends exactly once: explicitly via
// End(), or in its destructor. If the destructor runs during stack unwinding
// (an exception is propagating), the scope is marked as failed.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace inference_profiler {

namespace detail {
struct RequestState;
struct PhaseState;
}  // namespace detail

/// A timed child span of a request (or of another phase). Move-only.
class PhaseScope {
 public:
  PhaseScope() = default;  // inert scope (profiling disabled)
  PhaseScope(PhaseScope&&) noexcept;
  PhaseScope& operator=(PhaseScope&&) noexcept;
  PhaseScope(const PhaseScope&) = delete;
  PhaseScope& operator=(const PhaseScope&) = delete;
  ~PhaseScope();

  /// Nested phase whose span is a child of this phase.
  PhaseScope Phase(std::string_view name);
  /// For cache lookups: whether the lookup hit. Must be set before End().
  void SetCacheHit(bool hit);
  void SetAttribute(std::string_view key, std::string_view value);
  void SetAttribute(std::string_view key, int64_t value);
  void SetAttribute(std::string_view key, double value);
  void SetAttribute(std::string_view key, bool value);
  /// Marks the phase span as failed (does not end it).
  void Fail(std::string_view error_type);
  /// Ends the span and records metrics. Idempotent.
  void End();
  bool active() const noexcept { return state_ != nullptr; }
  /// Duration in milliseconds once ended, otherwise nullopt.
  std::optional<double> duration_ms() const noexcept { return duration_ms_; }

 private:
  friend class RequestScope;
  explicit PhaseScope(std::unique_ptr<detail::PhaseState> state);
  std::unique_ptr<detail::PhaseState> state_;
  std::optional<double> duration_ms_;
};

/// One inference request. Create with Profiler::StartRequest(). Move-only.
class RequestScope {
 public:
  RequestScope() = default;  // inert scope (profiling disabled)
  RequestScope(RequestScope&&) noexcept;
  RequestScope& operator=(RequestScope&&) noexcept;
  RequestScope(const RequestScope&) = delete;
  RequestScope& operator=(const RequestScope&) = delete;
  ~RequestScope();

  /// Generic phase; queue/prefill/generation get dedicated histograms.
  PhaseScope Phase(std::string_view name);
  PhaseScope Queue() { return Phase("queue"); }
  PhaseScope Prefill() { return Phase("prefill"); }
  PhaseScope Generation() { return Phase("generation"); }
  /// Cache operations; call SetCacheHit() on the lookup scope.
  PhaseScope CacheLookup(std::string_view cache = "kv");
  PhaseScope CacheInsert(std::string_view cache = "kv", int64_t entries = 1);
  PhaseScope CacheEvict(std::string_view cache = "kv", int64_t entries = 1);

  /// First output token is available; TTFT is measured from request start.
  void MarkFirstToken();
  /// Once per generated token: records inter-token latency (no per-token span).
  void RecordToken();
  /// Final token count; `ttft_ms` is used only if MarkFirstToken() was not called.
  void RecordGeneration(int64_t generated_tokens, std::optional<double> ttft_ms = std::nullopt);

  void SetAttribute(std::string_view key, std::string_view value);
  void SetAttribute(std::string_view key, int64_t value);
  void SetAttribute(std::string_view key, double value);
  void SetAttribute(std::string_view key, bool value);

  /// Marks the request failed with a short, non-sensitive error type.
  void Fail(std::string_view error_type);
  /// Marks the request cancelled (not an error).
  void Cancel(std::string_view reason = {});
  /// Ends the request span and records request metrics. Idempotent.
  void End();

  bool active() const noexcept { return state_ != nullptr; }
  bool ended() const noexcept;
  /// 32-hex-char trace id, or empty if not recording.
  std::string TraceId() const;
  /// W3C `traceparent` value to propagate downstream, or empty.
  std::string TraceParent() const;
  /// Final status: "ok", "error" or "cancelled".
  std::string Status() const;

 private:
  friend class Profiler;
  explicit RequestScope(std::shared_ptr<detail::RequestState> state);
  std::shared_ptr<detail::RequestState> state_;
};

}  // namespace inference_profiler
