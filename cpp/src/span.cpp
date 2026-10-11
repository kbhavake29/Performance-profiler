// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0

#include "inference_profiler/span.hpp"

#include <cctype>
#include <exception>
#include <stdexcept>
#include <utility>

#include "internal.hpp"
#include "opentelemetry/trace/span_startoptions.h"

namespace inference_profiler {

namespace detail {
namespace {

bool ValidPhaseName(std::string_view name) {
  if (name.empty() || name.size() > 64 || !std::islower(static_cast<unsigned char>(name[0]))) {
    return false;
  }
  for (char c : name) {
    const auto u = static_cast<unsigned char>(c);
    if (!(std::islower(u) || std::isdigit(u) || c == '_')) return false;
  }
  return true;
}

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

}  // namespace

nostd::string_view Sv(std::string_view s) { return nostd::string_view(s.data(), s.size()); }

namespace {

}  // namespace

std::unique_ptr<PhaseState> StartPhase(const std::shared_ptr<RequestState>& request,
                                       const nostd::shared_ptr<otel_trace::Span>& parent,
                                       const std::string& name, const std::string& span_name) {
  if (!ValidPhaseName(name)) throw std::invalid_argument("invalid phase name '" + name + "'");
  auto phase = std::make_unique<PhaseState>();
  phase->request = request;
  phase->name = name;
  otel_trace::StartSpanOptions opts;
  opts.parent = parent->GetContext();
  opts.kind = otel_trace::SpanKind::kInternal;
  phase->span = request->profiler->tracer->StartSpan(span_name, opts);
  phase->span->SetAttribute(attr::kPhase, name);
  phase->uncaught_at_start = std::uncaught_exceptions();
  if (name == "queue") request->EnterQueue();
  phase->start = Clock::now();
  return phase;
}

void RequestState::EnterQueue() {
  std::lock_guard<std::mutex> lock(mutex);
  if (queued) return;
  queued = true;
  auto& inst = profiler->instruments;
  inst.queue_depth->Add(1, metric_attrs);
  inst.requests_active->Add(-1, metric_attrs);
}

void RequestState::LeaveQueueLocked() {
  if (!queued) return;
  queued = false;
  auto& inst = profiler->instruments;
  inst.queue_depth->Add(-1, metric_attrs);
  inst.requests_active->Add(1, metric_attrs);
}

void RequestState::PhaseFinished(const std::string& name, double seconds) {
  auto& inst = profiler->instruments;
  const auto ctx = EmptyContext();
  if (name == "queue") {
    std::lock_guard<std::mutex> lock(mutex);
    LeaveQueueLocked();
    queue_seconds += seconds;
    inst.queue_duration->Record(seconds, metric_attrs, ctx);
  } else if (name == "prefill") {
    inst.prefill_duration->Record(seconds, metric_attrs, ctx);
  } else if (name == "generation") {
    {
      std::lock_guard<std::mutex> lock(mutex);
      generation_seconds = generation_seconds.value_or(0.0) + seconds;
    }
    inst.generation_duration->Record(seconds, metric_attrs, ctx);
  } else if (name != "cache_lookup" && name != "cache_insert" && name != "cache_evict") {
    auto attrs = metric_attrs;
    attrs[attr::kPhase] = profiler->limiter->Limit(attr::kPhase, name);
    inst.phase_duration->Record(seconds, attrs, ctx);
  }
}

void RequestState::RecordTtft(double seconds) {
  if (ttft_recorded) return;
  ttft_recorded = true;
  otel_trace::StartSpanOptions opts;
  opts.parent = span->GetContext();
  opts.start_system_time = opentelemetry::common::SystemTimestamp(start_system);
  opts.start_steady_time = opentelemetry::common::SteadyTimestamp(start);
  auto ttft = profiler->tracer->StartSpan("inference.time_to_first_token", opts);
  ttft->SetAttribute(attr::kPhase, "time_to_first_token");
  otel_trace::EndSpanOptions end;
  end.end_steady_time = opentelemetry::common::SteadyTimestamp(
      start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds)));
  ttft->End(end);
  span->SetAttribute("inference.time_to_first_token_ms", seconds * 1000.0);
  profiler->instruments.time_to_first_token->Record(seconds, metric_attrs, EmptyContext());
}

}  // namespace detail

// ================================================================= PhaseScope

PhaseScope::PhaseScope(std::unique_ptr<detail::PhaseState> state) : state_(std::move(state)) {}
PhaseScope::PhaseScope(PhaseScope&&) noexcept = default;
PhaseScope& PhaseScope::operator=(PhaseScope&& other) noexcept {
  if (this != &other) {
    End();
    state_ = std::move(other.state_);
    duration_ms_ = other.duration_ms_;
  }
  return *this;
}

PhaseScope::~PhaseScope() {
  try {
    if (state_ && std::uncaught_exceptions() > state_->uncaught_at_start) Fail("exception");
    End();
  } catch (...) {  // strict mode must never throw out of a destructor
  }
}

PhaseScope PhaseScope::Phase(std::string_view name) {
  if (!state_) return PhaseScope{};
  try {
    const std::string n(name);
    return PhaseScope(detail::StartPhase(state_->request, state_->span, n, "inference." + n));
  } catch (const std::exception& e) {
    state_->request->profiler->HandleError("Phase", e);
    return PhaseScope{};
  }
}

void PhaseScope::SetCacheHit(bool hit) {
  if (state_) state_->hit = hit;
}

void PhaseScope::SetAttribute(std::string_view key, std::string_view value) {
  if (state_) state_->span->SetAttribute(detail::Sv(key), detail::Sv(value));
}
void PhaseScope::SetAttribute(std::string_view key, int64_t value) {
  if (state_) state_->span->SetAttribute(detail::Sv(key), value);
}
void PhaseScope::SetAttribute(std::string_view key, double value) {
  if (state_) state_->span->SetAttribute(detail::Sv(key), value);
}
void PhaseScope::SetAttribute(std::string_view key, bool value) {
  if (state_) state_->span->SetAttribute(detail::Sv(key), value);
}

void PhaseScope::Fail(std::string_view error_type) {
  if (!state_) return;
  state_->failed = true;
  state_->span->SetStatus(opentelemetry::trace::StatusCode::kError, detail::Sv(error_type));
  state_->span->SetAttribute(attr::kErrorType, detail::Sv(error_type));
}

void PhaseScope::End() {
  if (!state_) return;
  auto state = std::move(state_);
  const double seconds = detail::Seconds(detail::Clock::now() - state->start);
  duration_ms_ = seconds * 1000.0;
  auto& req = *state->request;
  try {
    auto& inst = req.profiler->instruments;
    if (!state->cache_operation.empty()) {
      auto attrs = req.metric_attrs;
      attrs[attr::kCacheOperation] = state->cache_operation;
      inst.cache_operation_duration->Record(seconds, attrs, detail::EmptyContext());
      if (state->cache_operation == "lookup" && state->hit.has_value()) {
        state->span->SetAttribute("inference.cache.hit", *state->hit);
        (*state->hit ? inst.cache_hits : inst.cache_misses)->Add(1, req.metric_attrs);
      } else if (state->cache_operation == "insert") {
        inst.cache_insertions->Add(static_cast<uint64_t>(state->entries), req.metric_attrs);
      } else if (state->cache_operation == "evict") {
        inst.cache_evictions->Add(static_cast<uint64_t>(state->entries), req.metric_attrs);
      }
    }
    req.PhaseFinished(state->name, seconds);
    state->span->End();
  } catch (const std::exception& e) {
    req.profiler->HandleError("PhaseScope::End", e);
  }
}

// =============================================================== RequestScope

RequestScope::RequestScope(std::shared_ptr<detail::RequestState> state)
    : state_(std::move(state)) {}
RequestScope::RequestScope(RequestScope&&) noexcept = default;
RequestScope& RequestScope::operator=(RequestScope&& other) noexcept {
  if (this != &other) {
    End();
    state_ = std::move(other.state_);
  }
  return *this;
}

RequestScope::~RequestScope() {
  try {
    if (state_ && !state_->ended.load() &&
        std::uncaught_exceptions() > state_->uncaught_at_start) {
      Fail("exception");
    }
    End();
  } catch (...) {  // strict mode must never throw out of a destructor
  }
}

PhaseScope RequestScope::Phase(std::string_view name) {
  if (!state_ || state_->ended.load()) return PhaseScope{};
  try {
    const std::string n(name);
    return PhaseScope(detail::StartPhase(state_, state_->span, n, "inference." + n));
  } catch (const std::exception& e) {
    state_->profiler->HandleError("Phase", e);
    return PhaseScope{};
  }
}

PhaseScope RequestScope::CacheLookup(std::string_view cache) {
  if (!state_) return PhaseScope{};
  try {
    const std::string name = "cache_lookup";
    auto phase = detail::StartPhase(state_, state_->span, name, "inference." + name);
    phase->cache_operation = "lookup";
    phase->span->SetAttribute("inference.cache.name", detail::Sv(cache));
    return PhaseScope(std::move(phase));
  } catch (const std::exception& e) {
    state_->profiler->HandleError("CacheLookup", e);
    return PhaseScope{};
  }
}

PhaseScope RequestScope::CacheInsert(std::string_view cache, int64_t entries) {
  if (!state_) return PhaseScope{};
  try {
    auto phase = detail::StartPhase(state_, state_->span, "cache_insert", "inference.cache_insert");
    phase->cache_operation = "insert";
    phase->entries = entries;
    phase->span->SetAttribute("inference.cache.name", detail::Sv(cache));
    phase->span->SetAttribute("inference.cache.entries", entries);
    return PhaseScope(std::move(phase));
  } catch (const std::exception& e) {
    state_->profiler->HandleError("CacheInsert", e);
    return PhaseScope{};
  }
}

PhaseScope RequestScope::CacheEvict(std::string_view cache, int64_t entries) {
  if (!state_) return PhaseScope{};
  try {
    auto phase = detail::StartPhase(state_, state_->span, "cache_evict", "inference.cache_evict");
    phase->cache_operation = "evict";
    phase->entries = entries;
    phase->span->SetAttribute("inference.cache.name", detail::Sv(cache));
    phase->span->SetAttribute("inference.cache.entries", entries);
    return PhaseScope(std::move(phase));
  } catch (const std::exception& e) {
    state_->profiler->HandleError("CacheEvict", e);
    return PhaseScope{};
  }
}

void RequestScope::MarkFirstToken() {
  if (!state_ || state_->first_token) return;
  const auto now = detail::Clock::now();
  state_->first_token = now;
  state_->last_token = now;
  state_->token_count = 1;
  try {
    state_->RecordTtft(detail::Seconds(now - state_->start));
  } catch (const std::exception& e) {
    state_->profiler->HandleError("MarkFirstToken", e);
  }
}

void RequestScope::RecordToken() {
  if (!state_) return;
  if (!state_->first_token) {
    MarkFirstToken();
    return;
  }
  const auto now = detail::Clock::now();
  const double gap = detail::Seconds(now - *state_->last_token);
  state_->last_token = now;
  ++state_->token_count;
  state_->profiler->instruments.inter_token_latency->Record(gap, state_->metric_attrs,
                                                            detail::EmptyContext());
}

void RequestScope::RecordGeneration(int64_t generated_tokens, std::optional<double> ttft_ms) {
  if (!state_) return;
  try {
    if (generated_tokens < 0) throw std::invalid_argument("generated_tokens must be >= 0");
    if (ttft_ms && *ttft_ms < 0) throw std::invalid_argument("ttft_ms must be >= 0");
    auto& span = *state_->span;
    span.SetAttribute(attr::kOutputTokens, generated_tokens);
    if (ttft_ms && !state_->ttft_recorded) state_->RecordTtft(*ttft_ms / 1000.0);
    auto& inst = state_->profiler->instruments;
    if (generated_tokens > 0) {
      auto attrs = state_->metric_attrs;
      attrs[attr::kTokenType] = "output";
      inst.tokens->Add(static_cast<uint64_t>(generated_tokens), attrs);
    }
    std::optional<double> gen;
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      gen = state_->generation_seconds;
    }
    if (gen && *gen > 0 && generated_tokens > 0) {
      const double tps = static_cast<double>(generated_tokens) / *gen;
      span.SetAttribute("inference.generation.tokens_per_second", tps);
      inst.generation_throughput->Record(tps, state_->metric_attrs, detail::EmptyContext());
    }
  } catch (const std::exception& e) {
    state_->profiler->HandleError("RecordGeneration", e);
  }
}

void RequestScope::SetAttribute(std::string_view key, std::string_view value) {
  if (state_) state_->span->SetAttribute(detail::Sv(key), detail::Sv(value));
}
void RequestScope::SetAttribute(std::string_view key, int64_t value) {
  if (state_) state_->span->SetAttribute(detail::Sv(key), value);
}
void RequestScope::SetAttribute(std::string_view key, double value) {
  if (state_) state_->span->SetAttribute(detail::Sv(key), value);
}
void RequestScope::SetAttribute(std::string_view key, bool value) {
  if (state_) state_->span->SetAttribute(detail::Sv(key), value);
}

void RequestScope::Fail(std::string_view error_type) {
  if (!state_) return;
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->status = attr::kStatusError;
  state_->error_type = std::string(error_type.substr(0, 128));
}

void RequestScope::Cancel(std::string_view reason) {
  if (!state_) return;
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->status = attr::kStatusCancelled;
  if (!reason.empty()) state_->span->SetAttribute("inference.cancel.reason", detail::Sv(reason.substr(0, 128)));
}

void RequestScope::End() {
  if (!state_ || state_->ended.exchange(true)) return;
  auto& s = *state_;
  const double seconds = detail::Seconds(detail::Clock::now() - s.start);
  try {
    std::string status;
    std::string error_type;
    double queue_seconds = 0.0;
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      s.LeaveQueueLocked();
      status = s.status;
      error_type = s.error_type;
      queue_seconds = s.queue_seconds;
    }
    auto& span = *s.span;
    span.SetAttribute(attr::kStatus, status);
    span.SetAttribute("inference.queue.duration_ms", queue_seconds * 1000.0);
    if (status == attr::kStatusError) {
      span.SetAttribute(attr::kErrorType, error_type.empty() ? "error" : error_type);
      span.SetStatus(opentelemetry::trace::StatusCode::kError,
                     error_type.empty() ? "error" : error_type);
    }
    auto& inst = s.profiler->instruments;
    auto with_status = s.metric_attrs;
    with_status[attr::kStatus] = status;
    inst.request_duration->Record(seconds, with_status, detail::EmptyContext());
    inst.requests->Add(1, with_status);
    inst.requests_in_flight->Add(-1, s.metric_attrs);
    inst.requests_active->Add(-1, s.metric_attrs);
    span.End();
  } catch (const std::exception& e) {
    s.profiler->HandleError("RequestScope::End", e);
  }
}

bool RequestScope::ended() const noexcept { return !state_ || state_->ended.load(); }

std::string RequestScope::TraceId() const {
  if (!state_) return {};
  const auto ctx = state_->span->GetContext();
  if (!ctx.IsValid()) return {};
  char buf[32];
  ctx.trace_id().ToLowerBase16(buf);
  return std::string(buf, sizeof(buf));
}

std::string RequestScope::Status() const {
  if (!state_) return attr::kStatusOk;
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->status;
}

}  // namespace inference_profiler
