// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <string>
#include <thread>

#include "test_harness.hpp"

using inference_profiler::RequestOptions;
using inference_profiler::RequestScope;

namespace {

std::string Hex(const opentelemetry::trace::TraceId& id) {
  char buf[32];
  id.ToLowerBase16(buf);
  return std::string(buf, 32);
}
std::string Hex(const opentelemetry::trace::SpanId& id) {
  char buf[16];
  id.ToLowerBase16(buf);
  return std::string(buf, 16);
}

template <typename T>
T Attr(const test::sdk_trace::SpanData* span, const std::string& key) {
  const auto& attrs = span->GetAttributes();
  auto it = attrs.find(key);
  if (it == attrs.end()) throw test::Failure("missing attribute " + key);
  return opentelemetry::nostd::get<T>(it->second);
}

}  // namespace

TEST_CASE(span_request_lifecycle) {
  test::Harness h;
  {
    RequestOptions opts;
    opts.model = "demo-model";
    opts.request_id = "req-1";
    opts.prompt_tokens = 128;
    auto request = h.profiler->StartRequest(opts);
    CHECK(request.active());
    { auto q = request.Queue(); }
    {
      auto p = request.Prefill();
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    {
      auto g = request.Generation();
      for (int i = 0; i < 4; ++i) request.RecordToken();
    }
    request.RecordGeneration(4);
  }
  const auto* root = h.Find("inference.request");
  CHECK(root != nullptr);
  CHECK_EQ(Attr<std::string>(root, "gen_ai.request.model"), std::string("demo-model"));
  CHECK_EQ(Attr<std::string>(root, "inference.request.id"), std::string("req-1"));
  CHECK_EQ(Attr<int64_t>(root, "gen_ai.usage.input_tokens"), int64_t{128});
  CHECK_EQ(Attr<int64_t>(root, "gen_ai.usage.output_tokens"), int64_t{4});
  CHECK_EQ(Attr<std::string>(root, "inference.status"), std::string("ok"));
  CHECK_EQ(Attr<std::string>(root, "inference.runtime.version"), std::string("test-1.0"));
  CHECK(root->GetStatus() != opentelemetry::trace::StatusCode::kError);
  for (const char* name : {"inference.queue", "inference.prefill", "inference.generation",
                           "inference.time_to_first_token"}) {
    const auto* child = h.Find(name);
    CHECK(child != nullptr);
    CHECK_EQ(Hex(child->GetParentSpanId()), Hex(root->GetSpanId()));
    CHECK_EQ(Hex(child->GetTraceId()), Hex(root->GetTraceId()));
  }
  CHECK(h.Find("inference.prefill")->GetDuration() >= std::chrono::milliseconds(2));
}

TEST_CASE(span_nested_phases) {
  test::Harness h;
  {
    auto request = h.profiler->StartRequest("m");
    auto gen = request.Generation();
    {
      auto sampling = gen.Phase("sampling");
      auto inner = sampling.Phase("top_k");
    }
  }
  const auto* gen = h.Find("inference.generation");
  const auto* sampling = h.Find("inference.sampling");
  const auto* inner = h.Find("inference.top_k");
  CHECK(gen && sampling && inner);
  CHECK_EQ(Hex(sampling->GetParentSpanId()), Hex(gen->GetSpanId()));
  CHECK_EQ(Hex(inner->GetParentSpanId()), Hex(sampling->GetSpanId()));
}

TEST_CASE(span_raii_on_scope_exit) {
  test::Harness h;
  {
    auto request = h.profiler->StartRequest("m");
    auto phase = request.Prefill();
    CHECK_EQ(h.Count("inference.prefill"), 0);
    phase.End();
    CHECK(phase.duration_ms().has_value());
    CHECK_EQ(h.Count("inference.prefill"), 1);
    phase.End();  // idempotent
    CHECK_EQ(h.Count("inference.prefill"), 1);
    CHECK(!request.ended());
  }
  CHECK_EQ(h.Count("inference.request"), 1);
  auto request = h.profiler->StartRequest("m");
  request.End();
  request.End();
  CHECK(request.ended());
  CHECK_EQ(h.Count("inference.request"), 2);
}

TEST_CASE(span_exception_marks_error) {
  test::Harness h;
  try {
    auto request = h.profiler->StartRequest("m");
    auto prefill = request.Prefill();
    throw std::runtime_error("engine fault");
  } catch (const std::runtime_error&) {
  }
  const auto* root = h.Find("inference.request");
  const auto* prefill = h.Find("inference.prefill");
  CHECK(root && prefill);
  CHECK(root->GetStatus() == opentelemetry::trace::StatusCode::kError);
  CHECK(prefill->GetStatus() == opentelemetry::trace::StatusCode::kError);
  CHECK_EQ(Attr<std::string>(root, "inference.status"), std::string("error"));
  CHECK_EQ(Attr<std::string>(root, "error.type"), std::string("exception"));
  CHECK_EQ(h.MetricTotal("inference.requests", {{"inference.status", "error"}}), 1.0);
}

TEST_CASE(span_cancel_and_fail) {
  test::Harness h;
  {
    auto r = h.profiler->StartRequest("m");
    r.Cancel("client disconnected");
    CHECK_EQ(r.Status(), std::string("cancelled"));
  }
  const auto* cancelled = h.Find("inference.request");
  CHECK_EQ(Attr<std::string>(cancelled, "inference.status"), std::string("cancelled"));
  CHECK(cancelled->GetStatus() != opentelemetry::trace::StatusCode::kError);
  h.finished.clear();
  {
    auto r = h.profiler->StartRequest("m");
    r.Fail("upstream_timeout");
  }
  const auto* failed = h.Find("inference.request");
  CHECK_EQ(Attr<std::string>(failed, "error.type"), std::string("upstream_timeout"));
  CHECK_EQ(h.MetricTotal("inference.requests", {{"inference.status", "cancelled"}}), 1.0);
}

TEST_CASE(span_move_semantics) {
  test::Harness h;
  RequestScope outer;
  CHECK(!outer.active());
  {
    auto inner = h.profiler->StartRequest("m");
    outer = std::move(inner);
    CHECK(!inner.active());  // NOLINT(bugprone-use-after-move): verifying moved-from state
  }
  CHECK_EQ(h.Count("inference.request"), 0);  // moved-from destructor must not end the span
  outer.End();
  CHECK_EQ(h.Count("inference.request"), 1);
}

TEST_CASE(span_traceparent_propagation) {
  test::Harness h;
  RequestOptions opts;
  opts.model = "m";
  opts.traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
  std::string outgoing;
  {
    auto r = h.profiler->StartRequest(opts);
    CHECK_EQ(r.TraceId(), std::string("4bf92f3577b34da6a3ce929d0e0e4736"));
    outgoing = r.TraceParent();
  }
  const auto* root = h.Find("inference.request");
  CHECK_EQ(Hex(root->GetTraceId()), std::string("4bf92f3577b34da6a3ce929d0e0e4736"));
  CHECK_EQ(Hex(root->GetParentSpanId()), std::string("00f067aa0ba902b7"));
  CHECK_EQ(outgoing.substr(0, 36), std::string("00-4bf92f3577b34da6a3ce929d0e0e4736-"));
  CHECK_EQ(outgoing.substr(36, 16), Hex(root->GetSpanId()));

  opts.traceparent = "garbage";
  auto fresh = h.profiler->StartRequest(opts);  // invalid header starts a new trace
  CHECK(fresh.active());
  CHECK(fresh.TraceId() != std::string("4bf92f3577b34da6a3ce929d0e0e4736"));
}

TEST_CASE(span_disabled_noop) {
  auto cfg = test::Harness::DefaultConfig();
  cfg.enabled = false;
  test::Harness h(cfg);
  {
    auto r = h.profiler->StartRequest("m");
    CHECK(!r.active());
    auto p = r.Prefill();
    CHECK(!p.active());
    r.RecordToken();
    r.RecordGeneration(10);
    CHECK(r.TraceId().empty());
  }
  CHECK(h.Spans().empty());
  CHECK(!h.profiler->enabled());
  CHECK(h.profiler->Shutdown());
}

TEST_CASE(span_invalid_phase_name) {
  auto cfg = test::Harness::DefaultConfig();
  cfg.strict = false;
  test::Harness h(cfg);
  {
    auto r = h.profiler->StartRequest("m");
    auto bad = r.Phase("Not Valid");
    CHECK(!bad.active());
    auto empty_model = h.profiler->StartRequest("");
    CHECK(!empty_model.active());
  }
  CHECK_EQ(h.profiler->instrumentation_errors(), uint64_t{2});
  CHECK_EQ(h.Count("inference.request"), 1);
}

TEST_CASE(span_strict_mode_throws) {
  test::Harness h;  // strict by default in tests
  auto r = h.profiler->StartRequest("m");
  CHECK_THROWS(r.Phase("Bad-Name"), std::invalid_argument);
  CHECK_THROWS(h.profiler->StartRequest(""), std::invalid_argument);
}

TEST_CASE(span_cache_operations) {
  test::Harness h;
  {
    auto r = h.profiler->StartRequest("m");
    {
      auto lookup = r.CacheLookup();
      lookup.SetCacheHit(false);
    }
    { auto insert = r.CacheInsert("kv", 1); }
    { auto evict = r.CacheEvict("kv", 3); }
  }
  {
    auto r = h.profiler->StartRequest("m");
    auto lookup = r.CacheLookup();
    lookup.SetCacheHit(true);
  }
  CHECK_EQ(h.Count("inference.cache_lookup"), 2);
  CHECK_EQ(h.Count("inference.cache_insert"), 1);
  CHECK_EQ(h.Count("inference.cache_evict"), 1);
  CHECK_EQ(h.MetricTotal("inference.cache.hits"), 1.0);
  CHECK_EQ(h.MetricTotal("inference.cache.misses"), 1.0);
  CHECK_EQ(h.MetricTotal("inference.cache.insertions"), 1.0);
  CHECK_EQ(h.MetricTotal("inference.cache.evictions"), 3.0);
  CHECK_EQ(h.MetricTotal("inference.cache.operation.duration",
                         {{"inference.cache.operation", "lookup"}}),
           2.0);
}

TEST_CASE(span_ttft_explicit) {
  test::Harness h;
  {
    auto r = h.profiler->StartRequest("m");
    r.RecordGeneration(10, 35.0);
  }
  const auto* ttft = h.Find("inference.time_to_first_token");
  const auto* root = h.Find("inference.request");
  CHECK(ttft && root);
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(ttft->GetDuration()).count();
  CHECK(us >= 34000 && us <= 36000);
  CHECK(ttft->GetStartTime() == root->GetStartTime());
}
