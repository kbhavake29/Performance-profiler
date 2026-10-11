// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "test_harness.hpp"

using inference_profiler::CardinalityLimiter;
using inference_profiler::RequestOptions;

TEST_CASE(metrics_request_counters) {
  test::Harness h;
  for (int i = 0; i < 3; ++i) {
    RequestOptions opts;
    opts.model = "demo";
    opts.request_id = "unique-" + std::to_string(i);
    opts.prompt_tokens = 10;
    auto r = h.profiler->StartRequest(opts);
    r.RecordGeneration(5);
  }
  CHECK_EQ(h.MetricTotal("inference.requests", {{"inference.status", "ok"}}), 3.0);
  CHECK_EQ(h.MetricTotal("inference.tokens", {{"gen_ai.token.type", "input"}}), 30.0);
  CHECK_EQ(h.MetricTotal("inference.tokens", {{"gen_ai.token.type", "output"}}), 15.0);
  CHECK_EQ(h.MetricTotal("inference.requests", {{"gen_ai.request.model", "demo"},
                                                {"inference.runtime.version", "test-1.0"}}),
           3.0);
  // request ids must never become metric labels
  CHECK_EQ(h.MetricTotal("inference.requests", {{"inference.request.id", "unique-0"}}), 0.0);
}

TEST_CASE(metrics_histograms) {
  test::Harness h;
  {
    auto r = h.profiler->StartRequest("demo", 64);
    { auto q = r.Queue(); }
    { auto p = r.Prefill(); }
    {
      auto g = r.Generation();
      for (int i = 0; i < 3; ++i) r.RecordToken();
    }
    { auto custom = r.Phase("detokenize"); }
    r.RecordGeneration(3);
  }
  for (const char* name : {"inference.request.duration", "inference.queue.duration",
                           "inference.prefill.duration", "inference.generation.duration",
                           "inference.time_to_first_token", "inference.generation.throughput"}) {
    if (h.MetricTotal(name) != 1.0) throw test::Failure(std::string("expected 1 sample in ") + name);
  }
  CHECK_EQ(h.MetricTotal("inference.inter_token.latency"), 2.0);
  CHECK_EQ(h.MetricTotal("inference.phase.duration", {{"inference.phase", "detokenize"}}), 1.0);
}

TEST_CASE(metrics_gauges_return_to_zero) {
  test::Harness h;
  {
    auto r = h.profiler->StartRequest("demo");
    {
      auto q = r.Queue();
      CHECK_EQ(h.MetricTotal("inference.queue.depth"), 1.0);
      CHECK_EQ(h.MetricTotal("inference.requests.active"), 0.0);
      CHECK_EQ(h.MetricTotal("inference.requests.in_flight"), 1.0);
    }
    CHECK_EQ(h.MetricTotal("inference.queue.depth"), 0.0);
    CHECK_EQ(h.MetricTotal("inference.requests.active"), 1.0);
  }
  CHECK_EQ(h.MetricTotal("inference.requests.in_flight"), 0.0);
  CHECK_EQ(h.MetricTotal("inference.requests.active"), 0.0);
  // a queue phase left open is closed by End()
  {
    auto r = h.profiler->StartRequest("demo");
    auto q = r.Queue();
    r.End();
  }
  CHECK_EQ(h.MetricTotal("inference.queue.depth"), 0.0);
  CHECK_EQ(h.MetricTotal("inference.requests.active"), 0.0);
}

TEST_CASE(metrics_cardinality_limit) {
  CardinalityLimiter limiter(2);
  CHECK_EQ(limiter.Limit("model", "a"), std::string("a"));
  CHECK_EQ(limiter.Limit("model", "b"), std::string("b"));
  CHECK_EQ(limiter.Limit("model", "c"), std::string("_other"));
  CHECK_EQ(limiter.Limit("model", "a"), std::string("a"));
  CHECK(limiter.Overflowed("model"));
  CHECK(!limiter.Overflowed("phase"));

  auto cfg = test::Harness::DefaultConfig();
  cfg.max_attribute_values = 2;
  test::Harness h(cfg);
  for (int i = 0; i < 5; ++i) h.profiler->StartRequest("model-" + std::to_string(i)).End();
  CHECK_EQ(h.MetricTotal("inference.requests", {{"gen_ai.request.model", "_other"}}), 3.0);
}

TEST_CASE(metrics_process_rss) {
  CHECK(inference_profiler::ProcessRssBytes() > 1000000);
  test::Harness h;
  CHECK(h.MetricTotal("process.memory.usage") > 1000000.0);
  CHECK_EQ(h.MetricTotal("inference_profiler.enabled"), 1.0);
}
