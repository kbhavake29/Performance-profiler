// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0
//
// Native demo: a fake inference engine instrumented with the C++ SDK.
//
//   ./build/cpp_inference_demo                 # 40 requests, endpoint from env / localhost:4318
//   ./build/cpp_inference_demo --requests 100 --endpoint http://localhost:4318
//
// The "engine" sleeps for durations proportional to token counts. Replace the
// sleeps with calls into a real runtime.

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "inference_profiler/profiler.hpp"

namespace ip = inference_profiler;
using std::chrono::microseconds;

namespace {

struct Args {
  int requests = 40;
  int workers = 4;
  std::string endpoint;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
      return argv[++i];
    };
    if (a == "--requests") {
      args.requests = std::stoi(next());
    } else if (a == "--workers") {
      args.workers = std::stoi(next());
    } else if (a == "--endpoint") {
      args.endpoint = next();
    } else if (a == "--help" || a == "-h") {
      std::cout << "usage: cpp_inference_demo [--requests N] [--workers N] [--endpoint URL]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument " + a);
    }
  }
  if (args.requests < 1 || args.workers < 1) throw std::invalid_argument("counts must be >= 1");
  return args;
}

class EngineFault : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// One request through queue -> cache lookup -> prefill -> generation.
void HandleRequest(ip::Profiler& profiler, std::mt19937& rng, int index) {
  std::uniform_int_distribution<int> prompt_dist(64, 1024);
  std::uniform_int_distribution<int> gen_dist(16, 64);
  std::uniform_int_distribution<int> prefix_dist(0, 24);
  std::uniform_int_distribution<int> step_us(4000, 8000);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  const int prompt_tokens = prompt_dist(rng);
  const int max_tokens = gen_dist(rng);

  ip::RequestOptions opts;
  opts.model = "demo-model";
  opts.prompt_tokens = prompt_tokens;
  opts.runtime_version = "native-demo-1.0";
  opts.request_id = "cpp-" + std::to_string(index);
  if (index % 5 == 0) {
    // Simulates an upstream service that already started a trace.
    opts.traceparent = "00-4bf92f3577b34da6a3ce929d0e0e47" + std::to_string(10 + index % 90) +
                       "-00f067aa0ba902b7-01";
  }
  auto request = profiler.StartRequest(opts);

  {
    auto queue = request.Queue();
    std::this_thread::sleep_for(microseconds(static_cast<int>(unit(rng) * 3000)));
  }
  bool hit = false;
  {
    auto lookup = request.CacheLookup();
    std::this_thread::sleep_for(microseconds(200));
    hit = prefix_dist(rng) < 12;
    lookup.SetCacheHit(hit);
  }
  {
    auto prefill = request.Prefill();
    const double ms = (4.0 + prompt_tokens * 0.03) * (hit ? 0.3 : 1.0);
    std::this_thread::sleep_for(microseconds(static_cast<int>(ms * 1000)));
  }
  {
    auto generation = request.Generation();
    for (int t = 0; t < max_tokens; ++t) {
      std::this_thread::sleep_for(microseconds(step_us(rng)));
      request.RecordToken();
      if (index % 23 == 22 && t == 3) throw EngineFault("simulated decode fault");
    }
  }
  if (!hit) {
    auto insert = request.CacheInsert();
    std::this_thread::sleep_for(microseconds(200));
  }
  request.RecordGeneration(max_tokens);
  // `request` ends here (RAII); an exception above ends it with status=error.
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  ip::ProfilerConfig config;
  try {
    args = ParseArgs(argc, argv);
    config = ip::ProfilerConfig::FromEnv();
    if (std::getenv("OTEL_SERVICE_NAME") == nullptr) config.service_name = "cpp-inference-demo";
    config.runtime_version = "native-demo-1.0";
    config.export_interval = std::chrono::milliseconds(1000);
    if (!args.endpoint.empty()) config.otlp_endpoint = args.endpoint;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }

  auto profiler = ip::Profiler::Create(config);
  std::cout << "cpp_inference_demo: " << args.requests << " requests on " << args.workers
            << " threads, exporting to " << config.otlp_endpoint << " as '" << config.service_name
            << "'\n";

  std::atomic<int> next{0};
  std::atomic<int> ok{0};
  std::atomic<int> failed{0};
  std::vector<std::thread> workers;
  for (int w = 0; w < args.workers; ++w) {
    workers.emplace_back([&, w] {
      std::mt19937 rng(static_cast<unsigned>(1234 + w));
      for (int i = next++; i < args.requests; i = next++) {
        try {
          HandleRequest(*profiler, rng, i);
          ++ok;
        } catch (const EngineFault&) {
          ++failed;
        }
      }
    });
  }
  for (auto& t : workers) t.join();

  const bool flushed = profiler->ForceFlush();
  const bool clean = profiler->Shutdown();
  std::cout << "requests: " << ok << " ok, " << failed << " failed; flushed=" << std::boolalpha
            << flushed << ", clean_shutdown=" << clean
            << ", export_failures=" << profiler->export_failures() << "\n";
  if (profiler->export_failures() > 0) {
    std::cout << "warning: telemetry export failed - is the collector running at "
              << config.otlp_endpoint << "? (the workload itself was unaffected)\n";
  }
  return 0;
}
