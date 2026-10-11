// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <string>

#include "test_harness.hpp"

using inference_profiler::ConfigError;
using inference_profiler::ProfilerConfig;

namespace {
ProfilerConfig::EnvLookup Env(std::map<std::string, std::string> vars) {
  auto shared = std::make_shared<std::map<std::string, std::string>>(std::move(vars));
  return [shared](const char* name) -> const char* {
    auto it = shared->find(name);
    return it == shared->end() ? nullptr : it->second.c_str();
  };
}
}  // namespace

TEST_CASE(config_defaults) {
  ProfilerConfig cfg;
  CHECK(cfg.Validate().empty());
  CHECK_EQ(cfg.otlp_endpoint, std::string("http://localhost:4318"));
  CHECK_EQ(cfg.TracesUrl(), std::string("http://localhost:4318/v1/traces"));
  CHECK_EQ(cfg.MetricsUrl(), std::string("http://localhost:4318/v1/metrics"));
  CHECK(cfg.enabled);
  CHECK(!cfg.set_global_providers);
}

TEST_CASE(config_from_env) {
  auto cfg = ProfilerConfig::FromEnv(Env({
      {"OTEL_SERVICE_NAME", "svc"},
      {"OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/"},
      {"INFERENCE_PROFILER_ENABLED", "false"},
      {"INFERENCE_PROFILER_SAMPLE_RATE", "0.5"},
      {"INFERENCE_PROFILER_EXPORT_INTERVAL", "2000"},
      {"INFERENCE_PROFILER_MEMORY_INTERVAL", "0"},
      {"OTEL_RESOURCE_ATTRIBUTES", "service.name=ignored,team=infra"},
  }));
  CHECK_EQ(cfg.service_name, std::string("svc"));
  CHECK_EQ(cfg.TracesUrl(), std::string("http://collector:4318/v1/traces"));
  CHECK(!cfg.enabled);
  CHECK_EQ(cfg.sample_rate, 0.5);
  CHECK_EQ(cfg.export_interval.count(), 2000);
  CHECK(!cfg.report_process_memory);
  CHECK_EQ(cfg.resource_attributes.at("team"), std::string("infra"));

  auto from_resource = ProfilerConfig::FromEnv(Env({{"OTEL_RESOURCE_ATTRIBUTES", "service.name=r"}}));
  CHECK_EQ(from_resource.service_name, std::string("r"));
  auto disabled = ProfilerConfig::FromEnv(Env({{"OTEL_SDK_DISABLED", "true"}}));
  CHECK(!disabled.enabled);
}

TEST_CASE(config_env_errors) {
  CHECK_THROWS(ProfilerConfig::FromEnv(Env({{"INFERENCE_PROFILER_SAMPLE_RATE", "abc"}})),
               ConfigError);
  CHECK_THROWS(ProfilerConfig::FromEnv(Env({{"INFERENCE_PROFILER_SAMPLE_RATE", "2"}})),
               ConfigError);
  CHECK_THROWS(ProfilerConfig::FromEnv(Env({{"INFERENCE_PROFILER_ENABLED", "maybe"}})),
               ConfigError);
  CHECK_THROWS(ProfilerConfig::FromEnv(Env({{"INFERENCE_PROFILER_EXPORT_INTERVAL", "1e400"}})),
               ConfigError);
  CHECK_THROWS(ProfilerConfig::FromEnv(Env({{"OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4318"}})),
               ConfigError);
  try {
    ProfilerConfig::FromEnv(Env({{"INFERENCE_PROFILER_SAMPLE_RATE", "abc"}}));
  } catch (const ConfigError& e) {
    CHECK(std::string(e.what()).find("INFERENCE_PROFILER_SAMPLE_RATE") != std::string::npos);
  }
}

TEST_CASE(config_validation) {
  auto invalid = [](auto mutate) {
    ProfilerConfig cfg;
    mutate(cfg);
    return !cfg.Validate().empty();
  };
  CHECK(invalid([](ProfilerConfig& c) { c.service_name = ""; }));
  CHECK(invalid([](ProfilerConfig& c) { c.service_name = "bad\nname"; }));
  CHECK(invalid([](ProfilerConfig& c) { c.sample_rate = 1.5; }));
  CHECK(invalid([](ProfilerConfig& c) { c.sample_rate = -0.1; }));
  CHECK(invalid([](ProfilerConfig& c) { c.otlp_endpoint = "ftp://host"; }));
  CHECK(invalid([](ProfilerConfig& c) { c.otlp_endpoint = "http://user:pw@host:4318"; }));
  CHECK(invalid([](ProfilerConfig& c) { c.otlp_endpoint = "http://"; }));
  CHECK(invalid([](ProfilerConfig& c) { c.export_interval = std::chrono::milliseconds(5); }));
  CHECK(invalid([](ProfilerConfig& c) { c.max_attribute_values = 0; }));
  CHECK(!invalid([](ProfilerConfig& c) { c.otlp_endpoint = "https://otel.example:4318/base"; }));
  ProfilerConfig bad;
  bad.sample_rate = 3;
  CHECK_THROWS(inference_profiler::Profiler::Create(bad), ConfigError);
}
