// Copyright The Inference Performance Profiler Authors
// SPDX-License-Identifier: Apache-2.0
//
// Exporter injection points. Including this header pulls in OpenTelemetry SDK
// headers; regular application code only needs profiler.hpp.
#pragma once

#include <memory>

#include "inference_profiler/profiler.hpp"
#include "opentelemetry/sdk/metrics/push_metric_exporter.h"
#include "opentelemetry/sdk/trace/exporter.h"

namespace inference_profiler {

struct ExporterOverrides {
  /// Replaces the OTLP span exporter. Driven by a synchronous SimpleSpanProcessor
  /// unless `batch_spans` is true.
  std::unique_ptr<opentelemetry::sdk::trace::SpanExporter> span_exporter;
  /// Replaces the OTLP metric exporter (driven by a PeriodicExportingMetricReader).
  std::unique_ptr<opentelemetry::sdk::metrics::PushMetricExporter> metric_exporter;
  bool batch_spans = false;
};

}  // namespace inference_profiler
