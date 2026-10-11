"""Inference Performance Profiler: OpenTelemetry instrumentation and analysis for LLM inference."""

from inference_profiler.config import ConfigError, ProfilerConfig
from inference_profiler.context import current_request, extract_context, inject_context
from inference_profiler.instrumentation import (
    InferenceRequest,
    PhaseScope,
    Profiler,
    RequestCancelled,
    RequestHandle,
)
from inference_profiler.version import __version__

__all__ = [
    "ConfigError",
    "InferenceRequest",
    "PhaseScope",
    "Profiler",
    "ProfilerConfig",
    "RequestCancelled",
    "RequestHandle",
    "__version__",
    "current_request",
    "extract_context",
    "inject_context",
]
