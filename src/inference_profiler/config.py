"""Profiler configuration.

Precedence (highest first):

1. Keyword overrides passed to :meth:`ProfilerConfig.from_env`.
2. Environment variables (``OTEL_*`` and ``INFERENCE_PROFILER_*``).
3. The defaults defined on :class:`ProfilerConfig`.

Constructing ``ProfilerConfig(...)`` directly uses only the values you pass plus
the defaults; it never reads the environment. ``Profiler()`` without a config
calls :meth:`ProfilerConfig.from_env`.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal
from urllib.parse import urlparse

ExporterKind = Literal["otlp", "console", "none"]

DEFAULT_SERVICE_NAME = "inference-service"
DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"

_MAX_NAME_LEN = 128
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class ConfigError(ValueError):
    """Raised when a configuration value is missing, malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class ProfilerConfig:
    """Immutable, validated profiler configuration.

    Durations are expressed in milliseconds to match the OpenTelemetry
    ``OTEL_METRIC_EXPORT_INTERVAL`` convention.
    """

    service_name: str = DEFAULT_SERVICE_NAME
    #: Base OTLP/HTTP endpoint. ``/v1/traces`` and ``/v1/metrics`` are appended.
    otlp_endpoint: str = DEFAULT_OTLP_ENDPOINT
    enabled: bool = True
    #: Fraction of request traces to sample (parent-based). Metrics are never sampled.
    sample_rate: float = 1.0
    export_interval_ms: int = 5_000
    export_timeout_ms: int = 10_000
    #: Interval of the background RSS sampler. ``0`` disables it.
    memory_interval_ms: int = 5_000
    shutdown_timeout_ms: int = 5_000
    exporter: ExporterKind = "otlp"
    runtime_version: str = "unknown"
    deployment_environment: str = "local"
    #: Raise instrumentation errors instead of logging them (for tests/development).
    strict: bool = False
    #: Emit one ``inference.token`` span per generated token for sampled requests.
    token_spans: bool = False
    token_span_sample_rate: float = 0.01
    max_token_spans_per_request: int = 256
    #: Maximum distinct values per metric attribute before values collapse to ``_other``.
    max_attribute_values: int = 64
    #: Maximum spans held in memory waiting for export (BatchSpanProcessor queue).
    max_span_queue_size: int = 2_048
    #: Register the providers as OpenTelemetry globals. Off by default to avoid
    #: clobbering providers configured by the host application.
    set_global_providers: bool = False
    resource_attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        _check_name("service_name", self.service_name)
        _check_name("runtime_version", self.runtime_version)
        _check_name("deployment_environment", self.deployment_environment)
        _check_endpoint(self.otlp_endpoint)
        _check_ratio("sample_rate", self.sample_rate)
        _check_ratio("token_span_sample_rate", self.token_span_sample_rate)
        _check_int_range("export_interval_ms", self.export_interval_ms, 100, 3_600_000)
        _check_int_range("export_timeout_ms", self.export_timeout_ms, 100, 600_000)
        _check_int_range("shutdown_timeout_ms", self.shutdown_timeout_ms, 0, 600_000)
        _check_int_range("memory_interval_ms", self.memory_interval_ms, 0, 3_600_000)
        if 0 < self.memory_interval_ms < 50:
            raise ConfigError("memory_interval_ms must be 0 (disabled) or >= 50")
        _check_int_range(
            "max_token_spans_per_request", self.max_token_spans_per_request, 0, 100_000
        )
        _check_int_range("max_attribute_values", self.max_attribute_values, 1, 10_000)
        _check_int_range("max_span_queue_size", self.max_span_queue_size, 1, 1_000_000)
        if self.exporter not in ("otlp", "console", "none"):
            raise ConfigError(
                f"exporter must be one of otlp, console, none (got {self.exporter!r})"
            )
        for key, value in self.resource_attributes.items():
            _check_name("resource attribute key", key)
            if not isinstance(value, str) or _CONTROL_CHARS.search(value):
                raise ConfigError(f"resource attribute {key!r} must be a printable string")

    # ------------------------------------------------------------------ helpers
    @property
    def traces_endpoint(self) -> str:
        return self.otlp_endpoint.rstrip("/") + "/v1/traces"

    @property
    def metrics_endpoint(self) -> str:
        return self.otlp_endpoint.rstrip("/") + "/v1/metrics"

    def with_overrides(self, **overrides: Any) -> ProfilerConfig:
        return replace(self, **overrides)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> ProfilerConfig:
        """Build a config from defaults, then environment variables, then ``overrides``."""
        env = os.environ if env is None else env
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ConfigError(f"unknown configuration field(s): {sorted(unknown)}")

        values: dict[str, Any] = {}
        for var, (name, parser) in _ENV_VARS.items():
            raw = env.get(var)
            if raw is None or raw.strip() == "":
                continue
            try:
                values[name] = parser(raw.strip())
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"invalid value for {var}: {raw!r} ({exc})") from exc

        # OTEL_RESOURCE_ATTRIBUTES may also carry service.name; explicit vars win.
        res_attrs = _parse_resource_attributes(env.get("OTEL_RESOURCE_ATTRIBUTES", ""))
        if "service_name" not in values and "service.name" in res_attrs:
            values["service_name"] = res_attrs["service.name"]
        res_attrs.pop("service.name", None)
        if res_attrs:
            values["resource_attributes"] = res_attrs

        if env.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
            values["enabled"] = False

        values.update(overrides)
        return cls(**values)


# ---------------------------------------------------------------------- parsing


def parse_bool(raw: str) -> bool:
    lowered = raw.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError("expected a boolean (true/false/1/0)")


def _parse_ms(raw: str) -> int:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("expected a finite number of milliseconds")
    return int(value)


def _parse_exporter(raw: str) -> str:
    lowered = raw.lower()
    if lowered not in ("otlp", "console", "none"):
        raise ValueError("expected otlp, console or none")
    return lowered


def _parse_resource_attributes(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        if key.strip():
            attrs[key.strip()] = value.strip()
    return attrs


_ENV_VARS: dict[str, tuple[str, Any]] = {
    "OTEL_SERVICE_NAME": ("service_name", str),
    "OTEL_EXPORTER_OTLP_ENDPOINT": ("otlp_endpoint", str),
    "INFERENCE_PROFILER_ENABLED": ("enabled", parse_bool),
    "INFERENCE_PROFILER_SAMPLE_RATE": ("sample_rate", float),
    "INFERENCE_PROFILER_EXPORT_INTERVAL": ("export_interval_ms", _parse_ms),
    "INFERENCE_PROFILER_EXPORT_TIMEOUT": ("export_timeout_ms", _parse_ms),
    "INFERENCE_PROFILER_MEMORY_INTERVAL": ("memory_interval_ms", _parse_ms),
    "INFERENCE_PROFILER_SHUTDOWN_TIMEOUT": ("shutdown_timeout_ms", _parse_ms),
    "INFERENCE_PROFILER_EXPORTER": ("exporter", _parse_exporter),
    "INFERENCE_PROFILER_RUNTIME_VERSION": ("runtime_version", str),
    "INFERENCE_PROFILER_ENVIRONMENT": ("deployment_environment", str),
    "INFERENCE_PROFILER_STRICT": ("strict", parse_bool),
    "INFERENCE_PROFILER_TOKEN_SPANS": ("token_spans", parse_bool),
    "INFERENCE_PROFILER_TOKEN_SPAN_SAMPLE_RATE": ("token_span_sample_rate", float),
}

ENV_VARIABLES: tuple[str, ...] = (*_ENV_VARS, "OTEL_RESOURCE_ATTRIBUTES", "OTEL_SDK_DISABLED")


def _check_name(label: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    if len(value) > _MAX_NAME_LEN:
        raise ConfigError(f"{label} must be at most {_MAX_NAME_LEN} characters")
    if _CONTROL_CHARS.search(value):
        raise ConfigError(f"{label} must not contain control characters")


def _check_endpoint(endpoint: object) -> None:
    if not isinstance(endpoint, str):
        raise ConfigError("otlp_endpoint must be a string")
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigError(
            f"otlp_endpoint must be an http(s) URL such as {DEFAULT_OTLP_ENDPOINT} (got {endpoint!r})"
        )
    if parsed.username or parsed.password:
        raise ConfigError(
            "otlp_endpoint must not embed credentials; use OTEL_EXPORTER_OTLP_HEADERS"
        )
    if parsed.query or parsed.fragment:
        raise ConfigError("otlp_endpoint must not contain a query string or fragment")


def _check_ratio(label: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number between 0 and 1")
    if not 0.0 <= float(value) <= 1.0:
        raise ConfigError(f"{label} must be between 0 and 1 (got {value})")


def _check_int_range(label: str, value: object, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    if not low <= value <= high:
        raise ConfigError(f"{label} must be between {low} and {high} (got {value})")
