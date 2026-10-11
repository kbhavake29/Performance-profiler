from __future__ import annotations

import pytest

from inference_profiler import ConfigError, ProfilerConfig


def test_defaults_are_safe_local_values() -> None:
    cfg = ProfilerConfig()
    assert cfg.otlp_endpoint == "http://localhost:4318"
    assert cfg.traces_endpoint == "http://localhost:4318/v1/traces"
    assert cfg.metrics_endpoint == "http://localhost:4318/v1/metrics"
    assert cfg.enabled and cfg.sample_rate == 1.0 and not cfg.strict
    assert not cfg.set_global_providers


def test_env_overrides_defaults() -> None:
    env = {
        "OTEL_SERVICE_NAME": "svc",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318/",
        "INFERENCE_PROFILER_ENABLED": "false",
        "INFERENCE_PROFILER_SAMPLE_RATE": "0.25",
        "INFERENCE_PROFILER_EXPORT_INTERVAL": "2000",
        "INFERENCE_PROFILER_MEMORY_INTERVAL": "0",
        "OTEL_RESOURCE_ATTRIBUTES": "team=infra, region = local",
    }
    cfg = ProfilerConfig.from_env(env)
    assert cfg.service_name == "svc"
    assert cfg.traces_endpoint == "http://collector:4318/v1/traces"
    assert cfg.enabled is False
    assert cfg.sample_rate == 0.25
    assert cfg.export_interval_ms == 2000
    assert cfg.memory_interval_ms == 0
    assert dict(cfg.resource_attributes) == {"team": "infra", "region": "local"}


def test_explicit_overrides_beat_env() -> None:
    cfg = ProfilerConfig.from_env({"OTEL_SERVICE_NAME": "from-env"}, service_name="explicit")
    assert cfg.service_name == "explicit"


def test_service_name_from_resource_attributes_when_not_set() -> None:
    cfg = ProfilerConfig.from_env({"OTEL_RESOURCE_ATTRIBUTES": "service.name=res-svc"})
    assert cfg.service_name == "res-svc"
    assert "service.name" not in cfg.resource_attributes


def test_otel_sdk_disabled_disables_profiler() -> None:
    assert ProfilerConfig.from_env({"OTEL_SDK_DISABLED": "true"}).enabled is False


def test_empty_env_values_are_ignored() -> None:
    assert ProfilerConfig.from_env({"OTEL_SERVICE_NAME": "  "}).service_name == "inference-service"


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"INFERENCE_PROFILER_SAMPLE_RATE": "1.5"}, "sample_rate"),
        ({"INFERENCE_PROFILER_SAMPLE_RATE": "abc"}, "INFERENCE_PROFILER_SAMPLE_RATE"),
        ({"INFERENCE_PROFILER_ENABLED": "maybe"}, "INFERENCE_PROFILER_ENABLED"),
        ({"INFERENCE_PROFILER_EXPORT_INTERVAL": "10"}, "export_interval_ms"),
        ({"INFERENCE_PROFILER_EXPORT_INTERVAL": "nan"}, "INFERENCE_PROFILER_EXPORT_INTERVAL"),
        ({"INFERENCE_PROFILER_MEMORY_INTERVAL": "10"}, "memory_interval_ms"),
        ({"OTEL_EXPORTER_OTLP_ENDPOINT": "localhost:4318"}, "http"),
        ({"OTEL_EXPORTER_OTLP_ENDPOINT": "ftp://x"}, "http"),
        ({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://user:pw@host:4318"}, "credentials"),
        ({"INFERENCE_PROFILER_EXPORTER": "zipkin"}, "INFERENCE_PROFILER_EXPORTER"),
    ],
)
def test_invalid_env_rejected(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        ProfilerConfig.from_env(env)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"service_name": ""},
        {"service_name": "a\nb"},
        {"service_name": "x" * 200},
        {"sample_rate": -0.1},
        {"sample_rate": True},
        {"export_timeout_ms": 5},
        {"max_attribute_values": 0},
        {"exporter": "kafka"},
        {"resource_attributes": {"k": "bad\x00value"}},
    ],
)
def test_invalid_constructor_values_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        ProfilerConfig(**kwargs)  # type: ignore[arg-type]


def test_unknown_override_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown"):
        ProfilerConfig.from_env({}, colour="blue")


def test_config_is_immutable() -> None:
    cfg = ProfilerConfig()
    with pytest.raises(AttributeError):
        cfg.sample_rate = 0.5  # type: ignore[misc]
    assert cfg.with_overrides(sample_rate=0.5).sample_rate == 0.5
