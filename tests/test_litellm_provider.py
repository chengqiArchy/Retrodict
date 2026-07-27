from __future__ import annotations

from types import SimpleNamespace

import pytest
from thinharness.providers import ProviderError

from arc3 import litellm_provider
from arc3.litellm_provider import (
    LiteLLMResponsesProvider,
    build_litellm_model,
    configure_litellm_instrumentation,
    litellm_model_name,
)


def test_litellm_model_name_translates_harness_provider_separator() -> None:
    assert litellm_model_name("openai:gpt-5.5") == "openai/gpt-5.5"
    assert litellm_model_name("openrouter:moonshotai/kimi-k2.6") == "openrouter/moonshotai/kimi-k2.6"


def test_logfire_litellm_instrumentation_is_enabled_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setenv("LOGFIRE_TOKEN", "logfire-test-token")
    monkeypatch.setattr(litellm_provider, "_instrumentation_enabled", False)
    monkeypatch.setattr("logfire.configure", lambda **kwargs: calls.append(("configure", kwargs)))
    monkeypatch.setattr("logfire.instrument_litellm", lambda: calls.append(("instrument", None)))

    assert configure_litellm_instrumentation() is True
    assert configure_litellm_instrumentation() is True

    assert [name for name, _ in calls] == ["configure", "instrument"]
    configure_kwargs = calls[0][1]
    assert isinstance(configure_kwargs, dict)
    assert configure_kwargs["service_name"] == "retrodict-litellm"
    assert configure_kwargs["send_to_logfire"] == "if-token-present"
    assert configure_kwargs["distributed_tracing"] is True


@pytest.mark.asyncio
async def test_provider_routes_responses_payload_through_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_responses(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(model_dump=lambda **_: {"id": "resp_1", "output": []})

    monkeypatch.setattr("litellm.responses", fake_responses)
    provider = LiteLLMResponsesProvider(
        "openai:gpt-5.5",
        api_key="test-key",
        api_base="https://gateway.example/v1",
        timeout=42,
    )

    result = await provider.create_response({"model": "ignored", "input": "hello"})

    assert result == {"id": "resp_1", "output": []}
    assert captured == {
        "model": "openai/gpt-5.5",
        "input": "hello",
        "timeout": 42,
        "api_key": "test-key",
        "api_base": "https://gateway.example/v1",
    }


@pytest.mark.asyncio
async def test_openrouter_responses_sets_endpoint_and_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_responses(**kwargs: object) -> object:
        captured.update(kwargs)
        return {"id": "resp_1", "output": []}

    monkeypatch.setattr("litellm.responses", fake_responses)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-openrouter-test-key")
    provider = LiteLLMResponsesProvider("openrouter:openai/gpt-5.5")

    await provider.create_response({"input": "hello"})

    assert captured["model"] == "openrouter/openai/gpt-5.5"
    assert captured["api_base"] == "https://openrouter.ai/api/v1"
    assert captured["extra_headers"] == {"Authorization": "Bearer sk-or-v1-openrouter-test-key"}


def test_openai_model_uses_duck_shared_proxy_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MAIN_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-proxy-test-key")

    provider = LiteLLMResponsesProvider("openai:gpt-5.5")

    assert provider.api_key == "shared-proxy-test-key"
    assert provider.base_url == "http://localhost:8317/v1"


def test_duck_api_base_override_wins_over_local_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-proxy-test-key")
    monkeypatch.setenv("LLM_MAIN_API_BASE", "https://proxy.example/v1")

    provider = LiteLLMResponsesProvider("openai:gpt-5.5")

    assert provider.base_url == "https://proxy.example/v1"


def test_build_model_preserves_thinharness_responses_semantics() -> None:
    model = build_litellm_model("openai:gpt-5.5", timeout=30, max_tokens=8192, effort="high")

    assert model.model == "gpt-5.5"
    assert isinstance(model.provider, LiteLLMResponsesProvider)
    assert model.provider.model_name == "openai/gpt-5.5"
    assert model.settings.max_tokens == 8192
    assert model.settings.effort == "high"


@pytest.mark.asyncio
async def test_provider_maps_litellm_errors_to_harness_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLiteLLMError(Exception):
        status_code = 429

    def fail(**_: object) -> object:
        raise FakeLiteLLMError("rate limited")

    monkeypatch.setattr("litellm.responses", fail)
    provider = LiteLLMResponsesProvider("openai:gpt-5.5")

    with pytest.raises(ProviderError, match="LiteLLM request failed: rate limited") as caught:
        await provider.create_response({"input": "hello"})

    assert caught.value.status_code == 429
