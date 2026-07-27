from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from thinharness import RequestConstants, ToolOutput
from thinharness.providers import ProviderError

from arc3 import litellm_provider
from arc3.litellm_provider import (
    LiteLLMChatCompletionsProvider,
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


@pytest.mark.asyncio
async def test_provider_routes_chat_completion_through_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_completion(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(
            model_dump=lambda **_: {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
            }
        )

    monkeypatch.setattr("litellm.completion", fake_completion)
    provider = LiteLLMChatCompletionsProvider(
        "openai:gpt-5.5",
        api_key="test-key",
        api_base="https://gateway.example/v1",
        timeout=42,
    )
    messages = [{"role": "user", "content": "hello"}]

    result = await provider.create_chat_completion(
        {"model": "ignored", "messages": messages, "reasoning": {"effort": "high"}}
    )

    assert result["choices"][0]["message"]["content"] == "done"
    assert captured == {
        "model": "openai/gpt-5.5",
        "messages": messages,
        "reasoning_effort": "high",
        "timeout": 42,
        "api_key": "test-key",
        "api_base": "https://gateway.example/v1",
    }


@pytest.mark.asyncio
async def test_chat_completion_tool_loop_sends_full_message_history(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict[str, object]] = []

    def fake_completion(**kwargs: object) -> object:
        requests.append(copy.deepcopy(kwargs))
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "python", "arguments": '{"code":"print(1)"}'},
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "done"}
        return {"choices": [{"message": message}]}

    monkeypatch.setattr("litellm.completion", fake_completion)
    model = build_litellm_model("openai:gpt-5.5", timeout=30, max_tokens=512, effort="high")
    session = model.new_session()
    constants = RequestConstants(
        instructions="Use tools.",
        tools=[
            {
                "type": "function",
                "name": "python",
                "description": "Run Python.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            }
        ],
    )

    first = await session.start("inspect", constants)
    second = await session.continue_with_tools([ToolOutput(first.tool_calls[0].id, "1")], constants)

    assert second.text == "done"
    assert requests[1]["messages"] == [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "python", "arguments": '{"code":"print(1)"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "1"},
    ]
    assert "previous_response_id" not in requests[1]


def test_openai_model_uses_duck_shared_proxy_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MAIN_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-proxy-test-key")

    provider = LiteLLMChatCompletionsProvider("openai:gpt-5.5")

    assert provider.api_key == "shared-proxy-test-key"
    assert provider.base_url == "http://localhost:8317/v1"


def test_duck_api_base_override_wins_over_local_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-proxy-test-key")
    monkeypatch.setenv("LLM_MAIN_API_BASE", "https://proxy.example/v1")

    provider = LiteLLMChatCompletionsProvider("openai:gpt-5.5")

    assert provider.base_url == "https://proxy.example/v1"


def test_build_model_uses_full_history_chat_completions() -> None:
    model = build_litellm_model("openai:gpt-5.5", timeout=30, max_tokens=8192, effort="high")

    assert model.model == "gpt-5.5"
    assert isinstance(model.provider, LiteLLMChatCompletionsProvider)
    assert model.provider.model_name == "openai/gpt-5.5"
    assert model.settings.max_tokens == 8192
    assert model.settings.effort == "high"


@pytest.mark.asyncio
async def test_provider_maps_litellm_errors_to_harness_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLiteLLMError(Exception):
        status_code = 429

    def fail(**_: object) -> object:
        raise FakeLiteLLMError("rate limited")

    monkeypatch.setattr("litellm.completion", fail)
    provider = LiteLLMChatCompletionsProvider("openai:gpt-5.5")

    with pytest.raises(ProviderError, match="LiteLLM request failed: rate limited") as caught:
        await provider.create_chat_completion({"messages": [{"role": "user", "content": "hello"}]})

    assert caught.value.status_code == 429
