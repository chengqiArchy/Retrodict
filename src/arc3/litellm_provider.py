"""LiteLLM transport for ThinHarness' Responses-model state machine."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

from thinharness import (
    ModelSettings,
    OpenAIProvider,
    OpenRouterModel,
    OpenRouterProvider,
    Provider,
    parse_model_ref,
)
from thinharness.providers import ProviderError

_instrumentation_lock = threading.Lock()
_instrumentation_enabled = False


def configure_litellm_instrumentation() -> bool:
    """Enable Logfire's LiteLLM inference spans once when a token is configured."""
    global _instrumentation_enabled

    token = os.getenv("LOGFIRE_TOKEN", "").strip()
    if not token:
        return False
    with _instrumentation_lock:
        if _instrumentation_enabled:
            return True
        import logfire
        from openinference.instrumentation import TraceConfig

        logfire.configure(
            token=token,
            service_name=os.getenv("LOGFIRE_SERVICE_NAME", "retrodict-litellm"),
            environment=os.getenv("LOGFIRE_ENVIRONMENT", "development"),
            send_to_logfire="if-token-present",
            distributed_tracing=True,
            console=False,
        )
        logfire.instrument_litellm(
            config=TraceConfig(
                hide_input_messages=True,
                hide_output_messages=True,
            )
        )
        _instrumentation_enabled = True
    return True


def litellm_model_name(model_ref: str) -> str:
    """Translate ThinHarness' ``provider:model`` spelling to LiteLLM's provider path."""
    provider, model = parse_model_ref(model_ref)
    return f"{provider}/{model}"


def _provider_setting(model_ref: str, setting: str) -> str | None:
    provider, _ = parse_model_ref(model_ref)
    names = [f"LITELLM_{setting}", f"{provider.upper()}_{setting}"]
    if provider == "openai" and setting == "BASE_URL":
        names.append("OPENAI_API_BASE")
    if setting == "BASE_URL":
        names.append("LLM_MAIN_API_BASE")
    if provider == "openai" and setting == "API_KEY":
        names.append("OPENROUTER_API_KEY")
    return next((value for name in names if (value := os.getenv(name, "").strip())), None)


def _default_base_url(provider_name: str, api_key: str | None) -> str:
    """Match Duck's local CLIProxy default for shared non-native provider keys."""
    if api_key and not api_key.startswith("sk-or-v1-") and provider_name in {"openai", "openrouter"}:
        return "http://localhost:8317/v1"
    if provider_name == "openrouter":
        return "https://openrouter.ai/api/v1"
    return ""


class LiteLLMResponsesProvider(OpenAIProvider):
    """Send OpenAI Responses-shaped payloads through LiteLLM's Python SDK."""

    name = "LiteLLM"

    def __init__(
        self,
        model_ref: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: int = 120,
    ) -> None:
        provider_name, _ = parse_model_ref(model_ref)
        resolved_api_key = api_key or _provider_setting(model_ref, "API_KEY")
        default_base_url = _default_base_url(provider_name, resolved_api_key)
        Provider.__init__(
            self,
            api_key=resolved_api_key,
            base_url=api_base or _provider_setting(model_ref, "BASE_URL") or default_base_url,
            timeout=timeout,
        )
        self.provider_name = provider_name
        self.model_name = litellm_model_name(model_ref)

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call LiteLLM and normalize its typed response for ThinHarness."""
        from litellm import responses

        request = {**payload, "model": self.model_name, "timeout": self.timeout}
        if self.api_key:
            request["api_key"] = self.api_key
            if self.provider_name == "openrouter":
                request["extra_headers"] = {"Authorization": f"Bearer {self.api_key}"}
        if self.base_url:
            request["api_base"] = self.base_url
        try:
            # LiteLLM's aresponses currently dispatches through a worker thread
            # itself, then leaves its async success callback pending at loop
            # shutdown. Run the synchronous API in our worker instead so the
            # event loop remains non-blocking and instrumentation fully flushes.
            response = await asyncio.to_thread(responses, **request)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            raise ProviderError(f"LiteLLM request failed: {exc}", status_code=status_code) from exc
        if isinstance(response, dict):
            return response
        model_dump = getattr(response, "model_dump", None)
        if not callable(model_dump):
            raise ProviderError(f"LiteLLM returned unsupported response type {type(response).__name__}")
        data = model_dump(mode="json")
        if not isinstance(data, dict):
            raise ProviderError(f"LiteLLM returned unsupported response payload {type(data).__name__}")
        return data


class LiteLLMChatCompletionsProvider(OpenRouterProvider):
    """Send full-history Chat Completions payloads through LiteLLM."""

    name = "LiteLLM"

    def __init__(
        self,
        model_ref: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: int = 120,
    ) -> None:
        provider_name, _ = parse_model_ref(model_ref)
        resolved_api_key = api_key or _provider_setting(model_ref, "API_KEY")
        default_base_url = _default_base_url(provider_name, resolved_api_key)
        Provider.__init__(
            self,
            api_key=resolved_api_key,
            base_url=api_base or _provider_setting(model_ref, "BASE_URL") or default_base_url,
            timeout=timeout,
        )
        self.provider_name = provider_name
        self.model_name = litellm_model_name(model_ref)

    async def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call LiteLLM completion while preserving ThinHarness' full message history."""
        from litellm import completion

        request = {
            **payload,
            "model": self.model_name,
            "timeout": self.timeout,
            # Some model metadata marks GPT-5 variants as Responses-only, which
            # makes LiteLLM silently bridge completion() to responses(). This
            # proxy supports Chat Completions, so keep the requested transport
            # and the full message history intact.
            "_skip_responses_api_bridge": True,
        }
        reasoning = request.pop("reasoning", None)
        if isinstance(reasoning, dict) and reasoning.get("effort"):
            request["reasoning_effort"] = reasoning["effort"]
        if self.api_key:
            request["api_key"] = self.api_key
        if self.base_url:
            request["api_base"] = self.base_url
        try:
            response = await asyncio.to_thread(completion, **request)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            raise ProviderError(f"LiteLLM request failed: {exc}", status_code=status_code) from exc
        if isinstance(response, dict):
            return response
        model_dump = getattr(response, "model_dump", None)
        if not callable(model_dump):
            raise ProviderError(f"LiteLLM returned unsupported response type {type(response).__name__}")
        data = model_dump(mode="json")
        if not isinstance(data, dict):
            raise ProviderError(f"LiteLLM returned unsupported response payload {type(data).__name__}")
        return data


def build_litellm_model(
    model_ref: str,
    *,
    timeout: int,
    max_tokens: int | None,
    effort: str | None,
) -> OpenRouterModel:
    """Build a full-history Chat Completions model while LiteLLM owns transport."""
    _, model = parse_model_ref(model_ref)
    provider = LiteLLMChatCompletionsProvider(model_ref, timeout=timeout)
    settings = ModelSettings(max_tokens=max_tokens, effort=effort)
    return OpenRouterModel(model, provider=provider, settings=settings)
