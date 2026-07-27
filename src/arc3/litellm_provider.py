"""LiteLLM transport for ThinHarness' Responses-model state machine."""

from __future__ import annotations

import os
from typing import Any

from thinharness import ModelSettings, OpenAIProvider, OpenAIResponsesModel, Provider, parse_model_ref
from thinharness.providers import ProviderError


def litellm_model_name(model_ref: str) -> str:
    """Translate ThinHarness' ``provider:model`` spelling to LiteLLM's provider path."""
    provider, model = parse_model_ref(model_ref)
    return f"{provider}/{model}"


def _provider_setting(model_ref: str, setting: str) -> str | None:
    provider, _ = parse_model_ref(model_ref)
    names = [f"LITELLM_{setting}", f"{provider.upper()}_{setting}"]
    if provider == "openai" and setting == "BASE_URL":
        names.append("OPENAI_API_BASE")
    return next((value for name in names if (value := os.getenv(name, "").strip())), None)


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
        default_base_url = "https://openrouter.ai/api/v1" if provider_name == "openrouter" else ""
        Provider.__init__(
            self,
            api_key=api_key or _provider_setting(model_ref, "API_KEY"),
            base_url=api_base or _provider_setting(model_ref, "BASE_URL") or default_base_url,
            timeout=timeout,
        )
        self.provider_name = provider_name
        self.model_name = litellm_model_name(model_ref)

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call LiteLLM and normalize its typed response for ThinHarness."""
        from litellm import aresponses

        request = {**payload, "model": self.model_name, "timeout": self.timeout}
        if self.api_key:
            request["api_key"] = self.api_key
            if self.provider_name == "openrouter":
                request["extra_headers"] = {"Authorization": f"Bearer {self.api_key}"}
        if self.base_url:
            request["api_base"] = self.base_url
        try:
            response = await aresponses(**request)
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
) -> OpenAIResponsesModel:
    """Build the model adapter ThinHarness consumes while LiteLLM owns transport."""
    _, model = parse_model_ref(model_ref)
    provider = LiteLLMResponsesProvider(model_ref, timeout=timeout)
    settings = ModelSettings(max_tokens=max_tokens, effort=effort)
    return OpenAIResponsesModel(model, provider=provider, settings=settings)
