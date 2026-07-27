"""Run one traced ThinHarness -> LiteLLM request and print its Logfire trace ID."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import logfire
from opentelemetry import trace
from opentelemetry.trace import format_trace_id
from thinharness import Harness, HarnessConfig

from arc3.litellm_provider import build_litellm_model, configure_litellm_instrumentation


async def smoke() -> str:
    model_ref = os.getenv("RETRODICT_SMOKE_MODEL", "openai:gpt-5.5")
    configure_litellm_instrumentation()
    with tempfile.TemporaryDirectory(prefix="retrodict-litellm-smoke-") as tmp:
        config = HarnessConfig(
            root=Path(tmp),
            model=model_ref,
            system_prompt="Answer concisely.",
            builtin_tools=[],
            max_model_requests=1,
            request_timeout=60,
            max_tokens=128,
            effort="low",
            local_tracing=False,
        )
        model = build_litellm_model(model_ref, timeout=60, max_tokens=128, effort="low")
        harness = Harness(config, model=model)
        try:
            with logfire.span("retrodict LiteLLM smoke test"):
                trace_id = format_trace_id(trace.get_current_span().get_span_context().trace_id)
                result = await harness.run("Reply exactly: OK")
                if not result.text.strip():
                    raise RuntimeError("LiteLLM smoke returned empty text")
                logfire.info("LiteLLM smoke succeeded", model=model_ref)
        finally:
            await harness.aclose()
    logfire.force_flush()
    return trace_id


def main() -> None:
    print(asyncio.run(smoke()))


if __name__ == "__main__":
    main()
