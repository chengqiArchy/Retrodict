"""Run one traced ThinHarness -> LiteLLM request and print its Logfire trace ID."""

from __future__ import annotations

import asyncio
import os

import logfire
from opentelemetry import trace
from opentelemetry.trace import format_trace_id
from thinharness import RequestConstants, ToolOutput

from arc3.litellm_provider import build_litellm_model, configure_litellm_instrumentation


async def smoke() -> str:
    model_ref = os.getenv("RETRODICT_SMOKE_MODEL", "openai:gpt-5.5")
    configure_litellm_instrumentation()
    model = build_litellm_model(model_ref, timeout=60, max_tokens=128, effort="low")
    session = model.new_session()
    constants = RequestConstants(
        instructions="Use the provided tool exactly once, then answer concisely.",
        tools=[
            {
                "type": "function",
                "name": "echo",
                "description": "Return the supplied value.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            }
        ],
    )
    with logfire.span("retrodict LiteLLM chat completion tool smoke test"):
        trace_id = format_trace_id(trace.get_current_span().get_span_context().trace_id)
        first = await session.start(
            "Call echo with value OK. After receiving its result, reply exactly: OK",
            constants,
        )
        if not first.tool_calls:
            raise RuntimeError("LiteLLM smoke did not return a tool call")
        second = await session.continue_with_tools(
            [ToolOutput(call_id=first.tool_calls[0].id, output="OK")],
            constants,
        )
        if second.text.strip() != "OK":
            raise RuntimeError(f"LiteLLM smoke returned unexpected final text: {second.text!r}")
        logfire.info("LiteLLM full-history tool smoke succeeded", model=model_ref)
    logfire.force_flush()
    return trace_id


def main() -> None:
    print(asyncio.run(smoke()))


if __name__ == "__main__":
    main()
