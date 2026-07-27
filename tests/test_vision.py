from __future__ import annotations

from types import SimpleNamespace

import pytest

from arc3 import vision


def test_describe_opening_routes_multimodal_input_through_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_responses(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(output_text="  a movable blue object  ")

    monkeypatch.setattr("litellm.responses", fake_responses)

    description, png = vision.describe_opening(
        [[0, 9], [14, 5]],
        model="openrouter:openai/gpt-5",
        api_key="test-key",
    )

    assert description == "a movable blue object"
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert captured["model"] == "openrouter/openai/gpt-5"
    assert captured["api_key"] == "test-key"
    messages = captured["input"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert content[0] == {"type": "input_text", "text": vision.PROMPT}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
