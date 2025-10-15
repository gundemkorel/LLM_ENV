from __future__ import annotations

import sys
from pathlib import Path
import asyncio
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from debate_rl import setup_art_training
from debate_rl.utils.api import ChatCompletionsClient, CompletionChoice, CompletionMessage, CompletionResponse


class DummyModel:
    inference_base_url = ""
    inference_api_key = ""
    model_name = "stub-llm1"

    def get_inference_name(self) -> str:
        return self.model_name


class StubChatClient(ChatCompletionsClient):
    def __init__(self, name: str, *, verdict: str | None = None) -> None:
        self.name = name
        self._verdict = verdict

    async def create(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse:
        if self._verdict is not None:
            text = f"In this debate, the {self._verdict} debater presented the stronger case."
            return CompletionResponse(
                choices=[
                    CompletionChoice(
                        message=CompletionMessage(role="assistant", content=text),
                        finish_reason="stop",
                    )
                ],
                usage={"model": model},
            )

        last_user = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
        reply = f"{self.name} counters: {last_user[:80]}"
        usage = {"prompt_tokens": len(last_user.split()), "completion_tokens": len(reply.split()), "model": model}
        return CompletionResponse(
            choices=[
                CompletionChoice(
                    message=CompletionMessage(role="assistant", content=reply),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )


def test_rollout_structure() -> None:
    opponent_client = StubChatClient("LLM2")
    judge_client = StubChatClient("Judge", verdict="first")

    def llm1_builder(_: DummyModel) -> ChatCompletionsClient:
        return StubChatClient("LLM1")

    max_turns = 4
    rollout_fn = setup_art_training(
        seed=321,
        max_turns=max_turns,
        opponent_client=opponent_client,
        opponent_model_name="stub-llm2",
        judge_client=judge_client,
        judge_model_name="stub-judge",
        llm1_client_builder=llm1_builder,
    )
    trajectory = asyncio.run(rollout_fn(DummyModel(), {"step": 0}))

    assert trajectory.reward in (1, -1)
    messages = trajectory.messages_and_choices
    assert messages[0]["role"] == "system"
    assert isinstance(messages[0]["content"], str)
    for entry in messages:
        assert set(entry.keys()) == {"role", "content"}
    roles = [entry["role"] for entry in messages]
    expected_roles = ["system", "user"] + [role for _ in range(max_turns) for role in ("assistant", "user")]
    assert roles == expected_roles

    assistant_messages = [entry for entry in messages if entry["role"] == "assistant"]
    assert len(assistant_messages) == max_turns

    turn_details = trajectory.metadata.get("turn_details")
    assert isinstance(turn_details, list)
    assert len(turn_details) == max_turns
    for turn in turn_details:
        assert set(turn.keys()) == {"llm1", "llm2"}
        assert {"text", "usage", "model", "finish_reason"} <= set(turn["llm1"].keys())
        assert {"text", "usage", "model", "finish_reason"} <= set(turn["llm2"].keys())

    assert trajectory.metadata["max_turns"] == max_turns
    assert "topic" in trajectory.metadata
    assert "judge" in trajectory.metadata
    winner = trajectory.metadata["judge"]["winner"]
    assert (winner == "LLM1") == (trajectory.reward == 1)
