from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

from debate_rl import setup_art_training
from debate_rl.utils.api import (
    ChatCompletionsClient,
    CompletionChoice,
    CompletionMessage,
    CompletionResponse,
)


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
            content = f"After reviewing the transcript, the {self._verdict} debater made the stronger case."
            return CompletionResponse(
                choices=[
                    CompletionChoice(
                        message=CompletionMessage(role="assistant", content=content),
                        finish_reason="stop",
                    )
                ],
                usage={"model": model},
            )

        last_user = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
        text = f"{self.name} response: {last_user.splitlines()[-1][:120]}"
        usage = {
            "prompt_tokens": len(last_user.split()),
            "completion_tokens": len(text.split()),
            "model": model,
        }
        return CompletionResponse(
            choices=[
                CompletionChoice(
                    message=CompletionMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )


async def main() -> None:
    opponent_client = StubChatClient("LLM2")
    judge_client = StubChatClient("Judge", verdict="first")

    def llm1_builder(_: DummyModel) -> ChatCompletionsClient:
        return StubChatClient("LLM1")

    rollout_fn = setup_art_training(
        seed=123,
        max_turns=3,
        opponent_client=opponent_client,
        opponent_model_name="stub-llm2",
        judge_client=judge_client,
        judge_model_name="stub-judge",
        llm1_client_builder=llm1_builder,
    )
    model = DummyModel()
    topics = [
        "Should cities invest in coastal flood barriers over the next decade?",
        "Is mandating electric buses by 2030 realistic for mid-sized towns?",
        "Can rooftop solar plus storage replace peaker plants in urban centers?",
    ]
    trajectories = []
    for step, topic in enumerate(topics):
        scenario = {"step": step, "topic": topic}
        trajectory = await rollout_fn(model, scenario)
        trajectories.append(trajectory)
    first = trajectories[0]
    print("Sample metadata:", first.metadata)
    print("Transcript:")
    for entry in first.messages_and_choices:
        print(f"  {entry['role']}: {entry['content']}")
    wins = sum(1 for trajectory in trajectories if trajectory.reward > 0)
    print(f"Aggregate win rate for LLM1 over {len(trajectories)} episodes: {wins / len(trajectories):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
