from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from ..utils.api import ChatCompletionsClient, CompletionChoice, CompletionResponse


@dataclass
class OpponentLLMConfig:
    """Configuration describing how the scripted opponent should call its backend."""

    model_name: str
    system_prompt: str
    temperature: float = 0.7
    max_completion_tokens: int = 512


class OpponentLLM:
    """Represents the untrainable second debater backed by a chat-completion client."""

    def __init__(self, *, client: ChatCompletionsClient, config: OpponentLLMConfig) -> None:
        self._client = client
        self._config = config

    async def generate_argument(
        self,
        *,
        topic: str,
        history: Sequence[Dict[str, str]],
        turn_index: int,
    ) -> Dict[str, Any]:
        """Produce an LLM2 argument from the perspective of the configured opponent."""

        if not history or history[-1].get("speaker") != "LLM1":
            raise ValueError("history must end with an LLM1 message before generating an opponent reply")

        llm1_last = history[-1]["text"]
        prior_turns = "\n".join(
            f"Turn {idx + 1} - {item['speaker']}: {item['text']}" for idx, item in enumerate(history[:-1])
        )
        prompt_lines: List[str] = [
            f"Topic: {topic}",
            f"This is turn {turn_index + 1} of the debate.",
            "You are preparing the counter-argument on behalf of LLM2.",
            "Respond with a concise rebuttal that highlights risks, trade-offs, or unanswered questions.",
            f"LLM1 just argued:\n{llm1_last}",
        ]
        if prior_turns:
            prompt_lines.append("Conversation so far:")
            prompt_lines.append(prior_turns)

        messages = [
            {"role": "system", "content": self._config.system_prompt},
            {"role": "user", "content": "\n\n".join(prompt_lines)},
        ]

        response = await self._client.create(
            messages=messages,
            model=self._config.model_name,
            temperature=self._config.temperature,
            max_completion_tokens=self._config.max_completion_tokens,
        )
        choice = _first_choice(response)
        text = choice.message.content
        return {
            "text": text,
            "usage": response.usage,
            "model": self._config.model_name,
            "finish_reason": choice.finish_reason,
        }


def _first_choice(response: CompletionResponse) -> CompletionChoice:
    if not response.choices:
        raise ValueError("chat completion response did not contain any choices")
    return response.choices[0]
