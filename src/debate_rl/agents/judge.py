import re
from dataclasses import dataclass
from typing import Any, Dict

from ..utils.api import ChatCompletionsClient, CompletionChoice, CompletionResponse


@dataclass
class JudgeConfig:
    """Configuration for the judge backend."""

    system_prompt: str
    model_name: str
    temperature: float = 0.0
    max_completion_tokens: int = 256


class DebateJudge:
    """Judge that queries a language model and parses the textual verdict."""

    def __init__(self, *, client: ChatCompletionsClient, config: JudgeConfig) -> None:
        self._client = client
        self._config = config

    async def evaluate(self, transcript: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self._config.system_prompt},
            {
                "role": "user",
                "content": (
                    "Review the following debate transcript between LLM1 and LLM2.\n"
                    "Identify which debater presented the stronger case.\n"
                    "Respond with a short explanation and clearly state whether the first or second debater won.\n\n"
                    f"Transcript:\n{transcript}"
                ),
            },
        ]
        response = await self._client.create(
            messages=messages,
            model=self._config.model_name,
            temperature=self._config.temperature,
            max_completion_tokens=self._config.max_completion_tokens,
        )
        choice = _first_choice(response)
        verdict = choice.message.content
        winner = self._parse_winner(verdict)
        return {
            "winner": winner,
            "rationale": verdict,
            "usage": response.usage,
            "model": self._config.model_name,
            "finish_reason": choice.finish_reason,
        }

    @staticmethod
    def _parse_winner(text: str) -> str:
        normalized = text.lower().strip()
        if "first" in normalized or "llm1" in normalized:
            return "LLM1"
        if "second" in normalized or "llm2" in normalized:
            return "LLM2"
        match = re.search(r"(llm[12])", normalized)
        if match:
            return match.group(1).upper()
        return "LLM1"


def _first_choice(response: CompletionResponse) -> CompletionChoice:
    if not response.choices:
        raise ValueError("chat completion response did not contain any choices")
    return response.choices[0]
