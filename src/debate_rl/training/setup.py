from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Dict

import art  # type: ignore

from ..agents.judge import DebateJudge, JudgeConfig
from ..agents.opponent_policy import OpponentLLM, OpponentLLMConfig
from ..utils.api import ChatCompletionsClient, build_openai_client_from_model
from .rollout import configure_rollout, rollout

_DEFAULT_TOPIC = "Should cities mandate all-electric new buildings to accelerate climate adaptation?"
_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompts.yaml"


try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    yaml = None


def _load_prompts() -> Dict[str, str]:
    text = _PROMPTS_PATH.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
        return {key: str(value) for key, value in data.items()}
    return _parse_prompts_without_yaml(text)


def _parse_prompts_without_yaml(text: str) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []
    in_block = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if current_key is None:
            if ":" not in line:
                continue
            key, remainder = line.split(":", 1)
            key = key.strip()
            remainder = remainder.strip()
            if remainder == "|":
                current_key = key
                buffer = []
                in_block = True
            else:
                prompts[key] = remainder
        else:
            if line.startswith("  "):
                buffer.append(line[2:])
            elif line.strip() == "":
                buffer.append("")
            else:
                prompts[current_key] = "\n".join(buffer).rstrip()
                current_key = None
                in_block = False
                if ":" in line:
                    key, remainder = line.split(":", 1)
                    key = key.strip()
                    remainder = remainder.strip()
                    if remainder == "|":
                        current_key = key
                        buffer = []
                        in_block = True
                    else:
                        prompts[key] = remainder
    if current_key is not None and in_block:
        prompts[current_key] = "\n".join(buffer).rstrip()
    return prompts


def setup_art_training(
    *,
    seed: int = 42,
    max_turns: int = 10,
    topic: str | None = None,
    llm1_system_prompt: str | None = None,
    llm2_system_prompt: str | None = None,
    opponent_client: ChatCompletionsClient | None = None,
    opponent_model_name: str | None = None,
    judge_client: ChatCompletionsClient | None = None,
    judge_model_name: str | None = None,
    llm1_client_builder: Callable[[art.Model], ChatCompletionsClient] | None = None,
    llm1_temperature: float = 0.7,
    llm1_max_tokens: int = 512,
    opponent_temperature: float = 0.7,
    opponent_max_tokens: int = 512,
    judge_temperature: float = 0.0,
    judge_max_tokens: int = 256,
    **kwargs: Any,
) -> Callable[[art.Model, Any], Awaitable[art.Trajectory]]:
    """Initialise module-level configuration and return the rollout coroutine."""

    if opponent_client is None:
        raise ValueError("opponent_client must be provided to setup_art_training")
    if judge_client is None:
        raise ValueError("judge_client must be provided to setup_art_training")

    prompts = _load_prompts()
    config = {
        "seed": int(seed),
        "max_turns": int(max_turns),
        "topic": topic or _DEFAULT_TOPIC,
        "llm1_system_prompt": llm1_system_prompt or prompts.get("llm1_system", ""),
        "llm2_system_prompt": llm2_system_prompt or prompts.get("llm2_system", ""),
        "judge_system_prompt": prompts.get("judge_system", ""),
        "llm1_model_params": {
            "temperature": llm1_temperature,
            "max_completion_tokens": llm1_max_tokens,
        },
        "extra_setup_kwargs": dict(kwargs),
    }

    opponent_config = OpponentLLMConfig(
        model_name=opponent_model_name or "opponent-llm",
        system_prompt=llm2_system_prompt or prompts.get("llm2_system", ""),
        temperature=opponent_temperature,
        max_completion_tokens=opponent_max_tokens,
    )
    opponent = OpponentLLM(client=opponent_client, config=opponent_config)

    judge_config = JudgeConfig(
        system_prompt=prompts.get("judge_system", ""),
        model_name=judge_model_name or "judge-llm",
        temperature=judge_temperature,
        max_completion_tokens=judge_max_tokens,
    )
    judge = DebateJudge(client=judge_client, config=judge_config)

    if llm1_client_builder is None:
        def _builder(model: art.Model) -> ChatCompletionsClient:
            return build_openai_client_from_model(model)

        llm1_client_builder = _builder

    configure_rollout(
        config=config,
        opponent=opponent,
        judge=judge,
        llm1_client_builder=llm1_client_builder,
    )
    return rollout
