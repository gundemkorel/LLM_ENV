from __future__ import annotations

import hashlib
import inspect
from typing import Any, Callable, Dict, List

import art  # type: ignore

from ..agents.judge import DebateJudge
from ..agents.opponent_policy import OpponentLLM
from ..utils.api import ChatCompletionsClient, CompletionChoice, CompletionResponse

_CONFIG: Dict[str, Any] = {}
_OPPONENT: OpponentLLM | None = None
_JUDGE: DebateJudge | None = None
_LLM1_CLIENT_BUILDER: Callable[[art.Model], ChatCompletionsClient] | None = None


def configure_rollout(
    *,
    config: Dict[str, Any],
    opponent: OpponentLLM,
    judge: DebateJudge,
    llm1_client_builder: Callable[[art.Model], ChatCompletionsClient],
) -> None:
    global _CONFIG, _OPPONENT, _JUDGE, _LLM1_CLIENT_BUILDER
    _CONFIG = dict(config)
    _OPPONENT = opponent
    _JUDGE = judge
    _LLM1_CLIENT_BUILDER = llm1_client_builder


def _scenario_get(scenario: Any, key: str, default: Any) -> Any:
    if scenario is None:
        return default
    if isinstance(scenario, dict):
        return scenario.get(key, default)
    return getattr(scenario, key, default)


def _initial_user_prompt(topic: str) -> str:
    return (
        "You are debating an opponent about the following topic."
        f"\nTopic: {topic}\nPresent a constructive opening argument that outlines benefits, "
        "implementation steps, and evidence."
    )


def _first_choice(response: CompletionResponse) -> CompletionChoice:
    if not response.choices:
        raise ValueError("chat completion response did not contain any choices")
    return response.choices[0]


async def _maybe_close(client: ChatCompletionsClient) -> None:
    close_callable = getattr(client, "aclose", None)
    if callable(close_callable):
        result = close_callable()
        if inspect.isawaitable(result):
            await result
        return
    close_callable = getattr(client, "close", None)
    if callable(close_callable):
        result = close_callable()
        if inspect.isawaitable(result):
            await result


async def rollout(model: art.Model, scenario: Any) -> art.Trajectory:
    if not _CONFIG or _OPPONENT is None or _JUDGE is None or _LLM1_CLIENT_BUILDER is None:
        raise RuntimeError("setup_art_training must be called before rollout.")

    llm1_client = _LLM1_CLIENT_BUILDER(model)

    try:
        max_turns = int(_CONFIG.get("max_turns", 10))
        base_seed = int(_CONFIG.get("seed", 0))
        topic = _scenario_get(scenario, "topic", _CONFIG.get("topic"))
        step = int(_scenario_get(scenario, "step", 0))
        scenario_seed = int(_scenario_get(scenario, "seed", base_seed))
        instance_input = f"{topic}|{scenario_seed}|{step}"
        instance_id = hashlib.sha256(instance_input.encode("utf-8")).hexdigest()[:8]

        metadata: Dict[str, Any] = {
            "instance_id": instance_id,
            "step": step,
            "topic": topic,
            "seed": scenario_seed,
            "max_turns": max_turns,
            "system_prompts": {
                "llm1": _CONFIG.get("llm1_system_prompt", ""),
                "llm2": _CONFIG.get("llm2_system_prompt", ""),
                "judge": _CONFIG.get("judge_system_prompt", ""),
            },
            "status": "running",
        }

        trajectory = art.Trajectory(
            messages_and_choices=[{"role": "system", "content": _CONFIG.get("llm1_system_prompt", "")}],
            metadata=metadata,
            reward=0,
        )

        trajectory.messages_and_choices.append({"role": "user", "content": _initial_user_prompt(topic)})

        dialogue_history: List[Dict[str, str]] = []
        turn_details: List[Dict[str, Any]] = []

        llm1_usage_total = 0
        llm2_usage_total = 0

        llm1_params = dict(_CONFIG.get("llm1_model_params", {}))

        for turn_index in range(max_turns):
            messages = trajectory.messages()
            completion = await llm1_client.create(
                messages=messages,
                model=model.get_inference_name(),
                **llm1_params,
            )
            llm1_choice = _first_choice(completion)
            llm1_text = llm1_choice.message.content
            trajectory.messages_and_choices.append({"role": "assistant", "content": llm1_text})
            dialogue_history.append({"speaker": "LLM1", "text": llm1_text})

            turn_record: Dict[str, Any] = {
                "llm1": {
                    "text": llm1_text,
                    "usage": completion.usage,
                    "model": model.get_inference_name(),
                    "finish_reason": llm1_choice.finish_reason,
                }
            }
            turn_details.append(turn_record)
            llm1_usage_total += int(completion.usage.get("completion_tokens", 0))

            opponent_reply = await _OPPONENT.generate_argument(
                topic=topic,
                history=dialogue_history,
                turn_index=turn_index,
            )
            trajectory.messages_and_choices.append({"role": "user", "content": opponent_reply["text"]})
            dialogue_history.append({"speaker": "LLM2", "text": opponent_reply["text"]})
            turn_record["llm2"] = {
                "text": opponent_reply["text"],
                "usage": opponent_reply["usage"],
                "model": opponent_reply["model"],
                "finish_reason": opponent_reply.get("finish_reason"),
            }
            llm2_usage_total += int(opponent_reply["usage"].get("completion_tokens", 0))

        transcript_lines = [
            f"SYSTEM[LLM1]: {_CONFIG.get('llm1_system_prompt', '').strip()}",
            f"SYSTEM[LLM2]: {_CONFIG.get('llm2_system_prompt', '').strip()}",
            f"TOPIC: {topic}",
        ]
        llm1_turn_count = 0
        llm2_turn_count = 0
        for entry in dialogue_history:
            if entry["speaker"] == "LLM1":
                llm1_turn_count += 1
                transcript_lines.append(f"LLM1[{llm1_turn_count}]: {entry['text']}")
            else:
                llm2_turn_count += 1
                transcript_lines.append(f"LLM2[{llm2_turn_count}]: {entry['text']}")
        transcript = "\n".join(transcript_lines)

        judge_result = await _JUDGE.evaluate(transcript)
        reward = 1 if judge_result["winner"] == "LLM1" else -1

        trajectory.reward = reward
        trajectory.metadata["judge"] = judge_result
        trajectory.metadata["transcript"] = transcript
        trajectory.metadata["turn_details"] = turn_details
        trajectory.metadata["status"] = "completed"
        trajectory.metrics["llm1_completion_tokens"] = llm1_usage_total
        trajectory.metrics["llm2_completion_tokens"] = llm2_usage_total
        trajectory.metrics["total_messages"] = len(trajectory.messages_and_choices)

        return trajectory
    except Exception as exc:  # pragma: no cover
        error_meta = {
            "error": "rollout_failed",
            "message": str(exc)[:200],
        }
        failure = art.Trajectory(messages_and_choices=[], metadata=error_meta, reward=-1)
        failure.metadata["status"] = "error"
        return failure
    finally:
        await _maybe_close(llm1_client)
