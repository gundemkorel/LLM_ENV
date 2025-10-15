from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from typing import Any, Dict, List

if importlib.util.find_spec("art") is not None:  # pragma: no cover
    import art  # type: ignore
else:  # pragma: no cover
    art = ModuleType("art")

    class Trajectory:  # type: ignore[override]
        def __init__(
            self,
            messages_and_choices: List[Dict[str, Any]] | None = None,
            metadata: Dict[str, Any] | None = None,
            reward: float = 0.0,
        ) -> None:
            self.messages_and_choices = list(messages_and_choices or [])
            self.metadata = dict(metadata or {})
            self.reward = reward
            self.metrics: Dict[str, Any] = {}
            self.logs: List[Dict[str, Any]] = []

        def messages(self) -> List[Dict[str, Any]]:
            output: List[Dict[str, Any]] = []
            for entry in self.messages_and_choices:
                role = entry.get("role")
                if role in {"system", "user", "assistant"}:
                    output.append({"role": role, "content": entry.get("content", "")})
            return output

    class Model:  # type: ignore[override]
        def __init__(
            self,
            inference_base_url: str = "",
            inference_api_key: str = "",
            model_name: str = "mock-llm",
        ) -> None:
            self.inference_base_url = inference_base_url
            self.inference_api_key = inference_api_key
            self.model_name = model_name

        def get_inference_name(self) -> str:
            return self.model_name

    art.Trajectory = Trajectory  # type: ignore[attr-defined]
    art.Model = Model  # type: ignore[attr-defined]
    sys.modules.setdefault("art", art)

from .training.setup import setup_art_training
from .training.rollout import rollout

__all__ = ["setup_art_training", "rollout"]
