from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Protocol, Sequence


@dataclass
class CompletionMessage:
    """Unified representation of a chat completion message."""

    role: str
    content: str


@dataclass
class CompletionChoice:
    """Single choice from a chat completion response."""

    message: CompletionMessage
    finish_reason: str | None = None


@dataclass
class CompletionResponse:
    """Minimal container for chat completion outputs."""

    choices: List[CompletionChoice]
    usage: Dict[str, Any] = field(default_factory=dict)
    raw_response: Any | None = None


class ChatCompletionsClient(Protocol):
    """Protocol describing the subset of chat completion APIs we rely on."""

    async def create(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse:
        ...


def _normalize_usage(raw_usage: Any) -> Dict[str, Any]:
    if raw_usage is None:
        return {}
    if isinstance(raw_usage, MutableMapping):
        return dict(raw_usage)
    if isinstance(raw_usage, Mapping):
        return dict(raw_usage)
    usage_dict: Dict[str, Any] = {}
    for key in dir(raw_usage):
        if key.startswith("_"):
            continue
        value = getattr(raw_usage, key)
        if isinstance(value, (int, float, str)):
            usage_dict[key] = value
    return usage_dict


def _choices_from_openai(response: Any) -> List[CompletionChoice]:
    choices: List[CompletionChoice] = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            role = "assistant"
            content = ""
        else:
            role = getattr(message, "role", "assistant") or "assistant"
            content = getattr(message, "content", "") or ""
        finish_reason = getattr(choice, "finish_reason", None)
        choices.append(
            CompletionChoice(
                message=CompletionMessage(role=role, content=content),
                finish_reason=finish_reason,
            )
        )
    if not choices:
        choices.append(
            CompletionChoice(
                message=CompletionMessage(role="assistant", content=""),
                finish_reason=None,
            )
        )
    return choices


class OpenAIChatCompletionsClient(ChatCompletionsClient):
    """Wrapper around the OpenAI Async chat completions API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        default_params: Dict[str, Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("The openai package is required to use OpenAIChatCompletionsClient.") from exc

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, **client_kwargs)
        self._default_params = dict(default_params or {})

    async def create(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse:
        params = dict(self._default_params)
        params.update(kwargs)
        response = await self._client.chat.completions.create(messages=list(messages), model=model, **params)
        return CompletionResponse(
            choices=_choices_from_openai(response),
            usage=_normalize_usage(getattr(response, "usage", None)),
            raw_response=response,
        )


def _choices_from_hf(response: Any) -> List[CompletionChoice]:
    choices: List[CompletionChoice] = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            role = "assistant"
            content = ""
        else:
            role = getattr(message, "role", "assistant") or "assistant"
            content = getattr(message, "content", "") or ""
        finish_reason = getattr(choice, "finish_reason", None)
        choices.append(
            CompletionChoice(
                message=CompletionMessage(role=role, content=content),
                finish_reason=finish_reason,
            )
        )
    if not choices and hasattr(response, "generated_text"):
        content = getattr(response, "generated_text", "") or ""
        choices.append(
            CompletionChoice(
                message=CompletionMessage(role="assistant", content=content),
                finish_reason=None,
            )
        )
    if not choices:
        choices.append(
            CompletionChoice(
                message=CompletionMessage(role="assistant", content=""),
                finish_reason=None,
            )
        )
    return choices


class HuggingFaceChatCompletionsClient(ChatCompletionsClient):
    """Wrapper around huggingface_hub.AsyncInferenceClient chat completions."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        timeout: float | None = None,
        default_params: Dict[str, Any] | None = None,
    ) -> None:
        try:
            from huggingface_hub import AsyncInferenceClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The huggingface-hub package is required to use HuggingFaceChatCompletionsClient."
            ) from exc

        self._client = AsyncInferenceClient(model=model, token=api_key, timeout=timeout)
        self._model = model
        self._default_params = dict(default_params or {})

    async def create(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse:
        params = dict(self._default_params)
        params.update(kwargs)
        response = await self._client.chat_completion(messages=list(messages), **params)
        usage = _normalize_usage(getattr(response, "usage", None))
        usage.setdefault("model", model or self._model)
        return CompletionResponse(
            choices=_choices_from_hf(response),
            usage=usage,
            raw_response=response,
        )


def build_openai_client_from_model(model: Any, **client_kwargs: Any) -> ChatCompletionsClient:
    """Create an OpenAI client using attributes exposed by an art.Model instance."""

    base_url = getattr(model, "inference_base_url", None)
    api_key = getattr(model, "inference_api_key", None)
    return OpenAIChatCompletionsClient(base_url=base_url, api_key=api_key, **client_kwargs)


__all__ = [
    "ChatCompletionsClient",
    "CompletionMessage",
    "CompletionChoice",
    "CompletionResponse",
    "OpenAIChatCompletionsClient",
    "HuggingFaceChatCompletionsClient",
    "build_openai_client_from_model",
]
