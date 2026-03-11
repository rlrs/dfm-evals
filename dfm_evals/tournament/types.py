import hashlib
import json
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, Field

Decision = Literal["A", "B", "TIE", "INVALID"]
CanonicalDecision = Literal["A", "B", "TIE"]
InvalidPolicy = Literal["skip", "count_as_tie"]


def deterministic_id(namespace: str, *parts: str, length: int = 16) -> str:
    """Create a deterministic identifier from a namespace and string parts."""
    if namespace.strip() == "":
        raise ValueError("namespace must not be empty")
    if length <= 0:
        raise ValueError("length must be greater than 0")
    if len(parts) == 0:
        raise ValueError("at least one part is required")

    payload = "\x1f".join((namespace, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{namespace}_{digest[:length]}"


def model_id(model_name: str) -> str:
    """Create a deterministic model identifier."""
    return deterministic_id("model", model_name, length=20)


def response_id(
    model_identifier: str,
    prompt_id: str,
    *,
    source_log: str | None = None,
    sample_uuid: str | None = None,
    sample_id: str | None = None,
    response_text: str | None = None,
) -> str:
    """Create a deterministic response identifier.

    When source/log identity is provided, this identifies an immutable response
    version rather than a mutable model/prompt slot.
    """
    if (
        source_log is None
        and sample_uuid is None
        and sample_id is None
        and response_text is None
    ):
        return deterministic_id("response", model_identifier, prompt_id, length=20)

    return deterministic_id(
        "response",
        model_identifier,
        prompt_id,
        source_log or "",
        sample_uuid or "",
        sample_id or "",
        response_text or "",
        length=20,
    )


def match_id(
    model_a: str,
    model_b: str,
    prompt_id: str,
    round_index: int,
    batch_id: str,
) -> str:
    """Create a deterministic match identifier."""
    return deterministic_id(
        "match",
        model_a,
        model_b,
        prompt_id,
        str(round_index),
        batch_id,
        length=20,
    )


def default_project_id(
    models: Sequence[str],
    prompts: Sequence[Any],
    seed: int | None = None,
) -> str:
    """Create a stable project identifier from generation-defining fields."""
    model_part = ",".join(sorted(models))
    prompt_part = ",".join(sorted(_project_prompt_part(prompt) for prompt in prompts))
    seed_part = "" if seed is None else str(seed)
    return deterministic_id("project", model_part, prompt_part, seed_part, length=20)


def _project_prompt_part(prompt: Any) -> str:
    if isinstance(prompt, str):
        payload = {"id": prompt}
        return _canonical_json(payload)

    prompt_id: Any
    prompt_text: Any
    if isinstance(prompt, Mapping):
        prompt_id = prompt.get("id")
        prompt_text = prompt.get("text")
    else:
        prompt_id = getattr(prompt, "id", None)
        prompt_text = getattr(prompt, "text", None)

    if not isinstance(prompt_id, str) or prompt_id.strip() == "":
        raise ValueError("prompt must include a non-empty string id")
    if not isinstance(prompt_text, str):
        raise ValueError("prompt must include a string text value")

    return _canonical_json({"id": prompt_id, "text": prompt_text})


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ModelRating(BaseModel):
    """Current tournament rating for one model."""

    model_id: str
    mu: float
    sigma: float
    games: int = Field(default=0, ge=0)
    wins: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    ties: int = Field(default=0, ge=0)

    def conservative_score(self, conservative_k: float) -> float:
        """Compute conservative ranking score."""
        return self.mu - (conservative_k * self.sigma)
