from __future__ import annotations

import random

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate

from .generators import generate_samples
from .presets import get_preset
from .tokenizers import build_length_estimator

DEFAULT_VARIANT = "niah_single_1"
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_NUM_SAMPLES = 500
DEFAULT_SEED = 42
DEFAULT_TOKENIZER_BACKEND = "auto"
DEFAULT_CONTEXT_BUFFER_TOKENS = 64


@task(name="ruler")
def ruler(
    variant: str = DEFAULT_VARIANT,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
    tokenizer_backend: str = DEFAULT_TOKENIZER_BACKEND,
    tokenizer_model: str | None = None,
    remove_newline_tab: bool = False,
    shuffle: bool = False,
    limit: int | None = None,
    completion_tokens: int | None = None,
    max_gen_toks: int | None = None,
    context_buffer_tokens: int = DEFAULT_CONTEXT_BUFFER_TOKENS,
) -> Task:
    if max_seq_length < 128:
        raise ValueError("`max_seq_length` must be >= 128.")
    if num_samples < 1:
        raise ValueError("`num_samples` must be >= 1.")
    if seed < 0:
        raise ValueError("`seed` must be >= 0.")
    if limit is not None and limit < 1:
        raise ValueError("`limit` must be >= 1 when provided.")
    if completion_tokens is not None and completion_tokens < 1:
        raise ValueError("`completion_tokens` must be >= 1 when provided.")
    if max_gen_toks is not None and max_gen_toks < 1:
        raise ValueError("`max_gen_toks` must be >= 1 when provided.")
    if context_buffer_tokens < 0:
        raise ValueError("`context_buffer_tokens` must be >= 0.")

    preset = get_preset(variant)
    reserved_output_tokens = completion_tokens or preset.completion_tokens
    estimator = build_length_estimator(
        backend=tokenizer_backend,
        model_name=tokenizer_model,
    )
    samples = generate_samples(
        preset=preset,
        estimator=estimator,
        max_seq_length=max_seq_length,
        reserved_output_tokens=reserved_output_tokens,
        context_buffer_tokens=context_buffer_tokens,
        num_samples=num_samples,
        seed=seed,
        remove_newline_tab=remove_newline_tab,
    )

    if shuffle:
        random.Random(seed).shuffle(samples)
    if limit is not None:
        samples = samples[:limit]

    generate_kwargs = {}
    if max_gen_toks is not None:
        generate_kwargs["max_tokens"] = max_gen_toks

    return Task(
        dataset=MemoryDataset(
            samples=samples,
            name=f"RULER-{variant}",
            location=f"generated:{variant}",
        ),
        solver=[generate(**generate_kwargs)],
        scorer=ruler_scorer(),
    )


@scorer(metrics=[mean(), stderr()])
def ruler_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        expected = _target_values(target)
        prediction = state.output.completion or ""
        match_mode = state.metadata.get("match_mode", "all")
        if not isinstance(match_mode, str):
            match_mode = "all"

        if match_mode == "any":
            value = _string_match_any(prediction, expected)
        else:
            value = _string_match_all(prediction, expected)

        matched = [
            item for item in expected if item.lower() in prediction.lower()
        ]
        return Score(
            value=value,
            answer=prediction,
            explanation=f"matched={matched!r}, expected={expected!r}",
            metadata={
                "matched_count": len(matched),
                "expected_count": len(expected),
                "match_mode": match_mode,
            },
        )

    return score


def _target_values(target: Target) -> list[str]:
    raw_target = target.target
    if isinstance(raw_target, str):
        return [raw_target]
    return [str(value) for value in raw_target]


def _string_match_any(prediction: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    normalized = prediction.lower()
    return float(any(item.lower() in normalized for item in expected))


def _string_match_all(prediction: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    normalized = prediction.lower()
    matches = sum(1 for item in expected if item.lower() in normalized)
    return matches / len(expected)
