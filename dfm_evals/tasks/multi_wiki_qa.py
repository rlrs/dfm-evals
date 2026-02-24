from __future__ import annotations

import random
import re
import string
from collections import Counter
from typing import Any, Sequence

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

PUBLIC_SOURCE_DATASET_ID = "alexandrainst/multi-wiki-qa"
DEFAULT_LANGUAGE = "da"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_ANSWER_WORDS = 3
DEFAULT_MINI_SPLIT_SEED = 4242

MIN_NUM_CHARS_IN_CONTEXT = 30
MAX_NUM_CHARS_IN_CONTEXT = 5000
MIN_NUM_CHARS_IN_QUESTION = 10
MAX_NUM_CHARS_IN_QUESTION = 150

MINI_SPLIT_SIZES = {"train": 1024, "val": 256, "test": 2048}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
}

PROMPT_TEMPLATES = {
    "da": (
        "Tekst: {context}\n\n"
        "Besvar følgende spørgsmål om teksten ovenfor med maks. {max_words} ord.\n\n"
        "Spørgsmål: {question}"
    ),
    "en": (
        "Text: {context}\n\n"
        "Answer the following question about the text above in at most {max_words} words.\n\n"
        "Question: {question}"
    ),
}

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION = set(string.punctuation)


@task(name="multi_wiki_qa")
def multi_wiki_qa(
    language: str = DEFAULT_LANGUAGE,
    split: str = DEFAULT_SPLIT,
    dataset_id: str | None = None,
    public_source_dataset_id: str = PUBLIC_SOURCE_DATASET_ID,
    max_answer_words: int = DEFAULT_MAX_ANSWER_WORDS,
    mini_split_seed: int = DEFAULT_MINI_SPLIT_SEED,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
) -> Task:
    language = language.strip().lower()
    if not language:
        raise ValueError("`language` must be a non-empty language code.")
    split = _normalize_split_name(split)
    if not split:
        raise ValueError("`split` must be a non-empty string.")
    if max_answer_words < 1:
        raise ValueError("`max_answer_words` must be >= 1.")
    if mini_split_seed < 0:
        raise ValueError("`mini_split_seed` must be >= 0.")

    path = dataset_id.strip() if dataset_id is not None else ""
    if path:
        dataset = hf_dataset(
            path=path,
            split=split,
            sample_fields=lambda record: record_to_sample(
                record=record,
                language=language,
                max_answer_words=max_answer_words,
            ),
            auto_id=True,
            shuffle=shuffle,
            seed=seed,
            limit=limit,
        )
    else:
        dataset = _build_public_mini_dataset(
            source_dataset_id=public_source_dataset_id,
            language=language,
            split=split,
            max_answer_words=max_answer_words,
            mini_split_seed=mini_split_seed,
            shuffle=shuffle,
            seed=seed,
            limit=limit,
        )

    return Task(dataset=dataset, scorer=multi_wiki_qa_scorer())


def _normalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    if not normalized:
        return ""
    if normalized not in SPLIT_ALIASES:
        allowed = sorted(set(SPLIT_ALIASES.keys()))
        raise ValueError(
            f"Unsupported split '{split}'. Supported values: {allowed}"
        )
    return SPLIT_ALIASES[normalized]


def _build_public_mini_dataset(
    source_dataset_id: str,
    language: str,
    split: str,
    max_answer_words: int,
    mini_split_seed: int,
    shuffle: bool,
    seed: int | None,
    limit: int | None,
) -> MemoryDataset:
    records = _load_public_records(source_dataset_id=source_dataset_id, language=language)
    filtered_records = [record for record in records if _is_valid_public_record(record)]
    split_records = _select_mini_split_records(
        records=filtered_records,
        split=split,
        mini_split_seed=mini_split_seed,
    )

    samples: list[Sample] = []
    seen_ids: dict[str, int] = {}
    for idx, record in enumerate(split_records):
        record_with_id = dict(record)
        raw_id_value = record_with_id.get("id")
        base_id = (
            str(raw_id_value)
            if raw_id_value is not None and str(raw_id_value).strip()
            else f"{language}_{split}_{idx}"
        )
        duplicate_count = seen_ids.get(base_id, 0)
        seen_ids[base_id] = duplicate_count + 1
        if duplicate_count > 0:
            record_with_id["id"] = f"{base_id}__{duplicate_count}"
        else:
            record_with_id["id"] = base_id
        samples.append(
            record_to_sample(
                record=record_with_id,
                language=language,
                max_answer_words=max_answer_words,
            )
        )

    if shuffle:
        random.Random(seed).shuffle(samples)
    if limit is not None:
        samples = samples[:limit]

    return MemoryDataset(
        samples=samples,
        name=f"MultiWikiQA-{language}",
        location=source_dataset_id,
    )


def _load_public_records(source_dataset_id: str, language: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading public MultiWikiQA requires the `datasets` package."
        ) from exc

    dataset = load_dataset(source_dataset_id, name=language, split="train")
    return [dict(record) for record in dataset]


def _is_valid_public_record(record: dict[str, Any]) -> bool:
    context = record.get("context")
    question = record.get("question")
    if not isinstance(context, str) or not isinstance(question, str):
        return False

    if not (MIN_NUM_CHARS_IN_CONTEXT <= len(context) <= MAX_NUM_CHARS_IN_CONTEXT):
        return False
    if not (MIN_NUM_CHARS_IN_QUESTION <= len(question) <= MAX_NUM_CHARS_IN_QUESTION):
        return False

    answers = record.get("answers")
    if not isinstance(answers, dict):
        return False
    texts = answers.get("text")
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, Sequence) or len(texts) == 0:
        return False
    first_answer = texts[0]
    if not isinstance(first_answer, str) or not first_answer.strip():
        return False

    return True


def _select_mini_split_records(
    records: list[dict[str, Any]], split: str, mini_split_seed: int
) -> list[dict[str, Any]]:
    required_total = sum(MINI_SPLIT_SIZES.values())
    if len(records) < required_total:
        raise ValueError(
            f"Not enough filtered records to build mini splits: have {len(records)}, "
            f"need at least {required_total}."
        )

    rng = random.Random(mini_split_seed)
    available_indices = list(range(len(records)))

    val_indices = rng.sample(available_indices, MINI_SPLIT_SIZES["val"])
    val_index_set = set(val_indices)
    remaining_after_val = [idx for idx in available_indices if idx not in val_index_set]

    test_indices = rng.sample(remaining_after_val, MINI_SPLIT_SIZES["test"])
    test_index_set = set(test_indices)
    remaining_after_test = [
        idx for idx in remaining_after_val if idx not in test_index_set
    ]

    train_indices = rng.sample(remaining_after_test, MINI_SPLIT_SIZES["train"])

    indices_by_split = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    return [records[idx] for idx in indices_by_split[split]]


def record_to_sample(
    record: dict[str, Any], language: str, max_answer_words: int
) -> Sample:
    context = _require_string(record, "context")
    question = _require_string(record, "question")
    answers = _extract_answer_texts(record)
    prompt = _build_prompt(
        context=context,
        question=question,
        language=language,
        max_answer_words=max_answer_words,
    )
    sample_id = record.get("id")

    return Sample(
        id=str(sample_id) if sample_id is not None else None,
        input=prompt,
        target=answers,
        metadata={"language": language},
    )


def _build_prompt(
    context: str, question: str, language: str, max_answer_words: int
) -> str:
    template = PROMPT_TEMPLATES.get(language, PROMPT_TEMPLATES["en"])
    return template.format(
        context=context, question=question, max_words=max_answer_words
    )


def _require_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record field '{field}' must be a non-empty string.")
    return value


def _extract_answer_texts(record: dict[str, Any]) -> list[str]:
    answers = record.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("Record field 'answers' must be an object.")

    raw_texts = answers.get("text")
    texts: Sequence[Any]
    if isinstance(raw_texts, str):
        texts = [raw_texts]
    elif isinstance(raw_texts, Sequence):
        texts = raw_texts
    else:
        raise ValueError("Record field 'answers.text' must be a string or list.")

    deduped: list[str] = []
    seen: set[str] = set()
    for value in texts:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        deduped.append(cleaned)
        seen.add(cleaned)

    if not deduped:
        raise ValueError("Record field 'answers.text' does not contain any answers.")

    return deduped


@scorer(metrics={"exact_match": [mean(), stderr()], "f1": [mean(), stderr()]})
def multi_wiki_qa_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        references = [item for item in target if item.strip()]
        exact_match = _max_exact_match(prediction, references)
        f1_score = _max_f1(prediction, references)
        return Score(
            value={"exact_match": exact_match, "f1": f1_score},
            answer=prediction,
        )

    return score


def _max_exact_match(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    normalized_prediction = _normalize_answer(prediction)
    return max(
        float(normalized_prediction == _normalize_answer(reference))
        for reference in references
    )


def _max_f1(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(_f1_score(prediction, reference) for reference in references)


def _f1_score(prediction: str, reference: str) -> float:
    prediction_tokens = _normalize_answer(prediction).split()
    reference_tokens = _normalize_answer(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return (2 * precision * recall) / (precision + recall)


def _normalize_answer(text: str) -> str:
    lowered = text.casefold()
    without_punctuation = "".join(
        character for character in lowered if character not in _PUNCTUATION
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())
