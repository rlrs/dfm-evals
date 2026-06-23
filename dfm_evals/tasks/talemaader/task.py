"""Run this task with:

uv run inspect eval dfm_evals/tasks/talemaader/task.py --model openai/gpt-5-mini -T judge_model=openai/gpt-5-mini --limit 1
"""

import csv
import io
import random
from collections.abc import Iterable
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate

from dfm_evals.tasks.talemaader.prompts import JUDGE_INSTRUCTIONS_DA, JUDGE_TEMPLATE_DA

DEFAULT_SPLIT = "test"
DEFAULT_SPLIT_SEED = 4242
DEFAULT_TRAIN_SIZE = 128
DEFAULT_VAL_SIZE = 64
DEFAULT_MAX_GEN_TOKS: int | None = None

SOURCE_ZIP_URL = (
    "https://sprogtek-ressources.digst.govcloud.dk/1000%20danske%20talemaader"
    "%20og%20faste%20udtryk/talemaader_csv.zip"
)
SOURCE_CSV_NAME = "talemaader_leverance_1.csv"

PROMPT_TEMPLATE_DA = (
    "Du får en dansk talemåde. Forklar kort, hvad talemåden betyder.\n\n"
    'Talemåde: "{talemaade_udtryk}"\n\n'
    "Forklaring:"
)
TASK_NAME = "generative-talemaader"
DATASET_NAME = TASK_NAME

def _talemaader_task(
    split: str = DEFAULT_SPLIT,
    judge_model: str | None = None,
    judge_model_role: str | None = "grader",
    source_zip_url: str = SOURCE_ZIP_URL,
    source_csv_name: str = SOURCE_CSV_NAME,
    max_gen_toks: int | None = DEFAULT_MAX_GEN_TOKS,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
) -> Task:
    if not split.strip():
        raise ValueError("`split` must be a non-empty string.")
    if judge_model is None and judge_model_role is None:
        raise ValueError("Either `judge_model` or `judge_model_role` must be provided.")
    if judge_model is not None and not judge_model.strip():
        raise ValueError("`judge_model` must be None or a non-empty string.")
    if not source_zip_url.strip():
        raise ValueError("`source_zip_url` must be a non-empty string.")
    if not source_csv_name.strip():
        raise ValueError("`source_csv_name` must be a non-empty string.")
    if max_gen_toks is not None and max_gen_toks < 1:
        raise ValueError("`max_gen_toks` must be >= 1.")

    dataset = _memory_dataset(
        split=split,
        source_zip_url=source_zip_url,
        source_csv_name=source_csv_name,
        shuffle=shuffle,
        seed=seed,
        limit=limit,
    )
    generate_kwargs = {}
    if max_gen_toks is not None:
        generate_kwargs["max_tokens"] = max_gen_toks

    return Task(
        dataset=dataset,
        solver=[generate(**generate_kwargs)],
        scorer=model_graded_fact(
            template=JUDGE_TEMPLATE_DA,
            instructions=JUDGE_INSTRUCTIONS_DA,
            partial_credit=True,
            model=judge_model,
            model_role=judge_model_role,
        ),
    )


@task(name=TASK_NAME)
def generative_talemaader(
    split: str = DEFAULT_SPLIT,
    judge_model: str | None = None,
    judge_model_role: str | None = "grader",
    source_zip_url: str = SOURCE_ZIP_URL,
    source_csv_name: str = SOURCE_CSV_NAME,
    max_gen_toks: int | None = DEFAULT_MAX_GEN_TOKS,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
) -> Task:
    return _talemaader_task(
        split=split,
        judge_model=judge_model,
        judge_model_role=judge_model_role,
        source_zip_url=source_zip_url,
        source_csv_name=source_csv_name,
        max_gen_toks=max_gen_toks,
        shuffle=shuffle,
        seed=seed,
        limit=limit,
    )


def _memory_dataset(
    *,
    split: str,
    source_zip_url: str,
    source_csv_name: str,
    shuffle: bool,
    seed: int | None,
    limit: int | None,
) -> MemoryDataset:
    split_name = _normalize_split_name(split)
    records = _load_source_records(
        source_zip_url=source_zip_url,
        source_csv_name=source_csv_name,
    )
    split_records = _partition_records(records=records, split=split_name)
    samples = [record_to_sample(record=record) for record in split_records]

    if shuffle:
        random.Random(seed).shuffle(samples)
    if limit is not None:
        samples = samples[:limit]

    return MemoryDataset(
        samples=samples,
        name=DATASET_NAME,
        location=source_zip_url,
    )


def _normalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    match normalized:
        case "train" | "training":
            return "train"
        case "val" | "valid" | "validation" | "dev":
            return "val"
        case "test":
            return "test"
        case _:
            raise ValueError(
                f"Unsupported split '{split}'. Supported values: "
                "['dev', 'test', 'train', 'training', 'val', 'valid', 'validation']"
            )


def _load_source_records(
    *, source_zip_url: str, source_csv_name: str
) -> list[dict[str, Any]]:
    with urlopen(source_zip_url) as response:
        zip_bytes = response.read()

    with ZipFile(io.BytesIO(zip_bytes)) as zip_file:
        raw_csv = zip_file.read(source_csv_name)

    lines = raw_csv.decode("utf-8-sig").splitlines()
    reader = csv.DictReader(lines, delimiter="\t")
    rows = [dict(row) for row in reader]
    return _clean_rows(rows)


def _clean_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        talemaade = row.get("talemaade_udtryk")
        meaning = row.get("ddo_definition")
        if not isinstance(talemaade, str) or not talemaade.strip():
            continue
        if not isinstance(meaning, str) or not meaning.strip():
            continue
        cleaned_rows.append(
            {
                "id": f"dtm_{index}",
                "udtryk_id": row.get("udtryk_id"),
                "talemaade_udtryk": talemaade.strip(),
                "ddo_definition": meaning.strip(),
            }
        )
    return cleaned_rows


def _partition_records(
    *, records: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    if len(records) < (DEFAULT_TRAIN_SIZE + DEFAULT_VAL_SIZE + 1):
        raise ValueError(
            "Expected at least 193 rows in source dataset to create train/val/test splits."
        )

    shuffled = list(records)
    random.Random(DEFAULT_SPLIT_SEED).shuffle(shuffled)
    train_end = DEFAULT_TRAIN_SIZE
    val_end = train_end + DEFAULT_VAL_SIZE

    if split == "train":
        return shuffled[:train_end]
    if split == "val":
        return shuffled[train_end:val_end]
    return shuffled[val_end:]


def record_to_sample(record: dict[str, Any]) -> Sample:
    talemaade_udtryk = _require_non_empty_string(record, "talemaade_udtryk")
    meaning = _require_non_empty_string(record, "ddo_definition")

    raw_id = record.get("id")
    sample_id = str(raw_id).strip() if raw_id is not None else None

    return Sample(
        id=sample_id if sample_id else None,
        input=_build_prompt(talemaade_udtryk),
        target=[meaning],
        metadata={"talemaade_udtryk": talemaade_udtryk},
    )


def _build_prompt(talemaade_udtryk: str) -> str:
    return PROMPT_TEMPLATE_DA.format(talemaade_udtryk=talemaade_udtryk)


def _require_non_empty_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Record field '{field}' must be a non-empty string.")

    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"Record field '{field}' must be a non-empty string.")

    return cleaned
