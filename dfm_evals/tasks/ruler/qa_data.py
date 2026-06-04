from __future__ import annotations

import fcntl
import json
import os
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

QADatasetName = Literal["squad", "hotpotqa"]

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
HOTPOTQA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_ATTEMPTS = 3


@dataclass(frozen=True)
class QADocument:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class QAExample:
    question: str
    answers: list[str]
    documents: list[QADocument]


@dataclass(frozen=True)
class QABundle:
    examples: list[QAExample]
    distractor_documents: list[QADocument]


@lru_cache(maxsize=4)
def load_qa_bundle(dataset: QADatasetName) -> QABundle:
    cache_file = _cache_dir() / f"{dataset}.json"
    if not cache_file.is_file():
        if _offline_mode():
            raise FileNotFoundError(
                f"RULER QA dataset {dataset!r} is not cached at {cache_file}. "
                "Prefetch it before launching offline LUMI eval jobs."
            )
        _download_dataset(dataset, cache_file)

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid cached RULER QA dataset at {cache_file}. "
            "Delete the file and prefetch it again."
        ) from exc
    if dataset == "squad":
        return _parse_squad_bundle(payload)
    if dataset == "hotpotqa":
        return _parse_hotpotqa_bundle(payload)
    raise ValueError(f"Unsupported QA dataset {dataset!r}")


def _cache_dir() -> Path:
    override = os.environ.get("DFM_EVALS_RULER_QA_CACHE")
    if override:
        root = Path(override).expanduser()
    elif artifact_root := os.environ.get("POST_ARTIFACT_ROOT"):
        root = Path(artifact_root).expanduser() / "evals" / "ruler_qa_cache"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
        root = base / "dfm_evals" / "ruler" / "qa"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _download_dataset(dataset: QADatasetName, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(target.with_suffix(target.suffix + ".lock")):
        if target.is_file():
            return

        _download_dataset_locked(dataset, target)


def _download_dataset_locked(dataset: QADatasetName, target: Path) -> None:
    url = SQUAD_URL if dataset == "squad" else HOTPOTQA_URL
    timeout = float(
        os.environ.get("DFM_EVALS_RULER_QA_DOWNLOAD_TIMEOUT", DOWNLOAD_TIMEOUT_SECONDS)
    )
    last_error: BaseException | None = None

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "dfm-evals-ruler/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            json.loads(data.decode("utf-8"))
            tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
            return
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(min(2**attempt, 10))

    raise RuntimeError(
        f"Could not download RULER QA dataset {dataset!r} from {url}. "
        f"Prefetch it into {_cache_dir()} before launching LUMI eval jobs."
    ) from last_error


def _offline_mode() -> bool:
    return os.environ.get("DFM_EVALS_RULER_QA_OFFLINE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@contextmanager
def _exclusive_lock(lock_path: Path):
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def prefetch_qa_datasets(
    datasets: tuple[QADatasetName, ...] = ("squad", "hotpotqa"),
) -> list[Path]:
    load_qa_bundle.cache_clear()
    cache_dir = _cache_dir()
    paths: list[Path] = []
    for dataset in datasets:
        _download_dataset(dataset, cache_dir / f"{dataset}.json")
        load_qa_bundle(dataset)
        paths.append(cache_dir / f"{dataset}.json")
    return paths


def _parse_squad_bundle(payload: object) -> QABundle:
    if not isinstance(payload, dict):
        raise ValueError("Invalid SQuAD payload: expected an object.")

    raw_articles = payload.get("data")
    if not isinstance(raw_articles, list):
        raise ValueError("Invalid SQuAD payload: missing `data` list.")

    examples: list[QAExample] = []
    documents: list[QADocument] = []

    for article_index, article in enumerate(raw_articles):
        if not isinstance(article, dict):
            continue
        title = _clean_text(article.get("title")) or f"SQuAD Article {article_index + 1}"
        raw_paragraphs = article.get("paragraphs")
        if not isinstance(raw_paragraphs, list):
            continue

        for paragraph_index, paragraph in enumerate(raw_paragraphs):
            if not isinstance(paragraph, dict):
                continue
            context = _clean_text(paragraph.get("context"))
            if not context:
                continue

            document = QADocument(
                id=f"squad:{article_index}:{paragraph_index}",
                title=title,
                text=context,
            )
            documents.append(document)

            raw_qas = paragraph.get("qas")
            if not isinstance(raw_qas, list):
                continue

            for qa in raw_qas:
                if not isinstance(qa, dict):
                    continue
                if qa.get("is_impossible") is True:
                    continue

                question = _clean_text(qa.get("question"))
                if not question:
                    continue

                raw_answers = qa.get("answers")
                if not isinstance(raw_answers, list):
                    continue

                answers = _dedupe_preserve_order(
                    _clean_text(answer.get("text"))
                    for answer in raw_answers
                    if isinstance(answer, dict)
                )
                if not answers:
                    continue

                examples.append(
                    QAExample(
                        question=question,
                        answers=answers,
                        documents=[document],
                    )
                )

    return QABundle(examples=examples, distractor_documents=documents)


def _parse_hotpotqa_bundle(payload: object) -> QABundle:
    if not isinstance(payload, list):
        raise ValueError("Invalid HotpotQA payload: expected a list.")

    examples: list[QAExample] = []
    documents: list[QADocument] = []

    for example_index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue

        question = _clean_text(item.get("question"))
        answer = _clean_text(item.get("answer"))
        raw_context = item.get("context")
        if not question or not answer or not isinstance(raw_context, list):
            continue

        example_documents: list[QADocument] = []
        for document_index, raw_document in enumerate(raw_context):
            if (
                not isinstance(raw_document, list)
                or len(raw_document) != 2
                or not isinstance(raw_document[0], str)
                or not isinstance(raw_document[1], list)
            ):
                continue

            title = raw_document[0].strip() or f"Hotpot Document {document_index + 1}"
            sentences = [
                sentence.strip()
                for sentence in raw_document[1]
                if isinstance(sentence, str) and sentence.strip()
            ]
            if not sentences:
                continue

            document = QADocument(
                id=f"hotpot:{example_index}:{document_index}",
                title=title,
                text=" ".join(sentences),
            )
            example_documents.append(document)
            documents.append(document)

        if not example_documents:
            continue

        examples.append(
            QAExample(
                question=question,
                answers=[answer],
                documents=example_documents,
            )
        )

    return QABundle(examples=examples, distractor_documents=documents)


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _dedupe_preserve_order(values: object) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def main() -> None:
    for path in prefetch_qa_datasets():
        print(path)


if __name__ == "__main__":
    main()
