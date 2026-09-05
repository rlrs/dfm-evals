"""DAISY with one Wikipedia lookup.

Same dataset, prompt, decoding and scorer as :mod:`dfm_evals.tasks.daisy`. The one change
is a lookup before the model answers: the question is sent, as it is, to a BM25 index over
the Danish Wikipedia dump of 1 November 2023, the introductions of the top pages are placed
in front of the DAISY prompt, and the model answers as before. No model writes a query and
no field of the benchmark other than the question is used.

The index is a SQLite FTS5 database built on first use from the public
``wikimedia/wikipedia`` dataset (configuration ``20231101.da``, about 1.5 GB, ten minutes)
and cached under ``~/.cache/dfm_evals`` or ``$DFM_EVALS_DAWIKI``. Pass ``index_path`` to
use an index that already exists.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from dfm_evals.tasks.daisy import (
    DEFAULT_HUGGING_FACE_ID,
    DEFAULT_MAX_GEN_TOKS,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_QUESTION_FIELD,
    DEFAULT_SPLIT,
    DEFAULT_TARGET_FIELD,
    DEFAULT_TEMPERATURE,
    daisy_scorer,
    record_to_sample,
)

WIKIPEDIA_DATASET_ID = "wikimedia/wikipedia"
WIKIPEDIA_CONFIG = "20231101.da"
DEFAULT_TOP_K = 3
DEFAULT_INTRO_CHARS = 900
DEFAULT_TITLE_WEIGHT = 10.0
CONTEXT_HEADER = "Baggrundsviden fra dansk Wikipedia:\n"
_QUERY_TOKEN_RE = re.compile(r"[0-9A-Za-zÆØÅæøå]+")
_MAX_QUERY_TOKENS = 16


@task(name="daisy_lookup")
def daisy_lookup(
    hugging_face_id: str = DEFAULT_HUGGING_FACE_ID,
    split: str = DEFAULT_SPLIT,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    question_field: str = DEFAULT_QUESTION_FIELD,
    target_field: str = DEFAULT_TARGET_FIELD,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
    temperature: float = DEFAULT_TEMPERATURE,
    index_path: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    intro_chars: int = DEFAULT_INTRO_CHARS,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    preferred_metric: str | None = None,
) -> Task:
    _ = preferred_metric

    if not split.strip():
        raise ValueError("`split` must be a non-empty string.")
    if max_gen_toks < 1:
        raise ValueError("`max_gen_toks` must be >= 1.")
    if top_k < 1:
        raise ValueError("`top_k` must be >= 1.")
    if intro_chars < 1:
        raise ValueError("`intro_chars` must be >= 1.")

    return Task(
        dataset=hf_dataset(
            path=hugging_face_id,
            split=split.strip(),
            sample_fields=lambda record: record_to_sample(
                record=record,
                prompt_template=prompt_template,
                question_field=question_field,
                target_field=target_field,
            ),
            auto_id=True,
            shuffle=shuffle,
            seed=seed,
            limit=limit,
        ),
        solver=[
            prepend_wikipedia_lookup(
                index_path=index_path, top_k=top_k, intro_chars=intro_chars
            ),
            generate(max_tokens=max_gen_toks, temperature=temperature),
        ],
        scorer=[daisy_scorer()],
    )


@solver
def prepend_wikipedia_lookup(
    index_path: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    intro_chars: int = DEFAULT_INTRO_CHARS,
) -> Solver:
    """Put the introductions of the best-matching Wikipedia pages before the prompt.

    The query is the question itself; the model sees the pages and the unchanged DAISY
    prompt. The titles used are recorded in ``state.metadata["lookup_titles"]``.
    """

    index = WikipediaIndex(index_path)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question = state.metadata.get("question") or state.input_text
        pages = index.lookup(question, top_k=top_k, intro_chars=intro_chars)
        state.metadata["lookup_titles"] = [title for title, _ in pages]
        if pages:
            state.user_prompt.text = format_context(pages) + state.user_prompt.text
        return state

    return solve


def format_context(pages: Iterable[tuple[str, str]]) -> str:
    """Render ``(title, introduction)`` pairs as the block placed before the prompt.

    One line per page, ``[title] introduction``, newlines inside an introduction turned
    into spaces; a blank line separates the block from the DAISY prompt.
    """

    lines = [f"[{title}] {' '.join(intro.split())}" for title, intro in pages]
    return CONTEXT_HEADER + "\n".join(lines) + "\n\n"


def fts_query(question: str, max_tokens: int = _MAX_QUERY_TOKENS) -> str:
    """FTS5 query for a question: every word as a quoted term, joined with OR.

    Any word may match, so a question never returns zero pages; BM25 ranks the pages by
    how many of the rarer words they share with the question.
    """

    tokens = _QUERY_TOKEN_RE.findall(question)[:max_tokens]
    return " OR ".join(f'"{token}"' for token in tokens) if tokens else '""'


class WikipediaIndex:
    """SQLite FTS5 index over Danish Wikipedia (title, text), one connection per thread."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        title_weight: float = DEFAULT_TITLE_WEIGHT,
    ) -> None:
        self.path = Path(path).expanduser() if path else default_index_path()
        self.title_weight = title_weight
        self._local = threading.local()
        self._ensured = False

    def _connection(self) -> sqlite3.Connection:
        if not self._ensured:
            ensure_index(self.path)
            self._ensured = True
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(str(self.path), check_same_thread=False)
            self._local.connection = connection
        return connection

    def search(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
        rows = self._connection().execute(
            "SELECT title FROM pages WHERE pages MATCH ? "
            "ORDER BY bm25(pages, ?, 1.0) LIMIT ?",
            (fts_query(question), self.title_weight, top_k),
        )
        return [title for (title,) in rows]

    def intro(self, title: str, chars: int = DEFAULT_INTRO_CHARS) -> str:
        row = (
            self._connection()
            .execute("SELECT text FROM pages WHERE title = ? LIMIT 1", (title,))
            .fetchone()
        )
        return row[0][:chars] if row else ""

    def lookup(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        intro_chars: int = DEFAULT_INTRO_CHARS,
    ) -> list[tuple[str, str]]:
        return [
            (title, self.intro(title, intro_chars))
            for title in self.search(question, top_k)
        ]


def default_index_path() -> Path:
    env = os.environ.get("DFM_EVALS_DAWIKI")
    if env:
        return Path(env).expanduser()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "dfm_evals" / f"dawiki-{WIKIPEDIA_CONFIG}.sqlite"


def ensure_index(path: Path) -> Path:
    """Build the index from the public Wikipedia dump if it is not there yet."""

    if path.exists():
        return path
    from datasets import load_dataset

    rows = load_dataset(WIKIPEDIA_DATASET_ID, WIKIPEDIA_CONFIG, split="train")
    return build_index(path, ((record["title"], record["text"]) for record in rows))


def build_index(
    path: Path, pages: Iterable[Mapping[str, Any] | tuple[str, str]]
) -> Path:
    """Write ``(title, text)`` pages into a fresh FTS5 database at ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".building")
    if tmp.exists():
        tmp.unlink()
    connection = sqlite3.connect(str(tmp))
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE pages USING fts5("
            "title, text, tokenize='unicode61 remove_diacritics 0')"
        )
        batch: list[tuple[str, str]] = []
        for page in pages:
            title, text = (
                (page["title"], page["text"]) if isinstance(page, Mapping) else page
            )
            batch.append((title, text))
            if len(batch) >= 20000:
                connection.executemany(
                    "INSERT INTO pages(title, text) VALUES (?, ?)", batch
                )
                connection.commit()
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO pages(title, text) VALUES (?, ?)", batch
            )
        connection.execute("INSERT INTO pages(pages) VALUES('optimize')")
        connection.commit()
    finally:
        connection.close()
    tmp.replace(path)
    return path
