from __future__ import annotations

from pathlib import Path

import pytest

from dfm_evals.tasks.daisy_lookup import (
    CONTEXT_HEADER,
    WikipediaIndex,
    build_index,
    daisy_lookup,
    format_context,
    fts_query,
)

PAGES = [
    (
        "N.F.S. Grundtvig",
        "Nikolaj Frederik Severin Grundtvig var en dansk præst og salmedigter.",
    ),
    (
        "De levendes land",
        "De levendes land er en salme skrevet af N.F.S. Grundtvig i 1824.",
    ),
    ("Storebæltsbroen", "Storebæltsbroen forbinder Sjælland og Fyn over Storebælt."),
]


@pytest.fixture
def index(tmp_path: Path) -> WikipediaIndex:
    path = build_index(tmp_path / "dawiki.sqlite", PAGES)
    return WikipediaIndex(path)


def test_fts_query_quotes_every_word_and_joins_with_or() -> None:
    assert fts_query("Hvem skrev salmen De levendes Land?") == (
        '"Hvem" OR "skrev" OR "salmen" OR "De" OR "levendes" OR "Land"'
    )
    assert fts_query("???") == '""'


def test_search_returns_only_pages_that_share_a_word(index: WikipediaIndex) -> None:
    titles = index.search("Hvem skrev salmen De levendes Land?", top_k=2)

    assert titles == ["De levendes land"]


def test_search_ranks_a_title_match_above_a_text_match(index: WikipediaIndex) -> None:
    titles = index.search("Grundtvig", top_k=3)

    assert titles == ["N.F.S. Grundtvig", "De levendes land"]


def test_lookup_returns_titles_with_truncated_introductions(
    index: WikipediaIndex,
) -> None:
    pages = index.lookup("Hvad forbinder Storebæltsbroen?", top_k=1, intro_chars=16)

    assert pages == [("Storebæltsbroen", "Storebæltsbroen ")]


def test_format_context_places_titles_in_brackets_before_the_prompt() -> None:
    block = format_context([("De levendes land", "En salme fra 1824.")])

    assert block.startswith(CONTEXT_HEADER)
    assert "[De levendes land] En salme fra 1824." in block
    assert block.endswith("\n\n")


def test_search_without_hits_returns_empty(index: WikipediaIndex) -> None:
    assert index.search("xyzzy", top_k=3) == []


def test_daisy_lookup_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        daisy_lookup(top_k=0)
    with pytest.raises(ValueError):
        daisy_lookup(intro_chars=0)
