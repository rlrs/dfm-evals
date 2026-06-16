from __future__ import annotations

from dfm_evals.tasks.daisy import (
    bleu_score,
    daisy_scorer,
    exact_match_score,
    f1_score,
    normalize_text,
    record_to_sample,
    score_prediction,
)


def test_record_to_sample_maps_question_answer_and_subject() -> None:
    sample = record_to_sample(
        record={
            "id": "row-7",
            "Question": "Hvem har skrevet Den lille havfrue?",
            "Answer": "H.C. Andersen",
            "Subject": "Litteratur",
        },
        prompt_template="Spørgsmål: {question}\nSvar:",
        question_field="Question",
        target_field="Answer",
    )

    assert sample.id == "row-7"
    assert sample.input == "Spørgsmål: Hvem har skrevet Den lille havfrue?\nSvar:"
    assert sample.target == ["H.C. Andersen"]
    assert sample.metadata["subject"] == "Litteratur"


def test_normalize_text_matches_upstream_ascii_only_behavior() -> None:
    assert normalize_text("Hej, verden!") == "hej verden"
    assert normalize_text("ÆØÅ 123") == "123"


def test_upstream_metrics_handle_exact_match_partial_overlap_and_bleu() -> None:
    assert exact_match_score("H.C. Andersen", "h c andersen") == 1.0
    assert f1_score("København", "Aarhus") == 0.0
    assert f1_score("Hans Christian Andersen", "Andersen") == 0.5
    assert bleu_score("Den lille havfrue eventyr", "Den lille havfrue eventyr") == 1.0


def test_score_prediction_uses_best_reference_per_metric() -> None:
    metrics = score_prediction(
        "Den lille havfrue eventyr",
        ["Søren Kierkegaard forfatter", "Den lille havfrue eventyr"],
    )

    assert metrics == {
        "exact_match": 1.0,
        "f1": 1.0,
        "bleu": 1.0,
    }


def test_daisy_scorer_registers_named_metrics() -> None:
    registry_info = daisy_scorer().__registry_info__
    metrics = registry_info.metadata["metrics"]

    assert isinstance(metrics, dict)
    assert set(metrics) == {"exact_match", "f1", "bleu"}