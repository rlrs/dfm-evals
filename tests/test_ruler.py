from __future__ import annotations

import pytest

from dfm_evals.tasks.ruler.generators import generate_samples
from dfm_evals.tasks.ruler.presets import PRESETS, get_preset
from dfm_evals.tasks.ruler.qa_data import QABundle, QADocument, QAExample
from dfm_evals.tasks.ruler import qa_data
from dfm_evals.tasks.ruler.task import (
    _string_match_all,
    _string_match_any,
    ruler,
)
from dfm_evals.tasks.ruler.tokenizers import SimpleLengthEstimator


def test_ruler_presets_include_reference_variants() -> None:
    assert {
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery",
        "vt",
        "cwe",
        "fwe",
        "qa_1",
        "qa_2",
    }.issubset(PRESETS)


def test_generate_niah_multivalue_returns_four_targets() -> None:
    preset = get_preset("niah_multivalue")
    samples = generate_samples(
        preset=preset,
        estimator=SimpleLengthEstimator(),
        max_seq_length=512,
        reserved_output_tokens=preset.completion_tokens,
        context_buffer_tokens=32,
        num_samples=1,
        seed=7,
        remove_newline_tab=False,
    )

    sample = samples[0]
    assert isinstance(sample.target, list)
    assert len(sample.target) == 4
    assert sample.metadata["family"] == "niah"
    assert sample.metadata["variant"] == "niah_multivalue"
    assert "Question:" in sample.input


def test_generate_vt_returns_chain_length_plus_one_targets() -> None:
    preset = get_preset("vt")
    samples = generate_samples(
        preset=preset,
        estimator=SimpleLengthEstimator(),
        max_seq_length=512,
        reserved_output_tokens=preset.completion_tokens,
        context_buffer_tokens=32,
        num_samples=1,
        seed=11,
        remove_newline_tab=False,
    )

    sample = samples[0]
    assert isinstance(sample.target, list)
    assert len(sample.target) == 5
    assert sample.metadata["family"] == "variable_tracking"
    assert sample.metadata["variant"] == "vt"
    assert "assigned the value" in sample.input


def test_generate_cwe_returns_top_common_words() -> None:
    preset = get_preset("cwe")
    samples = generate_samples(
        preset=preset,
        estimator=SimpleLengthEstimator(),
        max_seq_length=512,
        reserved_output_tokens=preset.completion_tokens,
        context_buffer_tokens=32,
        num_samples=1,
        seed=19,
        remove_newline_tab=False,
    )

    sample = samples[0]
    assert isinstance(sample.target, list)
    assert len(sample.target) == 10
    assert sample.metadata["family"] == "common_words_extraction"
    assert "most common words" in sample.input


def test_generate_fwe_returns_three_targets() -> None:
    preset = get_preset("fwe")
    samples = generate_samples(
        preset=preset,
        estimator=SimpleLengthEstimator(),
        max_seq_length=512,
        reserved_output_tokens=preset.completion_tokens,
        context_buffer_tokens=32,
        num_samples=1,
        seed=23,
        remove_newline_tab=False,
    )

    sample = samples[0]
    assert isinstance(sample.target, list)
    assert len(sample.target) == 3
    assert sample.metadata["family"] == "freq_words_extraction"
    assert "coded text" in sample.input


def test_generate_qa_uses_cached_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dfm_evals.tasks.ruler import generators

    bundle = QABundle(
        examples=[
            QAExample(
                question="Which harbor city appears in the supporting documents?",
                answers=["Aarhus", "aarhus"],
                documents=[
                    QADocument(
                        id="support-1",
                        title="Support",
                        text="Aarhus is the city named in the supporting document.",
                    )
                ],
            )
        ],
        distractor_documents=[
            QADocument(
                id="dist-1",
                title="Distractor",
                text="Odense is discussed in this unrelated paragraph.",
            ),
            QADocument(
                id="dist-2",
                title="Distractor 2",
                text="A different document mentions Aalborg instead.",
            ),
        ],
    )

    monkeypatch.setattr(generators, "load_qa_bundle", lambda _: bundle)

    preset = get_preset("qa_1")
    samples = generate_samples(
        preset=preset,
        estimator=SimpleLengthEstimator(),
        max_seq_length=512,
        reserved_output_tokens=preset.completion_tokens,
        context_buffer_tokens=32,
        num_samples=1,
        seed=29,
        remove_newline_tab=False,
    )

    sample = samples[0]
    assert isinstance(sample.target, list)
    assert sample.target == ["Aarhus", "aarhus"]
    assert sample.metadata["family"] == "qa"
    assert sample.metadata["match_mode"] == "any"
    assert "Question:" in sample.input


def test_ruler_qa_cache_defaults_to_artifact_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("DFM_EVALS_RULER_QA_CACHE", raising=False)
    monkeypatch.setenv("POST_ARTIFACT_ROOT", str(tmp_path))

    assert qa_data._cache_dir() == tmp_path / "evals" / "ruler_qa_cache"


def test_ruler_qa_cache_env_override_wins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    override = tmp_path / "qa-cache"
    monkeypatch.setenv("DFM_EVALS_RULER_QA_CACHE", str(override))
    monkeypatch.setenv("POST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    assert qa_data._cache_dir() == override


def test_ruler_qa_offline_missing_cache_fails_before_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("DFM_EVALS_RULER_QA_CACHE", str(tmp_path))
    monkeypatch.setenv("DFM_EVALS_RULER_QA_OFFLINE", "1")
    qa_data.load_qa_bundle.cache_clear()

    with pytest.raises(FileNotFoundError, match="is not cached"):
        qa_data.load_qa_bundle("squad")


def test_string_match_helpers_behave_like_ruler_reference_metrics() -> None:
    assert _string_match_all("alpha beta", ["alpha", "beta"]) == 1.0
    assert _string_match_all("alpha", ["alpha", "beta"]) == 0.5
    assert _string_match_any("alpha beta", ["gamma", "beta"]) == 1.0
    assert _string_match_any("alpha beta", ["gamma", "delta"]) == 0.0


def test_ruler_task_limit_truncates_generated_dataset() -> None:
    task = ruler(
        variant="niah_single_1",
        num_samples=5,
        max_seq_length=512,
        tokenizer_backend="simple",
        limit=2,
    )

    assert task.dataset is not None
    assert len(list(task.dataset)) == 2
