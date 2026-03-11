import sys
import types
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


def _install_trueskill_stub() -> None:
    if "trueskill" in sys.modules:
        return

    trueskill = types.ModuleType("trueskill")

    class Rating:
        def __init__(self, mu: float = 25.0, sigma: float = 25.0 / 3.0) -> None:
            self.mu = mu
            self.sigma = sigma

    class TrueSkill:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def cdf(self, value: float) -> float:
            del value
            return 0.5

    def rate_1vs1(
        first: Rating,
        second: Rating,
        drawn: bool = False,
        env: TrueSkill | None = None,
    ) -> tuple[Rating, Rating]:
        del drawn, env
        return first, second

    trueskill.Rating = Rating
    trueskill.TrueSkill = TrueSkill
    trueskill.rate_1vs1 = rate_1vs1
    sys.modules["trueskill"] = trueskill


_install_trueskill_stub()


@lru_cache(maxsize=1)
def _tournament_modules() -> dict[str, object]:
    from dfm_evals.tournament._provenance import (
        GENERATION_PHASE,
        GENERATION_TASK_NAME,
        TOURNAMENT_PHASE_KEY,
        TOURNAMENT_PROJECT_KEY,
        resolve_tournament_project_id,
    )
    from dfm_evals.tournament.config import TournamentConfig, TournamentPrompt
    from dfm_evals.tournament.generation import run_generation
    from dfm_evals.tournament.indexer import index_generation_responses
    from dfm_evals.tournament.store import TournamentStore, initialize_tournament_store
    from dfm_evals.tournament.types import default_project_id, model_id

    return {
        "GENERATION_PHASE": GENERATION_PHASE,
        "GENERATION_TASK_NAME": GENERATION_TASK_NAME,
        "TOURNAMENT_PHASE_KEY": TOURNAMENT_PHASE_KEY,
        "TOURNAMENT_PROJECT_KEY": TOURNAMENT_PROJECT_KEY,
        "TournamentConfig": TournamentConfig,
        "TournamentPrompt": TournamentPrompt,
        "TournamentStore": TournamentStore,
        "default_project_id": default_project_id,
        "index_generation_responses": index_generation_responses,
        "initialize_tournament_store": initialize_tournament_store,
        "model_id": model_id,
        "resolve_tournament_project_id": resolve_tournament_project_id,
        "run_generation": run_generation,
    }


def _config(tmp_path: Path, models: list[str]) -> object:
    modules = _tournament_modules()
    TournamentConfig = modules["TournamentConfig"]
    TournamentPrompt = modules["TournamentPrompt"]
    return TournamentConfig(
        run_dir=tmp_path / "logs",
        contestant_models=models,
        prompts=[TournamentPrompt(id="prompt-1", text="Hello")],
        judge_model="judge/model",
        judge_prompt_template="{prompt}\nA:{response_a}\nB:{response_b}",
    )


def test_resolve_tournament_project_id_prefers_persisted_state_id(tmp_path: Path) -> None:
    modules = _tournament_modules()
    config = _config(tmp_path, ["model/A", "model/B"])
    updated = config.model_copy(
        update={"contestant_models": ["model/A", "model/B", "model/C"]}
    )

    modules["initialize_tournament_store"](config)

    resolved_project_id = modules["resolve_tournament_project_id"](updated)
    new_default_project_id = modules["default_project_id"](
        updated.contestant_models,
        updated.prompts,
        seed=updated.seed,
    )

    assert resolved_project_id != new_default_project_id
    assert resolved_project_id == modules["resolve_tournament_project_id"](config)


def test_default_project_id_changes_when_prompt_text_changes(tmp_path: Path) -> None:
    modules = _tournament_modules()
    TournamentPrompt = modules["TournamentPrompt"]
    config = _config(tmp_path, ["model/A", "model/B"])
    updated = config.model_copy(
        update={"prompts": [TournamentPrompt(id="prompt-1", text="Updated prompt text")]}
    )

    original_project_id = modules["default_project_id"](
        config.contestant_models,
        config.prompts,
        seed=config.seed,
    )
    updated_project_id = modules["default_project_id"](
        updated.contestant_models,
        updated.prompts,
        seed=updated.seed,
    )

    assert original_project_id != updated_project_id
    assert modules["resolve_tournament_project_id"](config) == original_project_id
    assert modules["resolve_tournament_project_id"](updated) == updated_project_id


def test_run_generation_uses_persisted_project_id_in_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modules = _tournament_modules()
    config = _config(tmp_path, ["model/A", "model/B"])
    updated = config.model_copy(
        update={"contestant_models": ["model/A", "model/B", "model/C"]}
    )
    modules["initialize_tournament_store"](config)

    generation_module = sys.modules["dfm_evals.tournament.generation"]
    captured_metadata: list[dict[str, str]] = []

    def fake_eval_set(
        *,
        tasks: object,
        model: object,
        log_dir: str,
        metadata: dict[str, str],
        score: bool,
        log_dir_allow_dirty: bool,
    ) -> tuple[bool, list[object]]:
        del tasks, model, log_dir, score, log_dir_allow_dirty
        captured_metadata.append(metadata)
        return True, []

    monkeypatch.setattr(generation_module, "get_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(generation_module, "eval_set", fake_eval_set)
    monkeypatch.setattr(generation_module, "_close_model", lambda model: None)

    modules["run_generation"](updated, models=["model/C"])

    assert len(captured_metadata) == 1
    assert (
        captured_metadata[0][modules["TOURNAMENT_PROJECT_KEY"]]
        == modules["resolve_tournament_project_id"](config)
    )
    assert (
        captured_metadata[0][modules["TOURNAMENT_PHASE_KEY"]]
        == modules["GENERATION_PHASE"]
    )


def test_run_generation_indexes_state_after_each_completed_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modules = _tournament_modules()
    config = _config(tmp_path, ["model/A", "model/B"])
    modules["initialize_tournament_store"](config)

    generation_module = sys.modules["dfm_evals.tournament.generation"]
    indexed_configs: list[object] = []

    def fake_eval_set(
        *,
        tasks: object,
        model: object,
        log_dir: str,
        metadata: dict[str, str],
        score: bool,
        log_dir_allow_dirty: bool,
    ) -> tuple[bool, list[object]]:
        del tasks, model, log_dir, metadata, score, log_dir_allow_dirty
        return True, []

    monkeypatch.setattr(generation_module, "get_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(generation_module, "eval_set", fake_eval_set)
    monkeypatch.setattr(generation_module, "_close_model", lambda model: None)
    monkeypatch.setattr(
        generation_module,
        "index_generation_responses",
        lambda config: indexed_configs.append(config),
    )

    modules["run_generation"](config, models=["model/A", "model/B"])

    assert len(indexed_configs) == 2
    assert all(indexed.contestant_models == config.contestant_models for indexed in indexed_configs)


def test_run_generation_removes_existing_logs_for_selected_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modules = _tournament_modules()
    config = _config(tmp_path, ["model/A", "model/B"])
    generation_module = sys.modules["dfm_evals.tournament.generation"]

    old_log = config.generation_log_dir / "old-model-a.eval"
    old_log.parent.mkdir(parents=True, exist_ok=True)
    old_log.write_text("stale", encoding="utf-8")

    expected_metadata = {
        modules["TOURNAMENT_PHASE_KEY"]: modules["GENERATION_PHASE"],
        modules["TOURNAMENT_PROJECT_KEY"]: modules["resolve_tournament_project_id"](config),
    }

    def fake_eval_set(
        *,
        tasks: object,
        model: object,
        log_dir: str,
        metadata: dict[str, str],
        score: bool,
        log_dir_allow_dirty: bool,
    ) -> tuple[bool, list[object]]:
        del tasks, model, log_dir, metadata, score, log_dir_allow_dirty
        assert not old_log.exists()
        return True, []

    monkeypatch.setattr(generation_module, "get_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(generation_module, "_close_model", lambda model: None)
    monkeypatch.setattr(generation_module, "eval_set", fake_eval_set)
    monkeypatch.setattr(generation_module, "index_generation_responses", lambda config: None)
    monkeypatch.setattr(
        generation_module,
        "list_eval_logs",
        lambda path: [SimpleNamespace(name=old_log.as_uri())],
    )
    monkeypatch.setattr(
        generation_module,
        "read_eval_log",
        lambda log_info, header_only=True: SimpleNamespace(
            eval=SimpleNamespace(
                task=modules["GENERATION_TASK_NAME"],
                model="model/A",
                metadata=expected_metadata,
            )
        ),
    )

    modules["run_generation"](config, models=["model/A"])

    assert not old_log.exists()


def test_index_generation_responses_requires_matching_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modules = _tournament_modules()
    config = _config(tmp_path, ["model/A", "model/B"])
    expected_project_id = modules["resolve_tournament_project_id"](config)
    foreign_project_id = "foreign-project"

    indexer_module = sys.modules["dfm_evals.tournament.indexer"]

    matching_log = SimpleNamespace(
        name=(config.generation_log_dir / "matching.eval").as_posix(),
        mtime=100.0,
    )
    foreign_log = SimpleNamespace(
        name=(config.generation_log_dir / "foreign.eval").as_posix(),
        mtime=100.0,
    )

    headers = {
        matching_log.name: SimpleNamespace(
            eval=SimpleNamespace(
                task=modules["GENERATION_TASK_NAME"],
                model="model/A",
                metadata={
                    modules["TOURNAMENT_PHASE_KEY"]: modules["GENERATION_PHASE"],
                    modules["TOURNAMENT_PROJECT_KEY"]: expected_project_id,
                },
            )
        ),
        foreign_log.name: SimpleNamespace(
            eval=SimpleNamespace(
                task=modules["GENERATION_TASK_NAME"],
                model="model/A",
                metadata={
                    modules["TOURNAMENT_PHASE_KEY"]: modules["GENERATION_PHASE"],
                    modules["TOURNAMENT_PROJECT_KEY"]: foreign_project_id,
                },
            )
        ),
    }
    samples = {
        matching_log.name: [
            SimpleNamespace(
                metadata={"prompt_id": "prompt-1"},
                id="prompt-1",
                output=SimpleNamespace(completion="expected"),
                uuid="uuid-1",
            )
        ],
        foreign_log.name: [
            SimpleNamespace(
                metadata={"prompt_id": "prompt-1"},
                id="prompt-1",
                output=SimpleNamespace(completion="foreign"),
                uuid="uuid-2",
            )
        ],
    }

    monkeypatch.setattr(
        indexer_module,
        "list_eval_logs",
        lambda path: [matching_log, foreign_log],
    )
    monkeypatch.setattr(
        indexer_module,
        "read_eval_log",
        lambda log_info, header_only=True: headers[log_info.name],
    )
    monkeypatch.setattr(
        indexer_module,
        "read_eval_log_samples",
        lambda log_info, all_samples_required=False: samples[log_info.name],
    )

    report = modules["index_generation_responses"](config)

    assert report.logs_seen == 2
    assert report.logs_processed == 1
    assert report.logs_skipped_provenance == 1
    assert report.responses_inserted == 1

    with modules["TournamentStore"](config.state_dir) as store:
        model_identifier = modules["model_id"]("model/A")
        row = store.connection().execute(
            """
            SELECT response_text
            FROM responses
            WHERE model_id = ?
              AND prompt_id = ?
              AND current = 1
            """,
            (model_identifier, "prompt-1"),
        ).fetchone()

    assert row is not None
    assert row["response_text"] == "expected"


def test_index_generation_responses_rejects_stale_logs_when_prompt_text_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modules = _tournament_modules()
    TournamentPrompt = modules["TournamentPrompt"]
    config = _config(tmp_path, ["model/A", "model/B"])
    updated = config.model_copy(
        update={"prompts": [TournamentPrompt(id="prompt-1", text="Updated prompt text")]}
    )
    stale_project_id = modules["resolve_tournament_project_id"](config)
    indexer_module = sys.modules["dfm_evals.tournament.indexer"]

    stale_log = SimpleNamespace(
        name=(updated.generation_log_dir / "stale.eval").as_posix(),
        mtime=100.0,
    )

    monkeypatch.setattr(indexer_module, "list_eval_logs", lambda path: [stale_log])
    monkeypatch.setattr(
        indexer_module,
        "read_eval_log",
        lambda log_info, header_only=True: SimpleNamespace(
            eval=SimpleNamespace(
                task=modules["GENERATION_TASK_NAME"],
                model="model/A",
                metadata={
                    modules["TOURNAMENT_PHASE_KEY"]: modules["GENERATION_PHASE"],
                    modules["TOURNAMENT_PROJECT_KEY"]: stale_project_id,
                },
            )
        ),
    )
    monkeypatch.setattr(
        indexer_module,
        "read_eval_log_samples",
        lambda log_info, all_samples_required=False: [
            SimpleNamespace(
                metadata={"prompt_id": "prompt-1"},
                id="prompt-1",
                output=SimpleNamespace(completion="stale completion"),
                uuid="uuid-stale",
            )
        ],
    )

    report = modules["index_generation_responses"](updated)

    assert report.logs_seen == 1
    assert report.logs_processed == 0
    assert report.logs_skipped_provenance == 1
    assert report.responses_inserted == 0
    assert report.missing_by_model == {
        "model/A": ["prompt-1"],
        "model/B": ["prompt-1"],
    }
