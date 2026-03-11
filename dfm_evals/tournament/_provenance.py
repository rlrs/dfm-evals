import sqlite3
from pathlib import Path

from .config import TournamentConfig
from .store import TournamentStore
from .types import default_project_id

TOURNAMENT_PHASE_KEY = "inspect_ai:tournament_phase"
TOURNAMENT_PROJECT_KEY = "inspect_ai:tournament_project_id"
GENERATION_PHASE = "generation"
GENERATION_TASK_NAME = "tournament_generation"
RUN_STATE_PROJECT_ID_KEY = "project_id"


def resolve_tournament_project_id(
    config: TournamentConfig,
    *,
    store: TournamentStore | None = None,
) -> str:
    """Resolve the stable project id for a tournament.

    Persisted state wins over a recomputed default so operations like
    `add-model` continue to write logs under the original tournament identity.
    An explicit config `project_id` must agree with persisted state if both are
    present.
    """
    configured_project_id = _normalized_project_id(config.project_id)
    persisted_project_id = _persisted_project_id(config, store=store)

    if (
        persisted_project_id is not None
        and configured_project_id is not None
        and persisted_project_id != configured_project_id
    ):
        raise ValueError(
            "Configured project_id does not match persisted tournament state: "
            + f"{configured_project_id!r} != {persisted_project_id!r}"
        )

    if persisted_project_id is not None:
        return persisted_project_id
    if configured_project_id is not None:
        return configured_project_id

    return default_project_id(
        config.contestant_models,
        config.prompts,
        seed=config.seed,
    )


def generation_log_metadata(
    config: TournamentConfig,
    *,
    store: TournamentStore | None = None,
) -> dict[str, str]:
    """Build eval metadata used for tournament generation logs."""
    return {
        TOURNAMENT_PHASE_KEY: GENERATION_PHASE,
        TOURNAMENT_PROJECT_KEY: resolve_tournament_project_id(config, store=store),
    }


def _persisted_project_id(
    config: TournamentConfig,
    *,
    store: TournamentStore | None,
) -> str | None:
    if store is not None:
        persisted = store.get_run_state(RUN_STATE_PROJECT_ID_KEY)
        normalized = _normalized_project_id(persisted)
        if normalized is not None:
            return normalized

    return _state_run_value(config.state_dir, RUN_STATE_PROJECT_ID_KEY)


def _state_run_value(path: Path, key: str) -> str | None:
    db_path = path if path.suffix == ".db" else path / "tournament.db"
    if not db_path.exists() or not db_path.is_file():
        return None

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT value FROM run_state WHERE key = ?",
            (key,),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()

    if row is None:
        return None
    return _normalized_project_id(str(row[0]))


def _normalized_project_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized != "" else None
