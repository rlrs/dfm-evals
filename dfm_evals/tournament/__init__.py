from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CanonicalDecision": ".types",
    "Decision": ".types",
    "InvalidPolicy": ".types",
    "ModelRating": ".types",
    "GenerationRunResult": ".generation",
    "ExportResult": ".exports",
    "PromptResponsesExportResult": ".exports",
    "JudgeMatch": ".judge_task",
    "JudgeRunResult": ".judge_task",
    "AddModelsResult": ".orchestrator",
    "RegisterModelsResult": ".orchestrator",
    "UpdateConfigResult": ".orchestrator",
    "TournamentRunResult": ".orchestrator",
    "TournamentStatus": ".orchestrator",
    "MatchOutcome": ".rating",
    "ModelStanding": ".rating",
    "ResponseIndexReport": ".indexer",
    "RatingUpdateResult": ".rating",
    "ScheduleBatchResult": ".scheduler",
    "ScheduledMatch": ".scheduler",
    "AdjacentCheck": ".stopping",
    "ConvergenceCheck": ".stopping",
    "HardStopCheck": ".stopping",
    "ParsedJudgeDecision": ".scorer",
    "TournamentConfig": ".config",
    "TournamentPrompt": ".config",
    "TournamentStore": ".store",
    "TournamentViewDataSource": ".viewer",
    "TournamentViewExportResult": ".viewer",
    "TournamentViewServer": ".viewer",
    "TrueSkillRatingParams": ".config",
    "build_generation_task": ".generation",
    "build_judge_samples": ".judge_task",
    "build_judge_task": ".judge_task",
    "canonicalize_side_decision": ".scorer",
    "check_convergence": ".stopping",
    "check_hard_stops": ".stopping",
    "check_hard_stops_for_config": ".stopping",
    "create_tournament_view_server": ".viewer",
    "export_tournament_view_html": ".viewer",
    "decision_valid_rate": ".scorer",
    "deterministic_id": ".types",
    "index_generation_responses": ".indexer",
    "initialize_tournament_store": ".store",
    "judge_noop_solver": ".judge_task",
    "load_tournament_config": ".config",
    "match_id": ".types",
    "model_id": ".types",
    "pairwise_judge": ".scorer",
    "parse_judge_decision": ".scorer",
    "probability_higher": ".stopping",
    "export_rankings": ".exports",
    "export_prompt_responses": ".exports",
    "reconcile_side_swap": ".scorer",
    "render_judge_prompt": ".scorer",
    "schedule_match_batch": ".scheduler",
    "serve_tournament_view": ".viewer",
    "summarize_ratings": ".rating",
    "apply_outcomes": ".rating",
    "apply_outcomes_to_store": ".rating",
    "run_generation": ".generation",
    "run_judge_batch": ".judge_task",
    "TrueSkillEngine": ".rating",
    "response_id": ".types",
    "resume_tournament": ".orchestrator",
    "run_tournament": ".orchestrator",
    "add_models": ".orchestrator",
    "register_models": ".orchestrator",
    "tournament_status": ".orchestrator",
    "update_tournament_config": ".orchestrator",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
