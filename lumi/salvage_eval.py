#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any


def score_value_to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"c", "correct", "yes", "true"}:
            return 1.0
        if lowered in {"p", "partial"}:
            return 0.5
        if lowered in {"i", "incorrect", "n", "noanswer", "no", "false"}:
            return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def stderr(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return math.sqrt(variance) / math.sqrt(n)


def compact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    scores = sample.get("scores") or {}
    compact_scores: dict[str, Any] = {}
    for scorer_name, score in scores.items():
        if isinstance(score, dict):
            compact_scores[scorer_name] = score.get("value")
        else:
            compact_scores[scorer_name] = None
    error = sample.get("error")
    error_message = error.get("message") if isinstance(error, dict) else error
    return {
        "id": sample.get("id"),
        "epoch": sample.get("epoch"),
        "started_at": sample.get("started_at"),
        "completed_at": sample.get("completed_at"),
        "total_time": sample.get("total_time"),
        "working_time": sample.get("working_time"),
        "scores": compact_scores,
        "error": error_message,
    }


def load_json_member(zf: zipfile.ZipFile, member: str) -> dict[str, Any]:
    return json.loads(zf.read(member))


def scorers_from_header_results(header: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    results = header.get("results") or {}
    scores = results.get("scores") or []
    scorers: dict[str, Any] = {}

    for score_entry in scores:
        if not isinstance(score_entry, dict):
            continue

        scorer_name = score_entry.get("name") or score_entry.get("scorer")
        if not scorer_name:
            continue

        metrics = score_entry.get("metrics") or {}
        metric_values: dict[str, Any] = {}
        for metric_name, metric_info in metrics.items():
            if isinstance(metric_info, dict) and "value" in metric_info:
                metric_values[metric_name] = metric_info["value"]

        scorer_summary: dict[str, Any] = {
            "scored_samples": score_entry.get("scored_samples"),
            "unscored_terminal_samples": score_entry.get("unscored_samples"),
            "metrics": metric_values,
        }

        if "accuracy" in metric_values:
            scorer_summary["accuracy"] = metric_values["accuracy"]
        elif "mean" in metric_values:
            scorer_summary["accuracy"] = metric_values["mean"]
        elif "final_acc" in metric_values:
            scorer_summary["accuracy"] = metric_values["final_acc"]

        if "stderr" in metric_values:
            scorer_summary["stderr"] = metric_values["stderr"]
        elif "mean_stderr" in metric_values:
            scorer_summary["stderr"] = metric_values["mean_stderr"]
        elif "final_stderr" in metric_values:
            scorer_summary["stderr"] = metric_values["final_stderr"]

        scorers[scorer_name] = scorer_summary

    results_summary = {
        "total_samples": results.get("total_samples"),
        "completed_samples": results.get("completed_samples"),
    }
    return scorers, results_summary


def is_incomplete_eval(eval_path: Path) -> bool:
    try:
        with zipfile.ZipFile(eval_path) as zf:
            names = set(zf.namelist())
            if "header.json" not in names:
                return True

            header = load_json_member(zf, "header.json")
            if header.get("status") != "success":
                return True

            if not header.get("results"):
                return True

            return False
    except zipfile.BadZipFile:
        return False


def salvage_eval(eval_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(eval_path) as zf:
        names = set(zf.namelist())
        start = load_json_member(zf, "_journal/start.json")
        header = load_json_member(zf, "header.json") if "header.json" in names else {}

        sample_entries: list[dict[str, Any]] = []
        for name in sorted(names):
            if name.startswith("samples/") and name.endswith(".json"):
                sample_entries.append(load_json_member(zf, name))

        eval_spec = start["eval"]
        dataset = eval_spec.get("dataset") or {}
        expected_ids = dataset.get("sample_ids") or []
        terminal_ids = [sample.get("id") for sample in sample_entries]
        missing_ids = [sample_id for sample_id in expected_ids if sample_id not in terminal_ids]

        scorer_values: dict[str, list[float]] = {}
        errors: list[dict[str, Any]] = []

        for sample in sample_entries:
            sample_id = sample.get("id")
            scores = sample.get("scores") or {}
            for scorer_name, score in scores.items():
                if not isinstance(score, dict) or "value" not in score:
                    continue
                scorer_values.setdefault(scorer_name, []).append(
                    score_value_to_float(score["value"])
                )

            error = sample.get("error")
            if error:
                message = error.get("message") if isinstance(error, dict) else str(error)
                errors.append({"id": sample_id, "message": message})

        if header.get("results"):
            scorers, results_summary = scorers_from_header_results(header)
        else:
            scorers = {}
            for scorer_name, values in scorer_values.items():
                mean = sum(values) / len(values) if values else 0.0
                scorers[scorer_name] = {
                    "scored_samples": len(values),
                    "unscored_terminal_samples": len(sample_entries) - len(values),
                    "accuracy": mean,
                    "stderr": stderr(values),
                    "nonzero_scores": sum(value > 0 for value in values),
                    "perfect_scores": sum(value == 1.0 for value in values),
                }
            results_summary = {
                "total_samples": None,
                "completed_samples": None,
            }

        return {
            "created": eval_spec.get("created"),
            "dataset_total_samples": dataset.get("samples"),
            "errors": errors,
            "has_consolidated_summaries": "summaries.json" in names,
            "has_header_json": "header.json" in names,
            "has_results_json": "results.json" in names,
            "log_path": str(eval_path),
            "missing_sample_ids": missing_ids,
            "missing_samples": len(missing_ids),
            "model": eval_spec.get("model"),
            "results_completed_samples": results_summary.get("completed_samples"),
            "results_total_samples": results_summary.get("total_samples"),
            "samples": [compact_sample(sample) for sample in sample_entries],
            "scorers": scorers,
            "summary_checkpoints": len(
                [name for name in names if name.startswith("_journal/summaries/")]
            ),
            "task": eval_spec.get("task"),
            "task_display_name": eval_spec.get("task_display_name"),
            "task_id": eval_spec.get("task_id"),
            "terminal_errors": len(errors),
            "terminal_samples_written": len(sample_entries),
        }


def write_salvage(eval_path: Path) -> Path:
    recovered = salvage_eval(eval_path)
    output_path = eval_path.with_suffix(eval_path.suffix + ".partial-results.json")
    output_path.write_text(json.dumps(recovered, indent=2, sort_keys=True) + "\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover partial results from incomplete Inspect .eval logs."
    )
    parser.add_argument("paths", nargs="+", type=Path, help=".eval file(s) or log dir(s)")
    parser.add_argument(
        "--include-complete",
        action="store_true",
        help="Also rewrite sidecars for complete .eval files.",
    )
    return parser.parse_args()


def iter_eval_paths(paths: list[Path]) -> list[Path]:
    eval_paths: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved.is_dir():
            eval_paths.extend(sorted(resolved.glob("*.eval")))
        elif resolved.suffix == ".eval":
            eval_paths.append(resolved)
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in eval_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def main() -> int:
    args = parse_args()
    eval_paths = iter_eval_paths(args.paths)
    if not eval_paths:
        print("No .eval files found.")
        return 0

    salvaged = 0
    skipped = 0
    for eval_path in eval_paths:
        if not args.include_complete and not is_incomplete_eval(eval_path):
            skipped += 1
            continue
        output_path = write_salvage(eval_path)
        salvaged += 1
        print(f"Recovered partial results: {output_path}")

    print(f"Salvaged: {salvaged}, skipped complete: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
