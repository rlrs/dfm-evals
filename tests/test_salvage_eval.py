from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path


def load_salvage_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "lumi" / "salvage_eval.py"
    )
    spec = importlib.util.spec_from_file_location("salvage_eval", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_eval_zip(path: Path, members: dict[str, dict]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, json.dumps(payload))


def test_salvage_uses_header_results_for_structured_scores(tmp_path: Path) -> None:
    salvage_eval = load_salvage_module().salvage_eval
    eval_path = tmp_path / "ifeval.eval"

    write_eval_zip(
        eval_path,
        {
            "_journal/start.json": {
                "eval": {
                    "created": "2026-04-18T15:46:21+00:00",
                    "task": "inspect_evals/ifeval",
                    "task_display_name": "ifeval",
                    "task_id": "task123",
                    "model": "vllm/Qwen/Qwen3.5-9B",
                    "dataset": {"samples": 541, "sample_ids": [1]},
                }
            },
            "header.json": {
                "results": {
                    "total_samples": 250,
                    "completed_samples": 250,
                    "scores": [
                        {
                            "name": "instruction_following",
                            "scored_samples": 250,
                            "unscored_samples": 0,
                            "metrics": {
                                "prompt_strict_acc": {"value": 0.848},
                                "prompt_loose_acc": {"value": 0.884},
                                "inst_strict_acc": {"value": 0.888},
                                "inst_loose_acc": {"value": 0.920},
                                "final_acc": {"value": 0.885},
                                "final_stderr": {"value": 0.022},
                            },
                        }
                    ],
                }
            },
            "_journal/summaries/1.json": [],
            "samples/sample_1.json": {
                "id": 1,
                "epoch": 1,
                "scores": {
                    "instruction_following": {
                        "value": {
                            "prompt_level_strict": True,
                            "inst_level_strict": 1,
                            "prompt_level_loose": True,
                            "inst_level_loose": 1,
                            "num_instructions": 1,
                        }
                    }
                },
            },
        },
    )

    recovered = salvage_eval(eval_path)

    assert recovered["results_total_samples"] == 250
    assert recovered["results_completed_samples"] == 250
    assert recovered["scorers"]["instruction_following"]["metrics"]["final_acc"] == 0.885
    assert recovered["scorers"]["instruction_following"]["accuracy"] == 0.885
    assert recovered["scorers"]["instruction_following"]["stderr"] == 0.022


def test_salvage_falls_back_to_scalar_sample_scores_without_header_results(
    tmp_path: Path,
) -> None:
    salvage_eval = load_salvage_module().salvage_eval
    eval_path = tmp_path / "gsm8k.eval"

    write_eval_zip(
        eval_path,
        {
            "_journal/start.json": {
                "eval": {
                    "created": "2026-04-18T15:44:27+00:00",
                    "task": "inspect_evals/gsm8k",
                    "task_display_name": "gsm8k",
                    "task_id": "task456",
                    "model": "vllm/Qwen/Qwen3.5-9B",
                    "dataset": {"samples": 1319, "sample_ids": ["a", "b"]},
                }
            },
            "samples/a.json": {"id": "a", "epoch": 1, "scores": {"match": {"value": "C"}}},
            "samples/b.json": {"id": "b", "epoch": 1, "scores": {"match": {"value": "I"}}},
        },
    )

    recovered = salvage_eval(eval_path)

    assert recovered["results_total_samples"] is None
    assert recovered["scorers"]["match"]["accuracy"] == 0.5
    assert recovered["scorers"]["match"]["scored_samples"] == 2


def test_is_incomplete_eval_treats_successful_header_as_complete(tmp_path: Path) -> None:
    module = load_salvage_module()
    eval_path = tmp_path / "complete.eval"

    write_eval_zip(
        eval_path,
        {
            "_journal/start.json": {
                "eval": {
                    "created": "2026-04-18T15:46:21+00:00",
                    "task": "inspect_evals/ifeval",
                    "task_display_name": "ifeval",
                    "task_id": "task123",
                    "model": "vllm/Qwen/Qwen3.5-9B",
                    "dataset": {"samples": 541, "sample_ids": [1]},
                }
            },
            "header.json": {
                "status": "success",
                "results": {
                    "total_samples": 250,
                    "completed_samples": 250,
                    "scores": [],
                },
            },
        },
    )

    assert module.is_incomplete_eval(eval_path) is False
