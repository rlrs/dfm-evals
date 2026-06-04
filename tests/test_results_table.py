from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path


def _write_eee_record(
    path: Path,
    *,
    model_id: str,
    model_name: str | None = None,
    gleu_score: float,
    exact_score: float | None,
    evaluation_timestamp: str = "1234.0",
    retrieved_timestamp: str = "1235.0",
    task_name: str = "gec_dala",
    benchmark_name: str = "demo",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_results = [
        {
            "evaluation_name": f"{task_name}/gleu/mean",
            "source_data": {
                "dataset_name": benchmark_name,
                "source_type": "other",
            },
            "metric_config": {
                "lower_is_better": False,
                "score_type": "continuous",
                "min_score": 0.0,
                "max_score": 1.0,
                "additional_details": {
                    "preferred_for_display": "true",
                },
            },
            "score_details": {
                "score": gleu_score,
                "details": {
                    "task": task_name,
                    "scorer": "gleu",
                    "metric": "mean",
                },
            },
        },
    ]
    if exact_score is not None:
        evaluation_results.append(
            {
                "evaluation_name": f"{task_name}/exact/mean",
                "source_data": {
                    "dataset_name": benchmark_name,
                    "source_type": "other",
                },
                "metric_config": {
                    "lower_is_better": False,
                    "score_type": "continuous",
                    "min_score": 0.0,
                    "max_score": 1.0,
                },
                "score_details": {
                    "score": exact_score,
                    "details": {
                        "task": task_name,
                        "scorer": "exact",
                        "metric": "mean",
                    },
                },
            }
        )

    record = {
        "schema_version": "0.2.1",
        "evaluation_id": f"{task_name}/{model_id}/{evaluation_timestamp}",
        "evaluation_timestamp": evaluation_timestamp,
        "retrieved_timestamp": retrieved_timestamp,
        "source_metadata": {
            "source_name": "inspect_ai",
            "source_type": "evaluation_run",
            "source_organization_name": "test",
            "evaluator_relationship": "third_party",
        },
        "eval_library": {
            "name": "inspect_ai",
            "version": "0.1.0",
        },
        "model_info": {
            "name": model_name or model_id,
            "id": model_id,
        },
        "evaluation_results": evaluation_results,
    }
    path.write_text(json.dumps(record), encoding="utf-8")


def _write_multi_wiki_qa_record(
    path: Path,
    *,
    model_id: str,
    exact_match_score: float,
    f1_score: float,
    task_name: str = "multi_wiki_qa",
    evaluation_timestamp: str = "1234.0",
    retrieved_timestamp: str = "1235.0",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_results = []
    for scorer, score in (("exact_match", exact_match_score), ("f1", f1_score)):
        evaluation_results.append(
            {
                "evaluation_name": f"{task_name}/{scorer}/mean",
                "source_data": {
                    "dataset_name": "MultiWikiQA-da",
                    "source_type": "hf_dataset",
                },
                "metric_config": {
                    "lower_is_better": False,
                    "score_type": "continuous",
                    "min_score": 0.0,
                    "max_score": 1.0,
                },
                "score_details": {
                    "score": score,
                    "details": {
                        "task": task_name,
                        "scorer": scorer,
                        "metric": "mean",
                    },
                },
            }
        )

    record = {
        "schema_version": "0.2.1",
        "evaluation_id": f"{task_name}/{model_id}/{evaluation_timestamp}",
        "evaluation_timestamp": evaluation_timestamp,
        "retrieved_timestamp": retrieved_timestamp,
        "source_metadata": {
            "source_name": "inspect_ai",
            "source_type": "evaluation_run",
            "source_organization_name": "test",
            "evaluator_relationship": "third_party",
        },
        "eval_library": {
            "name": "inspect_ai",
            "version": "0.1.0",
        },
        "model_info": {
            "name": model_id,
            "id": model_id,
        },
        "evaluation_results": evaluation_results,
    }
    path.write_text(json.dumps(record), encoding="utf-8")


def test_results_table_uses_gec_dala_exact_even_when_gleu_is_marked(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    _write_eee_record(
        data_root / "demo-benchmark" / "org" / "model-a" / "record.json",
        model_id="org/model-a",
        gleu_score=0.42,
        exact_score=0.18,
    )
    _write_eee_record(
        data_root / "demo-benchmark" / "org" / "model-b" / "record.json",
        model_id="org/model-b",
        gleu_score=0.55,
        exact_score=0.22,
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["FORMAT"] = "csv"

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
            "--task-rows",
            "--format",
            "csv",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines[0].startswith("task,scorer,metric,")
    assert "gec_dala,exact,mean,0.18,0.22" in lines


def test_results_table_uses_multi_wiki_qa_f1_as_primary(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    _write_multi_wiki_qa_record(
        data_root / "MultiWikiQA-da" / "org" / "model-a" / "record.json",
        model_id="org/model-a",
        exact_match_score=0.52,
        f1_score=0.706,
    )
    _write_multi_wiki_qa_record(
        data_root / "MultiWikiQA-da" / "org" / "model-b" / "record.json",
        model_id="org/model-b",
        exact_match_score=0.45,
        f1_score=0.739,
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["FORMAT"] = "csv"

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
            "--task-rows",
            "--format",
            "csv",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines[0].startswith("task,scorer,metric,")
    assert "multi_wiki_qa,f1,mean,0.706,0.739" in lines
    assert all("multi_wiki_qa,exact_match,mean" not in line for line in lines)


def test_results_table_gec_dala_missing_exact_stays_blank(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    _write_eee_record(
        data_root / "demo-benchmark" / "org" / "model-a" / "record.json",
        model_id="org/model-a",
        gleu_score=0.42,
        exact_score=0.18,
    )
    _write_eee_record(
        data_root / "demo-benchmark" / "org" / "model-b" / "record.json",
        model_id="org/model-b",
        gleu_score=0.55,
        exact_score=None,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["FORMAT"] = "csv"

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
            "--task-rows",
            "--format",
            "csv",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines[0] == "task,scorer,metric,org/model-a,org/model-b"
    assert "gec_dala,exact,mean,0.18," in lines
    assert all("gec_dala,gleu,mean" not in line for line in lines)


def test_results_table_normalizes_legacy_local_path_model_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    _write_eee_record(
        data_root / "demo-benchmark" / "unknown" / "flash" / "record.json",
        model_id="/flash",
        model_name="vllm//flash/project_465002183/trl-runs/hermes-4n-full-20260306-lr3e5-warmup50/final",
        gleu_score=0.42,
        exact_score=0.18,
        evaluation_timestamp="1234.0",
        retrieved_timestamp="1235.0",
    )
    _write_eee_record(
        data_root / "demo-benchmark" / "unknown" / "pfs" / "record.json",
        model_id="/pfs",
        model_name="vllm//pfs/lustref1/flash/project_465002183/trl-runs/hermes-4n-full-20260306-lr3e5-warmup50/final",
        gleu_score=0.55,
        exact_score=0.22,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["FORMAT"] = "csv"

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
            "--task-rows",
            "--format",
            "csv",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines[0] == "task,scorer,metric,local/hermes-4n-full-20260306-lr3e5-warmup50-final"
    assert "gec_dala,exact,mean,0.22" in lines


def test_results_table_keeps_ruler_lengths_as_distinct_task_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    _write_eee_record(
        data_root / "RULER-vt_8k" / "google" / "gemma-3-4b-it" / "vt-8k.json",
        model_id="google/gemma-3-4b-it",
        gleu_score=0.75,
        exact_score=0.0,
        task_name="RULER-vt@8k",
        benchmark_name="RULER-vt@8k",
    )
    _write_eee_record(
        data_root / "RULER-vt_32k" / "google" / "gemma-3-4b-it" / "vt-32k.json",
        model_id="google/gemma-3-4b-it",
        gleu_score=0.62,
        exact_score=0.0,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
        task_name="RULER-vt@32k",
        benchmark_name="RULER-vt@32k",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["FORMAT"] = "csv"

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
            "--task-rows",
            "--format",
            "csv",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines[0] == "task,scorer,metric,google/gemma-3-4b-it"
    assert "RULER-vt@32k,gleu,mean,0.62" in lines
    assert "RULER-vt@8k,gleu,mean,0.75" in lines


def test_results_table_writes_sortable_compare_models_html(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    html_root = tmp_path / "html"
    _write_eee_record(
        data_root / "demo-benchmark" / "org" / "model-a" / "record.json",
        model_id="Qwen/Qwen3.5-4B",
        gleu_score=0.42,
        exact_score=0.18,
    )
    _write_eee_record(
        data_root / "demo-benchmark" / "org" / "model-b" / "record.json",
        model_id="Qwen/Qwen3.5-9B",
        gleu_score=0.55,
        exact_score=0.22,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
    )
    _write_eee_record(
        data_root / "demo-benchmark" / "org" / "model-c" / "record.json",
        model_id="google/gemma-3-4b-it",
        gleu_score=0.40,
        exact_score=0.20,
        evaluation_timestamp="3234.0",
        retrieved_timestamp="3235.0",
    )
    _write_eee_record(
        data_root / "demo-benchmark" / "org" / "model-d" / "record.json",
        model_id="google/gemma-3-9b-it",
        gleu_score=0.50,
        exact_score=0.24,
        evaluation_timestamp="4234.0",
        retrieved_timestamp="4235.0",
    )
    _write_eee_record(
        data_root / "english-benchmark" / "org" / "model-a" / "record.json",
        model_id="Qwen/Qwen3.5-4B",
        gleu_score=0.72,
        exact_score=0.58,
        evaluation_timestamp="5234.0",
        retrieved_timestamp="5235.0",
        task_name="gsm8k",
        benchmark_name="english-benchmark",
    )
    _write_eee_record(
        data_root / "english-benchmark" / "org" / "model-b" / "record.json",
        model_id="Qwen/Qwen3.5-9B",
        gleu_score=0.81,
        exact_score=0.64,
        evaluation_timestamp="6234.0",
        retrieved_timestamp="6235.0",
        task_name="gsm8k",
        benchmark_name="english-benchmark",
    )
    _write_eee_record(
        data_root / "english-benchmark" / "org" / "model-c" / "record.json",
        model_id="google/gemma-3-4b-it",
        gleu_score=0.70,
        exact_score=0.56,
        evaluation_timestamp="7234.0",
        retrieved_timestamp="7235.0",
        task_name="gsm8k",
        benchmark_name="english-benchmark",
    )
    _write_eee_record(
        data_root / "english-benchmark" / "org" / "model-d" / "record.json",
        model_id="google/gemma-3-9b-it",
        gleu_score=0.76,
        exact_score=0.60,
        evaluation_timestamp="8234.0",
        retrieved_timestamp="8235.0",
        task_name="gsm8k",
        benchmark_name="english-benchmark",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["RESULTS_TABLE_HTML_DIR"] = str(html_root)
    env["RESULTS_TABLE_AUTO_OPEN"] = "0"
    env["RESULTS_TABLE_KEEP_OPEN"] = "0"
    env["RESULTS_TABLE_HOST"] = "127.0.0.1"
    env["RESULTS_TABLE_BIND_HOST"] = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        env["RESULTS_TABLE_PORT"] = str(sock.getsockname()[1])

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    url = completed.stdout.strip()
    assert url.startswith("http://127.0.0.1:")
    state_path = html_root / ".results-table-server.json"
    assert state_path.is_file()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    output_path = html_root / Path(urllib.parse.urlparse(url).path).name
    assert output_path.suffix == ".html"
    assert output_path.is_file()

    deadline = time.time() + 5.0
    last_exc: Exception | None = None
    document = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                document = response.read().decode("utf-8")
            break
        except Exception as exc:  # pragma: no cover - transient startup race
            last_exc = exc
            time.sleep(0.1)
    else:
        raise AssertionError(f"failed to fetch served HTML report: {last_exc}")

    assert '<table id="results-table" data-mode="relative" data-group-mode="sizefamily">' in document
    assert 'id="table-filter"' in document
    assert "Click a header to sort." in document
    assert "<h1>" not in document
    assert "all-runs · compare-models" not in document
    assert ">4B<" in document
    assert ">9B<" in document
    assert ">Size<" in document
    assert ">English<" in document
    assert ">Danish<" in document
    assert "color-scheme: light;" in document
    assert 'title="Qwen/Qwen3.5-4B"' in document
    assert "Relative vs baseline" in document
    assert '>Size+Family<' in document
    assert 'class="mode-btn active" data-mode="relative"' in document
    assert 'class="mode-btn row-group-btn active" data-row-group-mode="sizefamily"' in document
    assert 'data-rel-display-sizefamily="' in document
    assert 'title="English median"' in document
    assert 'title="Danish median"' in document
    assert 'data-rel-group="4B|qwen|english|gsm8k"' in document
    assert 'data-rel-group="9B|qwen|english|gsm8k"' in document
    assert 'class="value value-abs"' in document
    assert 'class="value value-rel"' in document

    os.kill(int(state["pid"]), signal.SIGTERM)


def test_results_table_absolute_heatmap_normalizes_within_size_and_eval(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    html_root = tmp_path / "html"
    _write_eee_record(
        data_root / "gsm8k" / "qwen" / "qwen35-4b" / "a.json",
        model_id="Qwen/Qwen3.5-4B",
        gleu_score=0.40,
        exact_score=0.10,
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "qwen" / "qwen35-4b-instruct" / "b.json",
        model_id="Qwen/Qwen3.5-4B-Instruct",
        gleu_score=0.50,
        exact_score=0.12,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "qwen" / "qwen35-9b" / "c.json",
        model_id="Qwen/Qwen3.5-9B",
        gleu_score=0.80,
        exact_score=0.20,
        evaluation_timestamp="3234.0",
        retrieved_timestamp="3235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "qwen" / "qwen35-9b-instruct" / "d.json",
        model_id="Qwen/Qwen3.5-9B-Instruct",
        gleu_score=0.90,
        exact_score=0.22,
        evaluation_timestamp="4234.0",
        retrieved_timestamp="4235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["RESULTS_TABLE_HTML_DIR"] = str(html_root)
    env["RESULTS_TABLE_AUTO_OPEN"] = "0"
    env["RESULTS_TABLE_KEEP_OPEN"] = "0"
    env["RESULTS_TABLE_HOST"] = "127.0.0.1"
    env["RESULTS_TABLE_BIND_HOST"] = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        env["RESULTS_TABLE_PORT"] = str(sock.getsockname()[1])

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    url = completed.stdout.strip()
    deadline = time.time() + 5.0
    document = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                document = response.read().decode("utf-8")
            break
        except Exception:  # pragma: no cover - transient startup race
            time.sleep(0.1)
    else:
        raise AssertionError("failed to fetch served HTML report")

    assert '--abs-bg:hsl(8.0 68.0% 88.0% / 0.82);' in document
    assert '--abs-bg:hsl(120.0 82.0% 78.0% / 0.82);' in document


def test_results_table_sizefamily_mode_uses_qwen_family_baseline(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    html_root = tmp_path / "html"
    _write_eee_record(
        data_root / "gsm8k" / "google" / "gemma-3-4b-it" / "a.json",
        model_id="google/gemma-3-4b-it",
        gleu_score=0.40,
        exact_score=0.10,
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "qwen" / "qwen35-4b" / "b.json",
        model_id="Qwen/Qwen3.5-4B",
        gleu_score=0.55,
        exact_score=0.12,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "qwen" / "qwen35-4b-instruct" / "c.json",
        model_id="Qwen/Qwen3.5-4B-Instruct",
        gleu_score=0.70,
        exact_score=0.18,
        evaluation_timestamp="3234.0",
        retrieved_timestamp="3235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["RESULTS_TABLE_HTML_DIR"] = str(html_root)
    env["RESULTS_TABLE_AUTO_OPEN"] = "0"
    env["RESULTS_TABLE_KEEP_OPEN"] = "0"
    env["RESULTS_TABLE_HOST"] = "127.0.0.1"
    env["RESULTS_TABLE_BIND_HOST"] = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        env["RESULTS_TABLE_PORT"] = str(sock.getsockname()[1])

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    url = completed.stdout.strip()
    deadline = time.time() + 5.0
    document = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                document = response.read().decode("utf-8")
            break
        except Exception:  # pragma: no cover - transient startup race
            time.sleep(0.1)
    else:
        raise AssertionError("failed to fetch served HTML report")

    assert 'data-rel-display-size="+0.30"' in document
    assert 'data-rel-display-sizefamily="+0.15"' in document
    assert (
        'data-rel-title-sizefamily="Qwen/Qwen3.5-4B-Instruct vs Qwen/Qwen3.5-4B"' in document
    )


def test_results_table_gemma4_e4b_does_not_group_with_gemma3(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    html_root = tmp_path / "html"
    _write_eee_record(
        data_root / "gsm8k" / "google" / "gemma-3-4b-it" / "a.json",
        model_id="google/gemma-3-4b-it",
        gleu_score=0.40,
        exact_score=0.10,
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "google" / "gemma-4-e4b-it" / "b.json",
        model_id="google/gemma-4-E4B-it",
        gleu_score=0.65,
        exact_score=0.18,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "local" / "gemma4-e4b-sft" / "c.json",
        model_id="local/gemma4-e4b-sft",
        gleu_score=0.80,
        exact_score=0.22,
        evaluation_timestamp="3234.0",
        retrieved_timestamp="3235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["RESULTS_TABLE_HTML_DIR"] = str(html_root)
    env["RESULTS_TABLE_AUTO_OPEN"] = "0"
    env["RESULTS_TABLE_KEEP_OPEN"] = "0"
    env["RESULTS_TABLE_HOST"] = "127.0.0.1"
    env["RESULTS_TABLE_BIND_HOST"] = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        env["RESULTS_TABLE_PORT"] = str(sock.getsockname()[1])

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    url = completed.stdout.strip()
    deadline = time.time() + 5.0
    document = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                document = response.read().decode("utf-8")
            break
        except Exception:  # pragma: no cover - transient startup race
            time.sleep(0.1)
    else:
        raise AssertionError("failed to fetch served HTML report")

    assert "4B / Gemma 3" in document
    assert "4B / Gemma 4" in document
    assert 'data-rel-group="4B|gemma3|english|gsm8k"' in document
    assert 'data-rel-group="4B|gemma4|english|gsm8k"' in document
    assert 'data-rel-display-sizefamily="+0.15"' in document
    assert (
        'data-rel-title-sizefamily="local/gemma4-e4b-sft vs google/gemma-4-E4B-it"'
        in document
    )
    assert "No Gemma 4 E4B IT baseline" not in document
    assert (
        'data-rel-title-sizefamily="google/gemma-4-E4B-it vs google/gemma-3-4b-it"'
        not in document
    )


def test_results_table_q35_local_models_group_as_qwen_family(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    html_root = tmp_path / "html"
    _write_eee_record(
        data_root / "gsm8k" / "qwen" / "qwen35-4b" / "vendor.json",
        model_id="Qwen/Qwen3.5-4B",
        gleu_score=0.55,
        exact_score=0.12,
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "local" / "q35-4b" / "a.json",
        model_id="local/q35-4B-run",
        gleu_score=0.60,
        exact_score=0.12,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "local" / "q35-4b-instruct" / "b.json",
        model_id="local/q35-4B-run-instruct",
        gleu_score=0.70,
        exact_score=0.18,
        evaluation_timestamp="3234.0",
        retrieved_timestamp="3235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["RESULTS_TABLE_HTML_DIR"] = str(html_root)
    env["RESULTS_TABLE_AUTO_OPEN"] = "0"
    env["RESULTS_TABLE_KEEP_OPEN"] = "0"
    env["RESULTS_TABLE_HOST"] = "127.0.0.1"
    env["RESULTS_TABLE_BIND_HOST"] = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        env["RESULTS_TABLE_PORT"] = str(sock.getsockname()[1])

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    url = completed.stdout.strip()
    deadline = time.time() + 5.0
    document = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                document = response.read().decode("utf-8")
            break
        except Exception:  # pragma: no cover - transient startup race
            time.sleep(0.1)
    else:
        raise AssertionError("failed to fetch served HTML report")

    assert 'data-rel-display-sizefamily="+0.15"' in document
    assert 'data-rel-title-sizefamily="local/q35-4B-run-instruct vs Qwen/Qwen3.5-4B"' in document


def test_results_table_ministral3_variants_group_on_instruct_baseline(tmp_path: Path) -> None:
    data_root = tmp_path / "eee"
    html_root = tmp_path / "html"
    _write_eee_record(
        data_root / "gsm8k" / "mistralai" / "ministral-3-8b-instruct" / "baseline.json",
        model_id="mistralai/Ministral-3-8B-Instruct-2512",
        gleu_score=0.55,
        exact_score=0.12,
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "local" / "ministram-3-8b-run" / "typo.json",
        model_id="local/ministram-3-8b-run",
        gleu_score=0.70,
        exact_score=0.18,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "local" / "ministral3-8b-run" / "compact.json",
        model_id="local/ministral3-8b-run",
        gleu_score=0.80,
        exact_score=0.20,
        evaluation_timestamp="3234.0",
        retrieved_timestamp="3235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["RESULTS_TABLE_HTML_DIR"] = str(html_root)
    env["RESULTS_TABLE_AUTO_OPEN"] = "0"
    env["RESULTS_TABLE_KEEP_OPEN"] = "0"
    env["RESULTS_TABLE_HOST"] = "127.0.0.1"
    env["RESULTS_TABLE_BIND_HOST"] = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        env["RESULTS_TABLE_PORT"] = str(sock.getsockname()[1])

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    url = completed.stdout.strip()
    deadline = time.time() + 5.0
    document = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                document = response.read().decode("utf-8")
            break
        except Exception:  # pragma: no cover - transient startup race
            time.sleep(0.1)
    else:
        raise AssertionError("failed to fetch served HTML report")

    assert "8B / Ministral 3" in document
    assert 'data-rel-group="8B|ministral3|english|gsm8k"' in document
    assert 'data-rel-display-sizefamily="+0.15"' in document
    assert 'data-rel-display-sizefamily="+0.25"' in document
    assert (
        'data-rel-title-sizefamily="local/ministram-3-8b-run vs '
        'mistralai/Ministral-3-8B-Instruct-2512"'
    ) in document
    assert (
        'data-rel-title-sizefamily="local/ministral3-8b-run vs '
        'mistralai/Ministral-3-8B-Instruct-2512"'
    ) in document
    assert "No Ministral 3 Instruct baseline" not in document


def test_results_table_apertus_data_name_does_not_override_qwen_model_family(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "eee"
    html_root = tmp_path / "html"
    _write_eee_record(
        data_root / "gsm8k" / "qwen" / "qwen35-4b" / "baseline.json",
        model_id="Qwen/Qwen3.5-4B",
        gleu_score=0.55,
        exact_score=0.12,
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "qwen" / "apertus-sft-data" / "qwen-data.json",
        model_id=(
            "qwen/qwen3.5-4b-base-chatml-hermes-swiss-ai-apertus-sft-mixture-"
            "synquid-wildchat-100k-qwen-final"
        ),
        gleu_score=0.70,
        exact_score=0.18,
        evaluation_timestamp="2234.0",
        retrieved_timestamp="2235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )
    _write_eee_record(
        data_root / "gsm8k" / "swiss-ai" / "apertus-qwen-data" / "apertus.json",
        model_id="swiss-ai/apertus-8b-2509-chatml-hermes-synquid-wildchat-100k-qwen-final",
        gleu_score=0.60,
        exact_score=0.14,
        evaluation_timestamp="3234.0",
        retrieved_timestamp="3235.0",
        task_name="gsm8k",
        benchmark_name="gsm8k",
    )

    script = Path(__file__).resolve().parents[1] / "lumi" / "results_table.sh"
    env = dict(os.environ)
    env["EEE_DATA_ROOT_HOST"] = str(data_root)
    env["RESULTS_TABLE_HTML_DIR"] = str(html_root)
    env["RESULTS_TABLE_AUTO_OPEN"] = "0"
    env["RESULTS_TABLE_KEEP_OPEN"] = "0"
    env["RESULTS_TABLE_HOST"] = "127.0.0.1"
    env["RESULTS_TABLE_BIND_HOST"] = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        env["RESULTS_TABLE_PORT"] = str(sock.getsockname()[1])

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--compare-models",
            "--all-runs",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    url = completed.stdout.strip()
    deadline = time.time() + 5.0
    document = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                document = response.read().decode("utf-8")
            break
        except Exception:  # pragma: no cover - transient startup race
            time.sleep(0.1)
    else:
        raise AssertionError("failed to fetch served HTML report")

    assert 'data-rel-display-sizefamily="+0.15"' in document
    assert (
        'data-rel-title-sizefamily="qwen/qwen3.5-4b-base-chatml-hermes-swiss-ai-apertus-sft-mixture-'
        'synquid-wildchat-100k-qwen-final vs Qwen/Qwen3.5-4B"'
    ) in document
    assert "4B / Qwen" in document
    assert "8B / Swiss-ai" in document
