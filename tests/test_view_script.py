from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _make_fake_singularity(bin_dir: Path) -> Path:
    singularity = bin_dir / "singularity"
    singularity.write_text(
        "#!/bin/bash\n"
        "printf 'FAKE_SINGULARITY:'\n"
        "printf ' %q' \"$@\"\n"
        "printf '\\n'\n",
        encoding="utf-8",
    )
    singularity.chmod(0o755)
    return singularity


def _common_env(tmp_path: Path) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    overlay_dir = tmp_path / "overlay"
    logs_dir = tmp_path / "artifacts" / "evals" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "run-a").mkdir()
    qwen_old = logs_dir / "fundamentals__qwen__job-111"
    qwen_new = logs_dir / "openthoughts_tblite__Qwen3.5-4B__job-222"
    gemma = logs_dir / "fundamentals__gemma__job-333"
    qwen_old.mkdir()
    qwen_new.mkdir()
    gemma.mkdir()
    os.utime(qwen_old, (1_700_000_000, 1_700_000_000))
    os.utime(qwen_new, (1_800_000_000, 1_800_000_000))
    os.utime(gemma, (1_750_000_000, 1_750_000_000))
    (overlay_dir / "venv" / "vllm-min").mkdir(parents=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_fake_singularity(bin_dir)

    sif = tmp_path / "fake.sif"
    sif.write_text("", encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["POST_ARTIFACT_ROOT"] = str(tmp_path / "artifacts")
    env["OVERLAY_DIR"] = str(overlay_dir)
    env["SIF"] = str(sif)
    env["HF_HOME"] = str(tmp_path / "hf-home")
    return env


def test_view_start_all_reaches_singularity(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "lumi" / "view.sh"
    env = _common_env(tmp_path)

    completed = subprocess.run(
        ["bash", str(script), "start", "--all"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    stdout = completed.stdout
    expected_log_dir = tmp_path / "artifacts" / "evals" / "logs"
    assert "Mode: start" in stdout
    assert f"Log dir: {expected_log_dir}" in stdout
    assert "Run label:" not in stdout
    assert "FAKE_SINGULARITY:" in stdout
    assert f"--log-dir {expected_log_dir}" in stdout
    assert "--recursive" not in stdout


def test_view_start_no_recursive_passes_inverted_inspect_flag(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "lumi" / "view.sh"
    env = _common_env(tmp_path)

    completed = subprocess.run(
        ["bash", str(script), "start", "--all", "--no-recursive"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    stdout = completed.stdout
    assert "Mode: start" in stdout
    assert "FAKE_SINGULARITY:" in stdout
    assert " --recursive" in stdout


def test_view_bundle_all_reaches_singularity(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "lumi" / "view.sh"
    env = _common_env(tmp_path)

    completed = subprocess.run(
        ["bash", str(script), "bundle", "--all"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    stdout = completed.stdout
    expected_log_dir = tmp_path / "artifacts" / "evals" / "logs"
    assert "Mode: bundle" in stdout
    assert f"Log dir: {expected_log_dir}" in stdout
    assert "FAKE_SINGULARITY:" in stdout
    assert f"--log-dir {expected_log_dir}" in stdout
    assert "--output-dir" in stdout


def test_view_list_filter_limits_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "lumi" / "view.sh"
    env = _common_env(tmp_path)

    completed = subprocess.run(
        ["bash", str(script), "list", "qwen"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    stdout = completed.stdout
    assert "matching 'qwen'" in stdout
    assert "openthoughts_tblite__Qwen3.5-4B__job-222" in stdout
    assert "fundamentals__qwen__job-111" in stdout
    assert "fundamentals__gemma__job-333" not in stdout


def test_view_start_bare_query_uses_newest_match(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "lumi" / "view.sh"
    env = _common_env(tmp_path)

    completed = subprocess.run(
        ["bash", str(script), "start", "qwen"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    stdout = completed.stdout
    expected_log_dir = (
        tmp_path
        / "artifacts"
        / "evals"
        / "logs"
        / "openthoughts_tblite__Qwen3.5-4B__job-222"
    )
    assert "Match query: qwen" in stdout
    assert "Match kind: substring" in stdout
    assert "Matched runs: 2 (using newest)" in stdout
    assert f"Log dir: {expected_log_dir}" in stdout
    assert "Run label: openthoughts_tblite__Qwen3.5-4B__job-222" in stdout
