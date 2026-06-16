from __future__ import annotations

from pathlib import Path

from dfm_evals import _registry
from dfm_evals._exports import (
    DISCOVERED_TASK_MODULES,
    REGISTRY_MODULES,
    SCORER_EXPORTS,
    SCORER_PACKAGE_EXPORTS,
    TASK_EXPORTS,
    discover_task_modules,
)
from dfm_evals.scorers import __all__ as scorer_exports
from dfm_evals.tasks import __all__ as task_exports


def test_load_registry_imports_manifest_modules(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        _registry,
        "import_module",
        lambda name: calls.append(name) or object(),
    )
    monkeypatch.setattr(
        _registry,
        "install_hf_eval_yaml_extensions",
        lambda: calls.append("install"),
    )

    _registry._load_registry()

    assert calls == [*REGISTRY_MODULES, "install"]


def test_task_exports_follow_manifest() -> None:
    assert task_exports == list(TASK_EXPORTS)


def test_scorer_exports_follow_manifest() -> None:
    assert scorer_exports[: len(SCORER_EXPORTS)] == list(SCORER_EXPORTS)
    assert scorer_exports == list(SCORER_PACKAGE_EXPORTS)


def test_discovered_task_modules_match_repo_layout() -> None:
    expected_subset = {
        "dfm_evals.tasks.bfcl.bfcl",
        "dfm_evals.tasks.daisy",
        "dfm_evals.tasks.dala",
        "dfm_evals.tasks.danish_citizen_tests",
        "dfm_evals.tasks.gec_dala",
        "dfm_evals.tasks.ifeval_da",
        "dfm_evals.tasks.multi_wiki_qa",
        "dfm_evals.tasks.piqa",
        "dfm_evals.tasks.ruler.task",
        "dfm_evals.tasks.talemaader.task",
    }

    assert expected_subset.issubset(DISCOVERED_TASK_MODULES)


def test_discover_task_modules_supports_flat_and_packaged_tasks(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("", encoding="utf-8")
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "_helpers.py").write_text("", encoding="utf-8")

    beta_dir = tmp_path / "beta"
    beta_dir.mkdir()
    (beta_dir / "task.py").write_text("", encoding="utf-8")
    (beta_dir / "helper.py").write_text("", encoding="utf-8")

    gamma_dir = tmp_path / "gamma"
    gamma_dir.mkdir()
    (gamma_dir / "gamma.py").write_text("", encoding="utf-8")

    zeta_dir = tmp_path / "zeta"
    zeta_dir.mkdir()
    (zeta_dir / "__init__.py").write_text("", encoding="utf-8")

    ignored_dir = tmp_path / "ignored"
    ignored_dir.mkdir()
    (ignored_dir / "helper.py").write_text("", encoding="utf-8")

    assert discover_task_modules(root_package="fake.tasks", tasks_dir=tmp_path) == (
        "fake.tasks.alpha",
        "fake.tasks.beta.task",
        "fake.tasks.gamma.gamma",
        "fake.tasks.zeta",
    )
