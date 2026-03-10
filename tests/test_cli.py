from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from dfm_evals import cli


def test_packaged_suites_include_openthoughts_tblite() -> None:
    suites = cli._load_named_suites(cli.DEFAULT_SUITES_FILE)

    suite = suites["openthoughts_tblite"]

    assert [task.name for task in suite.tasks] == ["inspect_harbor/openthoughts_tblite"]
    assert suite.args == [
        "--model",
        "{{target_model}}",
        "--no-fail-on-error",
        "--continue-on-fail",
        "--limit",
        "100",
        "--temperature",
        "0",
    ]


def test_optional_registry_import_is_ignored_when_package_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_import_module(name: str) -> object:
        calls.append(name)
        if name == "inspect_sandboxes._registry":
            raise ModuleNotFoundError("No module named 'inspect_sandboxes'", name="inspect_sandboxes")
        return object()

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    cli._ensure_registry_modules_loaded()

    assert calls == [
        "dfm_evals._registry",
        "inspect_sandboxes._registry",
        "inspect_harbor._registry",
    ]


def test_optional_registry_import_propagates_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> object:
        if name == "inspect_sandboxes._registry":
            raise ModuleNotFoundError("No module named 'modal'", name="modal")
        return object()

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="modal"):
        cli._ensure_registry_modules_loaded()


def test_load_model_info_overrides_parses_context_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DFM_EVALS_MODEL_INFO_OVERRIDES",
        (
            '{"vllm/custom-model":{"context_length":8192,"output_tokens":2048,'
            '"display_name":"Custom Model","organization":"DFM"},'
            '"vllm/another-model":4096}'
        ),
    )

    assert cli._load_model_info_overrides() == {
        "vllm/custom-model": {
            "context_length": 8192,
            "output_tokens": 2048,
            "model": "Custom Model",
            "organization": "DFM",
        },
        "vllm/another-model": {"context_length": 4096},
    }


def test_apply_model_info_overrides_registers_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DFM_EVALS_MODEL_INFO_OVERRIDES",
        '{"vllm/custom-model":{"context_length":8192}}',
    )

    calls: list[tuple[str, dict[str, object]]] = []

    class FakeModelInfo:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_module = SimpleNamespace(
        ModelInfo=FakeModelInfo,
        set_model_info=lambda model_name, info: calls.append((model_name, info.kwargs)),
    )
    monkeypatch.setitem(sys.modules, "inspect_ai.model", fake_module)

    cli._apply_model_info_overrides()

    assert calls == [("vllm/custom-model", {"context_length": 8192})]
