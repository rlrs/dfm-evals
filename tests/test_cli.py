from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dfm_evals import cli


def test_packaged_suites_include_openthoughts_tblite() -> None:
    suites = cli._load_named_suites(cli.DEFAULT_SUITES_FILE)

    suite = suites["openthoughts_tblite"]

    assert [task.name for task in suite.tasks] == ["inspect_harbor/openthoughts_tblite"]
    assert suite.tasks[0].args == [
        "-T",
        "sandbox_env_name=modal",
    ]
    assert suite.args == [
        "--model",
        "{{target_model}}",
        "--no-fail-on-error",
        "--continue-on-fail",
        "--message-limit",
        "100",
        "--limit",
        "250",
        "--sample-shuffle",
        "4242",
        "--temperature",
        "0",
    ]


def test_packaged_suites_include_ruler_smoke() -> None:
    suites = cli._load_named_suites(cli.DEFAULT_SUITES_FILE)

    suite = suites["ruler_smoke"]

    assert [task.name for task in suite.tasks] == ["ruler", "ruler"]
    assert suite.tasks[0].args == [
        "-T",
        "variant=niah_single_1",
        "-T",
        "num_samples=4",
        "-T",
        "max_seq_length=4096",
        "-T",
        "tokenizer_backend=hf",
        "-T",
        "tokenizer_model=google/gemma-3-4b-it",
    ]
    assert suite.tasks[1].args == [
        "-T",
        "variant=vt",
        "-T",
        "num_samples=4",
        "-T",
        "max_seq_length=4096",
        "-T",
        "tokenizer_backend=hf",
        "-T",
        "tokenizer_model=google/gemma-3-4b-it",
    ]
    assert suite.args == [
        "--model",
        "{{target_model}}",
        "--limit",
        "2",
        "--sample-shuffle",
        "4242",
        "--temperature",
        "0",
    ]


def test_packaged_fundamentals_include_short_ruler_lengths() -> None:
    suites = cli._load_named_suites(cli.DEFAULT_SUITES_FILE)

    suite = suites["fundamentals"]
    ifeval_tasks = [task for task in suite.tasks if task.name in {"inspect_evals/ifeval", "ifeval-da"}]
    ruler_tasks = [task for task in suite.tasks if task.name == "ruler"]

    assert [(task.name, task.args) for task in ifeval_tasks] == [
        ("inspect_evals/ifeval", ["--max-tokens", "3072"]),
        ("ifeval-da", ["--max-tokens", "3072"]),
    ]
    assert len(ruler_tasks) == 4
    assert sorted(task.args[5] for task in ruler_tasks) == [
        "max_seq_length=32768",
        "max_seq_length=32768",
        "max_seq_length=8192",
        "max_seq_length=8192",
    ]
    assert all("tokenizer_backend=auto" in task.args for task in ruler_tasks)
    assert all("tokenizer_model={{target_model}}" in task.args for task in ruler_tasks)
    assert all(any(arg.startswith("variant=") for arg in task.args) for task in ruler_tasks)


def test_packaged_fundamentals_include_wmt24pp_translation() -> None:
    suites = cli._load_named_suites(cli.DEFAULT_SUITES_FILE)

    suite = suites["fundamentals"]
    wmt24pp_tasks = [task for task in suite.tasks if task.name == "wmt24pp-en-da"]

    assert [(task.name, task.args) for task in wmt24pp_tasks] == [
        ("wmt24pp-en-da", ["-T", "max_gen_toks=512"]),
    ]


def test_packaged_suites_include_wmt24pp_translation_suite() -> None:
    suites = cli._load_named_suites(cli.DEFAULT_SUITES_FILE)

    suite = suites["wmt24pp_en_da"]

    assert [(task.name, task.args) for task in suite.tasks] == [
        ("wmt24pp-en-da", ["-T", "max_gen_toks=512"]),
    ]
    assert suite.args == [
        "--model",
        "{{target_model}}",
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
        "inspect_sandboxes.modal._compose",
        "inspect_sandboxes.modal._modal",
    ]


def test_optional_registry_import_ignores_missing_modal_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_import_module(name: str) -> object:
        calls.append(name)
        if name == "inspect_sandboxes._registry":
            raise ModuleNotFoundError("No module named 'modal'", name="modal")
        if name.startswith("inspect_sandboxes.modal."):
            raise ModuleNotFoundError("No module named 'modal'", name="modal")
        return object()

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    cli._ensure_registry_modules_loaded()

    assert calls == [
        "dfm_evals._registry",
        "inspect_sandboxes._registry",
        "inspect_harbor._registry",
        "inspect_sandboxes.modal._compose",
    ]


def test_optional_registry_import_still_propagates_other_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> object:
        if name == "inspect_sandboxes._registry":
            raise ModuleNotFoundError("No module named 'docker'", name="docker")
        return object()

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="docker"):
        cli._ensure_registry_modules_loaded()


def test_patch_inspect_sandboxes_modal_context_dir_uses_build_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []

    class FakeImage:
        @staticmethod
        def from_dockerfile(path: str, *, context_dir: str | None = None) -> tuple[str, str, str | None]:
            calls.append((path, context_dir))
            return ("image", path, context_dir)

    original_calls: list[tuple[object, str | None]] = []

    def original_convert(config: object, compose_path: str | None) -> dict[str, object]:
        original_calls.append((config, compose_path))
        return {"image": "old-image", "cpu": 1.0}

    fake_build = SimpleNamespace(context="/tmp/harbor-task/environment", dockerfile="Dockerfile")
    fake_service = SimpleNamespace(x_default=False, build=fake_build)
    fake_config = SimpleNamespace(services={"default": fake_service})
    fake_compose_module = SimpleNamespace(
        convert_compose_to_modal_params=original_convert,
        resolve_dockerfile_path=lambda build, compose_dir: Path(build.context) / (build.dockerfile or "Dockerfile"),
        modal=SimpleNamespace(Image=FakeImage),
    )
    fake_modal_module = SimpleNamespace(convert_compose_to_modal_params=original_convert)
    fake_modules = {
        "inspect_sandboxes.modal._compose": fake_compose_module,
        "inspect_sandboxes.modal._modal": fake_modal_module,
    }

    monkeypatch.setattr(cli.importlib, "import_module", lambda name: fake_modules[name])

    cli._patch_inspect_sandboxes_modal_context_dir()

    params = fake_modal_module.convert_compose_to_modal_params(fake_config, None)

    assert original_calls == [(fake_config, None)]
    assert calls == [
        (
            "/tmp/harbor-task/environment/Dockerfile",
            "/tmp/harbor-task/environment",
        )
    ]
    assert params == {
        "image": (
            "image",
            "/tmp/harbor-task/environment/Dockerfile",
            "/tmp/harbor-task/environment",
        ),
        "cpu": 1.0,
    }


def test_patch_inspect_sandboxes_modal_context_dir_ignores_missing_modal_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_import_module(name: str) -> object:
        calls.append(name)
        raise ModuleNotFoundError("No module named 'modal'", name="modal")

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    cli._patch_inspect_sandboxes_modal_context_dir()

    assert calls == ["inspect_sandboxes.modal._compose"]


def test_load_modal_sandbox_overrides_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cli.MODAL_SANDBOX_TIMEOUT_ENV, "0")

    with pytest.raises(ValueError, match=cli.MODAL_SANDBOX_TIMEOUT_ENV):
        cli._load_modal_sandbox_overrides()


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


def test_modal_output_context_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeContextManager:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, exc_type, exc, tb) -> None:
            events.append("exit")

    monkeypatch.setenv("DFM_EVALS_MODAL_ENABLE_OUTPUT", "1")
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(enable_output=lambda: FakeContextManager())
        if name == "modal"
        else object(),
    )

    with cli._modal_output_context():
        events.append("body")

    assert events == ["enter", "body", "exit"]


def test_modal_output_context_requires_modal_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DFM_EVALS_MODAL_ENABLE_OUTPUT", "1")

    def fake_import_module(name: str) -> object:
        if name == "modal":
            raise ModuleNotFoundError("No module named 'modal'", name="modal")
        return object()

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="requires the `modal` package"):
        cli._modal_output_context()
