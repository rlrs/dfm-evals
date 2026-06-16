from __future__ import annotations

from importlib import import_module
from pathlib import Path

REGISTRY_EXPORTS = {
    "danish_citizen_tests": "dfm_evals.tasks.danish_citizen_tests:danish_citizen_tests",
    "daisy": "dfm_evals.tasks.daisy:daisy",
    "generative_talemaader": "dfm_evals.tasks.talemaader.task:generative_talemaader",
    "multi_wiki_qa": "dfm_evals.tasks.multi_wiki_qa:multi_wiki_qa",
    "dala": "dfm_evals.tasks.dala:dala",
    "gec_dala": "dfm_evals.tasks.gec_dala:gec_dala",
    "bfcl": "dfm_evals.tasks.bfcl.bfcl:bfcl",
    "bfcl_da": "dfm_evals.tasks.bfcl.bfcl:bfcl_da",
    "ifeval_da": "dfm_evals.tasks.ifeval_da:ifeval_da",
    "piqa": "dfm_evals.tasks.piqa:piqa",
    "ruler": "dfm_evals.tasks.ruler.task:ruler",
    "gleu": "dfm_evals.scorers.gleu:gleu",
    "comet": "dfm_evals.scorers.comet:comet",
}

TOP_LEVEL_EXPORTS = (
    "generative_talemaader",
    "multi_wiki_qa",
    "bfcl",
    "bfcl_da",
    "ifeval_da",
    "gleu",
    "comet",
)

TASK_EXPORTS = tuple(
    name
    for name, target in REGISTRY_EXPORTS.items()
    if target.startswith("dfm_evals.tasks.")
)
SCORER_EXPORTS = tuple(
    name
    for name, target in REGISTRY_EXPORTS.items()
    if target.startswith("dfm_evals.scorers.")
)
SCORER_PACKAGE_EXPORTS = {
    **{name: REGISTRY_EXPORTS[name] for name in SCORER_EXPORTS},
    "compute_gleu": "dfm_evals.scorers.gleu:compute_gleu",
    "max_gleu_score": "dfm_evals.scorers.gleu:max_gleu_score",
}
SANDBOX_EXPORTS = {
    "PrimeSandboxEnvironment": "dfm_evals.sandboxes.prime:PrimeSandboxEnvironment",
}
REGISTRY_ONLY_MODULES = (
    "dfm_evals.sandboxes.prime",
    "dfm_evals.tournament.scorer",
)


def discover_task_modules(
    *,
    root_package: str = "dfm_evals.tasks",
    tasks_dir: Path | None = None,
) -> tuple[str, ...]:
    """Discover importable task entry modules by filesystem convention."""

    if tasks_dir is None:
        tasks_dir = Path(__file__).with_name("tasks")

    discovered_modules: list[str] = []
    for path in sorted(tasks_dir.iterdir(), key=lambda candidate: candidate.name):
        if path.name.startswith((".", "_")):
            continue

        if path.is_file():
            if path.suffix != ".py" or path.name == "__init__.py":
                continue
            discovered_modules.append(f"{root_package}.{path.stem}")
            continue

        if not path.is_dir():
            continue

        package_name = path.name
        candidate_names = ("task.py", f"{package_name}.py", "__init__.py")
        for candidate_name in candidate_names:
            candidate_path = path / candidate_name
            if not candidate_path.is_file():
                continue

            if candidate_name == "__init__.py":
                discovered_modules.append(f"{root_package}.{package_name}")
            else:
                discovered_modules.append(
                    f"{root_package}.{package_name}.{candidate_path.stem}"
                )
            break

    return tuple(discovered_modules)


DISCOVERED_TASK_MODULES = discover_task_modules()
REGISTRY_MODULES = (
    *REGISTRY_ONLY_MODULES,
    *dict.fromkeys(
        [
            *(target.split(":", 1)[0] for target in REGISTRY_EXPORTS.values()),
            *DISCOVERED_TASK_MODULES,
        ]
    ),
)


def load_export(
    *,
    name: str,
    exports: dict[str, str],
    namespace: dict[str, object],
    module_name: str,
) -> object:
    target = exports.get(name)
    if target is None:
        raise AttributeError(f"module '{module_name}' has no attribute '{name}'")

    target_module_name, attribute = target.split(":", 1)
    target_module = import_module(target_module_name)
    value = getattr(target_module, attribute)
    namespace[name] = value
    return value
