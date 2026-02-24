from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

DEFAULT_SUITES_FILE = "eval-sets.yaml"
PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")


@dataclass(frozen=True)
class EvalTaskSpec:
    name: str
    args: list[str]


@dataclass(frozen=True)
class EvalSuite:
    tasks: list[EvalTaskSpec]
    args: list[str]
    description: str = ""


@dataclass(frozen=True)
class ModelRouting:
    target_model: str | None = None
    target_base_url: str | None = None
    judge_model: str | None = None
    judge_base_url: str | None = None


def _normalize_remainder_args(args: Sequence[str]) -> list[str]:
    normalized = list(args)
    if len(normalized) > 0 and normalized[0] == "--":
        normalized = normalized[1:]
    return normalized


def _parse_string_list(value: object, *, field: str, suite: str) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(
            f"Invalid `{field}` in suite `{suite}`: expected a list of strings."
        )

    cleaned = [item.strip() for item in value]
    return [item for item in cleaned if item]


def _parse_task_specs(value: object, *, suite: str) -> list[EvalTaskSpec]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(
            f"Invalid `tasks` in suite `{suite}`: expected a list of strings or mappings."
        )

    specs: list[EvalTaskSpec] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            name = item.strip()
            task_args: list[str] = []
        elif isinstance(item, dict):
            raw_name = item.get("name", item.get("task"))
            if not isinstance(raw_name, str):
                raise ValueError(
                    f"Invalid `tasks[{index}]` in suite `{suite}`: expected `name` (or `task`) string."
                )
            name = raw_name.strip()
            task_args = _parse_string_list(
                item.get("args"), field=f"tasks[{index}].args", suite=suite
            )
            if "route" in item:
                raise ValueError(
                    f"Invalid `tasks[{index}].route` in suite `{suite}`: `route` is no longer supported. "
                    "Use placeholders in `args` (e.g. `grader_model={{judge_model}}`)."
                )
        else:
            raise ValueError(
                f"Invalid `tasks[{index}]` in suite `{suite}`: expected a string or mapping."
            )

        if not name:
            raise ValueError(
                f"Invalid `tasks[{index}]` in suite `{suite}`: task name cannot be empty."
            )

        specs.append(EvalTaskSpec(name=name, args=task_args))

    return specs


def _read_suites_config_text(path: str) -> str | None:
    if path == DEFAULT_SUITES_FILE:
        default_resource = resources.files("dfm_evals").joinpath(DEFAULT_SUITES_FILE)
        if not default_resource.is_file():
            return None
        return default_resource.read_text(encoding="utf-8")

    config_path = Path(path)
    if config_path.exists():
        return config_path.read_text(encoding="utf-8")

    return None


def _load_named_suites(path: str) -> dict[str, EvalSuite]:
    try:
        source_text = _read_suites_config_text(path)
    except OSError as exc:
        raise ValueError(f"Unable to read suites config `{path}`: {exc}") from exc

    if source_text is None:
        return {}

    try:
        loaded = yaml.safe_load(source_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Unable to read suites config `{path}`: {exc}") from exc

    if loaded is None:
        return {}

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Invalid suites config `{path}`: expected a mapping at the top level."
        )

    raw_suites = loaded.get("sets", {})
    if not isinstance(raw_suites, dict):
        raise ValueError(
            f"Invalid suites config `{path}`: `sets` must be a mapping."
        )

    suites: dict[str, EvalSuite] = {}
    for name, raw_suite in raw_suites.items():
        if not isinstance(name, str):
            raise ValueError("Invalid suites config: suite names must be strings.")

        if isinstance(raw_suite, list):
            tasks = _parse_task_specs(raw_suite, suite=name)
            suite = EvalSuite(tasks=tasks, args=[], description="")
        elif isinstance(raw_suite, dict):
            tasks = _parse_task_specs(raw_suite.get("tasks"), suite=name)
            args = _parse_string_list(raw_suite.get("args"), field="args", suite=name)
            description = raw_suite.get("description", "")
            if description is None:
                description = ""
            if not isinstance(description, str):
                raise ValueError(
                    f"Invalid `description` in suite `{name}`: expected a string."
                )
            suite = EvalSuite(tasks=tasks, args=args, description=description.strip())
        else:
            raise ValueError(
                f"Invalid suite `{name}`: expected a list of tasks or a mapping."
            )

        if len(suite.tasks) == 0:
            raise ValueError(f"Invalid suite `{name}`: `tasks` cannot be empty.")

        suites[name] = suite

    return suites


def _placeholder_values(routing: ModelRouting) -> dict[str, str | None]:
    return {
        "target_model": routing.target_model,
        "target_base_url": routing.target_base_url,
        "judge_model": routing.judge_model,
        "judge_base_url": routing.judge_base_url,
    }


def _resolve_placeholders_in_string(
    value: str,
    *,
    routing: ModelRouting,
    field: str,
    suite: str,
) -> str:
    placeholders = _placeholder_values(routing)

    def replace_match(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in placeholders:
            allowed = ", ".join(sorted(placeholders.keys()))
            raise ValueError(
                f"Invalid placeholder `{{{{{key}}}}}` in `{field}` for suite `{suite}`. "
                f"Supported placeholders: {allowed}"
            )
        resolved = placeholders[key]
        if resolved is None:
            option = f"--{key.replace('_', '-')}"
            raise ValueError(
                f"Placeholder `{{{{{key}}}}}` in `{field}` for suite `{suite}` requires `{option}`."
            )
        return resolved

    return PLACEHOLDER_PATTERN.sub(replace_match, value)


def _resolve_placeholders_in_args(
    values: Sequence[str],
    *,
    routing: ModelRouting,
    field: str,
    suite: str,
) -> list[str]:
    return [
        _resolve_placeholders_in_string(
            value,
            routing=routing,
            field=f"{field}[{index}]",
            suite=suite,
        )
        for index, value in enumerate(values)
    ]


def _resolve_suite_task_specs(
    suite: EvalSuite,
    *,
    suite_name: str,
    routing: ModelRouting,
) -> list[EvalTaskSpec]:
    resolved_tasks: list[EvalTaskSpec] = []
    for index, task in enumerate(suite.tasks):
        resolved_tasks.append(
            EvalTaskSpec(
                name=task.name,
                args=_resolve_placeholders_in_args(
                    task.args,
                    routing=routing,
                    field=f"tasks[{index}].args",
                    suite=suite_name,
                ),
            )
        )
    return resolved_tasks


def _run_suite_grouped(
    task_specs: Sequence[EvalTaskSpec],
    *,
    suite_args: Sequence[str],
    mode: str,
    extra_args: Sequence[str],
    prog_name: str = "evals",
) -> int:
    eval_command = "eval-set" if mode == "set" else "eval"
    grouped_tasks: dict[tuple[str, ...], list[str]] = {}
    for task in task_specs:
        key = tuple(task.args)
        grouped_tasks.setdefault(key, []).append(task.name)

    for task_args, task_names in grouped_tasks.items():
        exit_code = _forward_to_inspect(
            [eval_command, *task_names, *suite_args, *task_args, *extra_args],
            prog_name=prog_name,
        )
        if exit_code != 0:
            return exit_code
    return 0


def _forward_to_inspect(args: Sequence[str], *, prog_name: str = "evals") -> int:
    # Ensure local task registry is loaded.
    import dfm_evals._registry  # noqa: F401, I001
    from inspect_ai._cli.main import inspect as inspect_command

    forwarded = _normalize_remainder_args(args)

    try:
        inspect_command.main(
            args=forwarded,
            prog_name=prog_name,
            standalone_mode=False,
        )
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 1 if code else 0

    return 0


def _list_registered_tasks(prefix: str) -> list[str]:
    # Ensure local registry module is imported so task registration executes.
    import dfm_evals._registry  # noqa: F401, I001
    from inspect_ai._util.registry import registry_find, registry_info

    registered = registry_find(
        lambda info: info.type == "task" and info.name.startswith(prefix)
    )
    names = {registry_info(task).name for task in registered}
    return sorted(names)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Run eval tasks via inspect (forwarded to `inspect eval`)"
    )
    run_parser.add_argument("args", nargs=argparse.REMAINDER)
    set_parser = subparsers.add_parser(
        "set", help="Run eval sets via inspect (forwarded to `inspect eval-set`)"
    )
    set_parser.add_argument("args", nargs=argparse.REMAINDER)
    suite_parser = subparsers.add_parser(
        "suite",
        help="Run a named eval suite from a config file",
    )
    suite_parser.add_argument("name", help="Suite name from config")
    suite_parser.add_argument(
        "--file",
        default=DEFAULT_SUITES_FILE,
        help=f"Path to suite config file (default: {DEFAULT_SUITES_FILE})",
    )
    suite_parser.add_argument(
        "--mode",
        choices=["set", "run"],
        default="set",
        help="Use `set` (inspect eval-set) or `run` (inspect eval)",
    )
    suite_parser.add_argument(
        "--target-model",
        default=None,
        help="Value for {{target_model}} placeholders in suite/task args.",
    )
    suite_parser.add_argument(
        "--target-base-url",
        default=None,
        help="Value for {{target_base_url}} placeholders in suite/task args.",
    )
    suite_parser.add_argument(
        "--judge-model",
        default=None,
        help="Value for {{judge_model}} placeholders in suite/task args.",
    )
    suite_parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Value for {{judge_base_url}} placeholders in suite/task args.",
    )
    suites_parser = subparsers.add_parser(
        "suites",
        help="List named eval suites from a config file",
    )
    suites_parser.add_argument(
        "--file",
        default=DEFAULT_SUITES_FILE,
        help=f"Path to suite config file (default: {DEFAULT_SUITES_FILE})",
    )
    suites_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include description and tasks in output",
    )
    tasks_parser = subparsers.add_parser("tasks", help="List registered task names")
    tasks_parser.add_argument(
        "--prefix",
        default="",
        help="Only show tasks with this registry prefix",
    )

    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args, passthrough_args = parser.parse_known_args(argv_list)

    if len(passthrough_args) > 0 and args.command != "suite":
        parser.error(f"unrecognized arguments: {' '.join(passthrough_args)}")

    if args.command == "run":
        return _forward_to_inspect(["eval", *args.args], prog_name="evals")

    if args.command == "set":
        return _forward_to_inspect(["eval-set", *args.args], prog_name="evals")

    if args.command == "suite":
        try:
            suites = _load_named_suites(args.file)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        suite = suites.get(args.name)
        if suite is None:
            print(
                f"Error: suite `{args.name}` not found in `{args.file}`.",
                file=sys.stderr,
            )
            if len(suites) > 0:
                available = ", ".join(sorted(suites.keys()))
                print(f"Available suites: {available}", file=sys.stderr)
            return 2

        if len(passthrough_args) > 0 and "--" not in argv_list:
            parser.error(f"unrecognized arguments: {' '.join(passthrough_args)}")

        extra_args = _normalize_remainder_args(passthrough_args)
        routing = ModelRouting(
            target_model=args.target_model,
            target_base_url=args.target_base_url,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
        )
        resolved_suite_args = _resolve_placeholders_in_args(
            suite.args,
            routing=routing,
            field="args",
            suite=args.name,
        )
        resolved_tasks = _resolve_suite_task_specs(
            suite,
            suite_name=args.name,
            routing=routing,
        )
        task_names = [task.name for task in resolved_tasks]
        has_per_task_args = any(len(task.args) > 0 for task in resolved_tasks)

        if not has_per_task_args:
            eval_command = "eval-set" if args.mode == "set" else "eval"
            return _forward_to_inspect(
                [eval_command, *task_names, *resolved_suite_args, *extra_args],
                prog_name="evals",
            )

        return _run_suite_grouped(
            resolved_tasks,
            suite_args=resolved_suite_args,
            mode=args.mode,
            extra_args=extra_args,
            prog_name="evals",
        )

    if args.command == "suites":
        try:
            suites = _load_named_suites(args.file)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        for name in sorted(suites.keys()):
            suite = suites[name]
            if args.verbose:
                task_names = ",".join(task.name for task in suite.tasks)
                print(f"{name}\t{suite.description}\t{task_names}")
            else:
                print(name)
        return 0

    if args.command == "tasks":
        tasks = _list_registered_tasks(args.prefix)
        for task_name in tasks:
            print(task_name)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
