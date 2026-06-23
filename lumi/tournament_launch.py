#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from dfm_evals.tournament._definitions import (
    list_tournament_definitions,
    resolve_tournament_definition,
)
from dfm_evals.tournament._resolve import resolve_stateful_tournament_config
from dfm_evals.tournament.config import TournamentConfig, load_tournament_config

LOCAL_VLLM = "local_vllm"
EXTERNAL_OPENAI = "external_openai"


class LaunchDefaults(BaseModel):
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    nodes: int | None = None
    ctx: int | None = None
    gpu_mem: float | None = None
    visible_devices: str | None = None
    default_chat_template_kwargs: dict[str, Any] | None = None
    safetensors_load_strategy: str | None = None
    trust_remote_code: bool | None = None
    enable_auto_tool_choice: bool | None = None
    tool_call_parser: str | None = None
    enforce_eager: bool | None = None
    extra_env: dict[str, str] | None = None
    extra_vllm_args: list[str] | None = None

    model_config = {
        "extra": "forbid",
    }


class LaunchEntry(BaseModel):
    mode: str
    model: str | None = None
    served_model_name: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    nodes: int | None = None
    ctx: int | None = None
    gpu_mem: float | None = None
    visible_devices: str | None = None
    default_chat_template_kwargs: dict[str, Any] | None = None
    safetensors_load_strategy: str | None = None
    trust_remote_code: bool | None = None
    enable_auto_tool_choice: bool | None = None
    tool_call_parser: str | None = None
    enforce_eager: bool | None = None
    extra_env: dict[str, str] | None = None
    extra_vllm_args: list[str] | None = None

    model_config = {
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def validate_mode(self) -> "LaunchEntry":
        if self.mode not in (LOCAL_VLLM, EXTERNAL_OPENAI):
            raise ValueError(
                f"invalid mode `{self.mode}` (expected `{LOCAL_VLLM}` or `{EXTERNAL_OPENAI}`)"
            )
        return self


class ResolvedLaunchEntry(BaseModel):
    mode: str
    model: str | None = None
    served_model_name: str
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    nodes: int | None = None
    ctx: int | None = None
    gpu_mem: float | None = None
    visible_devices: str | None = None
    default_chat_template_kwargs: dict[str, Any] | None = None
    safetensors_load_strategy: str | None = None
    trust_remote_code: bool | None = None
    enable_auto_tool_choice: bool | None = None
    tool_call_parser: str | None = None
    enforce_eager: bool | None = None
    extra_env: dict[str, str] | None = None
    extra_vllm_args: list[str] | None = None

    model_config = {
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def validate_entry(self) -> "ResolvedLaunchEntry":
        if self.mode == LOCAL_VLLM:
            if self.model is None or self.model.strip() == "":
                raise ValueError("local_vllm entries require `model`")
            if self.tp is not None and self.tp <= 0:
                raise ValueError("tp must be greater than 0")
            if self.pp is not None and self.pp <= 0:
                raise ValueError("pp must be greater than 0")
            if self.dp is not None and self.dp <= 0:
                raise ValueError("dp must be greater than 0")
            if self.nodes is not None and self.nodes <= 0:
                raise ValueError("nodes must be greater than 0")
            if self.ctx is not None and self.ctx <= 0:
                raise ValueError("ctx must be greater than 0")
            if self.gpu_mem is not None and self.gpu_mem <= 0.0:
                raise ValueError("gpu_mem must be greater than 0")
        elif self.mode == EXTERNAL_OPENAI:
            if not any(
                value is not None and value.strip() != ""
                for value in (self.base_url, self.base_url_env)
            ):
                raise ValueError(
                    "external_openai entries require `base_url` or `base_url_env`"
                )
            if self.extra_env:
                raise ValueError("external_openai entries do not support `extra_env`")
            if self.extra_vllm_args:
                raise ValueError(
                    "external_openai entries do not support `extra_vllm_args`"
                )
        else:  # pragma: no cover
            raise ValueError(f"unsupported mode: {self.mode}")

        return self


class TournamentLaunchMap(BaseModel):
    defaults: LaunchDefaults = Field(default_factory=LaunchDefaults)
    judge_defaults: LaunchDefaults = Field(default_factory=LaunchDefaults)
    contestants: dict[str, LaunchEntry]
    judge: LaunchEntry

    model_config = {
        "extra": "forbid",
    }


def load_launch_map(path: str | Path) -> TournamentLaunchMap:
    launch_path = resolve_tournament_definition(path, kind="launch_map")
    if not launch_path.is_file():
        raise FileNotFoundError(f"Tournament launch-map file not found: {launch_path}")

    parsed = yaml.safe_load(launch_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Tournament launch-map must be a YAML/JSON object")

    launch_map = TournamentLaunchMap.model_validate(parsed)
    base_dir = launch_path.parent.resolve()

    contestants = {
        name: _resolve_relative_paths(entry, base_dir)
        for name, entry in launch_map.contestants.items()
    }
    judge = _resolve_relative_paths(launch_map.judge, base_dir)
    return launch_map.model_copy(update={"contestants": contestants, "judge": judge})


def write_runtime_config(
    *,
    source: str | Path,
    output: str | Path,
    stateful: bool = False,
    run_dir: str | Path | None = None,
    contestant_models: list[str] | None = None,
    judge_model: str | None = None,
) -> Path:
    config = _load_config(source, stateful=stateful)
    update: dict[str, Any] = {}
    if run_dir is not None:
        update["run_dir"] = Path(run_dir)
    if contestant_models is not None:
        update["contestant_models"] = list(contestant_models)
    if judge_model is not None:
        update["judge_model"] = judge_model
    if update:
        config = TournamentConfig.model_validate(
            {
                **config.model_dump(mode="python"),
                **update,
            }
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def resolve_definition_path(
    *,
    source: str | Path,
    kind: str,
) -> Path:
    if kind not in {"config", "launch_map"}:
        raise ValueError(f"unsupported definition kind: {kind}")
    return resolve_tournament_definition(source, kind=kind)


def list_definition_names() -> list[str]:
    return [path.name for path in list_tournament_definitions()]


def emit_target_shell(
    *,
    source: str | Path,
    stateful: bool = False,
) -> str:
    config = _load_config(source, stateful=stateful)
    run_label = _derive_run_label(config)
    lines = [
        _shell_assign("TARGET_RUN_DIR", config.run_dir.as_posix()),
        _shell_assign("TARGET_CONFIG_DIR", config.config_dir.as_posix()),
        _shell_assign("TARGET_INSPECT_DIR", config.inspect_dir.as_posix()),
        _shell_assign("TARGET_GENERATION_LOG_DIR", config.generation_log_dir.as_posix()),
        _shell_assign("TARGET_JUDGE_LOG_DIR", config.judge_log_dir.as_posix()),
        _shell_assign("TARGET_STATE_DIR", config.state_dir.as_posix()),
        _shell_assign("TARGET_EXPORTS_DIR", config.exports_dir.as_posix()),
        _shell_assign("TARGET_TRACES_DIR", config.traces_dir.as_posix()),
        _shell_assign("TARGET_SERVICES_DIR", config.services_dir.as_posix()),
        _shell_assign("TARGET_VLLM_LOG_DIR", config.vllm_log_dir.as_posix()),
        _shell_assign("TARGET_RUNTIME_CONFIG_PATH", config.runtime_config_path.as_posix()),
        _shell_assign("TARGET_RUN_LABEL", run_label),
        _shell_assign("TARGET_JUDGE_MODEL", config.judge_model),
        _shell_assign("TARGET_PROJECT_ID", config.project_id or ""),
        _shell_array_assign("TARGET_CONTESTANT_MODELS", config.contestant_models),
    ]
    return "\n".join(lines)


def emit_spec_shell(
    *,
    source: str | Path,
    launch_map_path: str | Path,
    spec_kind: str,
    model_name: str | None = None,
    stateful: bool = False,
) -> str:
    config = _load_config(source, stateful=stateful)
    launch_map = load_launch_map(launch_map_path)

    if spec_kind == "contestant":
        if model_name is None or model_name.strip() == "":
            raise ValueError("contestant spec requires --model")
        if model_name not in config.contestant_models:
            raise ValueError(
                f"contestant model `{model_name}` is not present in tournament config"
            )
        if model_name not in launch_map.contestants:
            raise ValueError(
                f"contestant model `{model_name}` is missing from the launch-map"
            )
        resolved = _resolve_entry(
            endpoint_model_ref=model_name,
            entry=launch_map.contestants[model_name],
            defaults=launch_map.defaults,
        )
        spec_name = model_name
    elif spec_kind == "judge":
        resolved = _resolve_entry(
            endpoint_model_ref=config.judge_model,
            entry=launch_map.judge,
            defaults=launch_map.defaults,
            secondary_defaults=launch_map.judge_defaults,
        )
        spec_name = config.judge_model
    else:
        raise ValueError(f"unsupported spec kind: {spec_kind}")

    lines = [
        _shell_assign("SPEC_KIND", spec_kind),
        _shell_assign("SPEC_NAME", spec_name),
        _shell_assign("SPEC_MODE", resolved.mode),
        _shell_assign("SPEC_MODEL", resolved.model or ""),
        _shell_assign("SPEC_SERVED_MODEL_NAME", resolved.served_model_name),
        _shell_assign("SPEC_BASE_URL", resolved.base_url or ""),
        _shell_assign("SPEC_BASE_URL_ENV", resolved.base_url_env or ""),
        _shell_assign("SPEC_API_KEY", resolved.api_key or ""),
        _shell_assign("SPEC_API_KEY_ENV", resolved.api_key_env or ""),
        _shell_assign("SPEC_TP", _stringify_optional(resolved.tp)),
        _shell_assign("SPEC_PP", _stringify_optional(resolved.pp)),
        _shell_assign("SPEC_DP", _stringify_optional(resolved.dp)),
        _shell_assign("SPEC_NODES", _stringify_optional(resolved.nodes)),
        _shell_assign("SPEC_CTX", _stringify_optional(resolved.ctx)),
        _shell_assign("SPEC_GPU_MEM", _stringify_optional(resolved.gpu_mem)),
        _shell_assign("SPEC_VISIBLE_DEVICES", resolved.visible_devices or ""),
        _shell_assign(
            "SPEC_DEFAULT_CHAT_TEMPLATE_KWARGS_JSON",
            json.dumps(resolved.default_chat_template_kwargs)
            if resolved.default_chat_template_kwargs is not None
            else "",
        ),
        _shell_assign(
            "SPEC_SAFETENSORS_LOAD_STRATEGY",
            resolved.safetensors_load_strategy or "",
        ),
        _shell_assign(
            "SPEC_TRUST_REMOTE_CODE",
            _bool_to_flag(resolved.trust_remote_code),
        ),
        _shell_assign(
            "SPEC_ENABLE_AUTO_TOOL_CHOICE",
            _bool_to_flag(resolved.enable_auto_tool_choice),
        ),
        _shell_assign("SPEC_TOOL_CALL_PARSER", resolved.tool_call_parser or ""),
        _shell_assign("SPEC_ENFORCE_EAGER", _bool_to_flag(resolved.enforce_eager)),
        _shell_array_assign(
            "SPEC_EXTRA_ENV_KV",
            [f"{key}={value}" for key, value in (resolved.extra_env or {}).items()],
        ),
        _shell_array_assign(
            "SPEC_EXTRA_VLLM_ARGS",
            resolved.extra_vllm_args or [],
        ),
    ]
    return "\n".join(lines)


def emit_status_shell(
    *,
    source: str | Path,
    stateful: bool = False,
    index_generation: bool = False,
) -> str:
    from dfm_evals.tournament.indexer import index_generation_responses
    from dfm_evals.tournament.orchestrator import tournament_status

    config = _load_config(source, stateful=stateful)
    missing_models: list[str] = []
    if index_generation:
        report = index_generation_responses(config)
        missing_models = sorted(
            model_name
            for model_name, prompt_ids in report.missing_by_model.items()
            if len(prompt_ids) > 0
        )
    status = tournament_status(config)
    lines = [
        _shell_assign("STATUS_RUN_STATUS", status.run_status),
        _shell_assign("STATUS_CONVERGED", _bool_to_flag(status.converged)),
        _shell_assign("STATUS_MISSING_RESPONSES", str(status.missing_responses)),
        _shell_assign("STATUS_MISSING_MODELS_CSV", ",".join(missing_models)),
        _shell_assign("STATUS_TOTAL_MATCHES", str(status.total_matches)),
        _shell_assign("STATUS_RATED_MATCHES", str(status.rated_matches)),
        _shell_assign("STATUS_PENDING_BATCH_ID", status.pending_batch_id or ""),
    ]
    return "\n".join(lines)


def emit_resource_shell(
    *,
    source: str | Path,
    launch_map_path: str | Path,
    phase: str,
    stateful: bool = False,
    model_names: list[str] | None = None,
) -> str:
    config = _load_config(source, stateful=stateful)
    launch_map = load_launch_map(launch_map_path)
    explicit_models = [name for name in (model_names or []) if name.strip() != ""]
    relevant_entries: list[ResolvedLaunchEntry] = []

    if phase in {"all", "generate"}:
        contestant_models = explicit_models if explicit_models else list(config.contestant_models)
        relevant_entries.extend(
            _resolve_contestant_entries(config, launch_map, contestant_models)
        )
    elif phase == "add-model":
        relevant_entries.extend(
            _resolve_contestant_entries(
                config,
                launch_map,
                explicit_models,
                require_in_config=False,
            )
        )
    elif phase in {"run", "resume"}:
        if explicit_models:
            relevant_entries.extend(
                _resolve_contestant_entries(config, launch_map, explicit_models)
            )
        else:
            from dfm_evals.tournament.indexer import index_generation_responses

            report = index_generation_responses(config)
            missing_models = sorted(
                model_name
                for model_name, prompt_ids in report.missing_by_model.items()
                if len(prompt_ids) > 0
            )
            relevant_entries.extend(
                _resolve_contestant_entries(config, launch_map, missing_models)
            )
    elif phase == "export":
        pass
    else:
        raise ValueError(f"unsupported phase: {phase}")

    if phase in {"all", "run", "resume", "add-model"}:
        judge_entry = _resolve_entry(
            endpoint_model_ref=config.judge_model,
            entry=launch_map.judge,
            defaults=launch_map.defaults,
            secondary_defaults=launch_map.judge_defaults,
        )
        relevant_entries.append(judge_entry)

    max_nodes = 1
    max_local_world_size = 0
    for entry in relevant_entries:
        if entry.mode != LOCAL_VLLM:
            continue
        nodes = entry.nodes or 1
        tp = entry.tp or 1
        pp = entry.pp or 1
        dp = entry.dp or 1
        max_nodes = max(max_nodes, nodes)
        max_local_world_size = max(max_local_world_size, tp * pp * dp)

    lines = [
        _shell_assign("RESOURCE_MAX_NODES", str(max_nodes)),
        _shell_assign("RESOURCE_MAX_LOCAL_WORLD_SIZE", str(max_local_world_size)),
    ]
    return "\n".join(lines)


def _load_config(source: str | Path, *, stateful: bool) -> TournamentConfig:
    if stateful:
        return resolve_stateful_tournament_config(source)
    return load_tournament_config(source)


def _resolve_relative_paths(entry: LaunchEntry, base_dir: Path) -> LaunchEntry:
    if entry.model is None:
        return entry
    return entry.model_copy(update={"model": _resolve_path_like(entry.model, base_dir)})


def _resolve_entry(
    *,
    endpoint_model_ref: str,
    entry: LaunchEntry,
    defaults: LaunchDefaults,
    secondary_defaults: LaunchDefaults | None = None,
) -> ResolvedLaunchEntry:
    merged: dict[str, Any] = defaults.model_dump(exclude_none=True)
    if secondary_defaults is not None:
        merged.update(secondary_defaults.model_dump(exclude_none=True))
    merged.update(entry.model_dump(exclude_none=True))

    if not merged.get("served_model_name"):
        merged["served_model_name"] = _served_model_name(endpoint_model_ref)

    return ResolvedLaunchEntry.model_validate(merged)


def _resolve_contestant_entries(
    config: TournamentConfig,
    launch_map: TournamentLaunchMap,
    model_names: list[str],
    *,
    require_in_config: bool = True,
) -> list[ResolvedLaunchEntry]:
    resolved: list[ResolvedLaunchEntry] = []
    for model_name in model_names:
        if require_in_config and model_name not in config.contestant_models:
            raise ValueError(
                f"contestant model `{model_name}` is not present in tournament config"
            )
        if model_name not in launch_map.contestants:
            raise ValueError(
                f"contestant model `{model_name}` is missing from the launch-map"
            )
        resolved.append(
            _resolve_entry(
                endpoint_model_ref=model_name,
                entry=launch_map.contestants[model_name],
                defaults=launch_map.defaults,
            )
        )
    return resolved


def _resolve_path_like(value: str, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return path.resolve().as_posix() if path.exists() else value

    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate.as_posix()
    return value


def _served_model_name(model_ref: str) -> str:
    stripped = model_ref.strip()
    for prefix in ("openai/", "vllm/"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break
    return stripped


def _derive_run_label(config: TournamentConfig) -> str:
    name = config.run_label.strip()
    if name == "":
        raise ValueError(f"could not derive run label from run_dir: {config.run_dir}")
    return name


def _shell_assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def _shell_array_assign(name: str, values: list[str]) -> str:
    quoted = " ".join(shlex.quote(value) for value in values)
    return f"{name}=({quoted})"


def _bool_to_flag(value: bool | None) -> str:
    if value is None:
        return ""
    return "1" if value else "0"


def _stringify_optional(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python lumi/tournament_launch.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_runtime_parser = subparsers.add_parser(
        "write-runtime-config",
        help="Write a runtime tournament config with optional path overrides",
    )
    write_runtime_parser.add_argument("--source", required=True)
    write_runtime_parser.add_argument("--output", required=True)
    write_runtime_parser.add_argument("--stateful", action="store_true")
    write_runtime_parser.add_argument("--run-dir", default=None)
    write_runtime_parser.add_argument(
        "--contestant-model",
        dest="contestant_models",
        action="append",
        default=None,
    )
    write_runtime_parser.add_argument("--judge-model", default=None)

    emit_target_parser = subparsers.add_parser(
        "emit-target-shell",
        help="Emit bash assignments for tournament config paths and model list",
    )
    emit_target_parser.add_argument("--source", required=True)
    emit_target_parser.add_argument("--stateful", action="store_true")

    emit_spec_parser = subparsers.add_parser(
        "emit-spec-shell",
        help="Emit bash assignments for one contestant or judge launch spec",
    )
    emit_spec_parser.add_argument("--source", required=True)
    emit_spec_parser.add_argument("--launch-map", required=True)
    emit_spec_parser.add_argument(
        "--kind",
        required=True,
        choices=["contestant", "judge"],
    )
    emit_spec_parser.add_argument("--model", default=None)
    emit_spec_parser.add_argument("--stateful", action="store_true")

    emit_status_parser = subparsers.add_parser(
        "emit-status-shell",
        help="Emit bash assignments for tournament status fields",
    )
    emit_status_parser.add_argument("--source", required=True)
    emit_status_parser.add_argument("--stateful", action="store_true")
    emit_status_parser.add_argument("--index-generation", action="store_true")

    emit_resource_parser = subparsers.add_parser(
        "emit-resource-shell",
        help="Emit bash assignments for required Slurm resources",
    )
    emit_resource_parser.add_argument("--source", required=True)
    emit_resource_parser.add_argument("--launch-map", required=True)
    emit_resource_parser.add_argument(
        "--phase",
        required=True,
        choices=["all", "generate", "run", "resume", "add-model", "export"],
    )
    emit_resource_parser.add_argument("--stateful", action="store_true")
    emit_resource_parser.add_argument(
        "--model",
        dest="model_names",
        action="append",
        default=None,
    )

    resolve_path_parser = subparsers.add_parser(
        "resolve-path",
        help="Resolve a tournament definition name, directory, or file path",
    )
    resolve_path_parser.add_argument("--source", required=True)
    resolve_path_parser.add_argument(
        "--kind",
        required=True,
        choices=["config", "launch_map"],
    )

    subparsers.add_parser(
        "list-definitions",
        help="List committed tournament definitions",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "write-runtime-config":
            path = write_runtime_config(
                source=args.source,
                output=args.output,
                stateful=args.stateful,
                run_dir=args.run_dir,
                contestant_models=args.contestant_models,
                judge_model=args.judge_model,
            )
            print(path.as_posix())
            return 0

        if args.command == "emit-target-shell":
            print(emit_target_shell(source=args.source, stateful=args.stateful))
            return 0

        if args.command == "emit-spec-shell":
            print(
                emit_spec_shell(
                    source=args.source,
                    launch_map_path=args.launch_map,
                    spec_kind=args.kind,
                    model_name=args.model,
                    stateful=args.stateful,
                )
            )
            return 0

        if args.command == "emit-status-shell":
            print(
                emit_status_shell(
                    source=args.source,
                    stateful=args.stateful,
                    index_generation=args.index_generation,
                )
            )
            return 0

        if args.command == "emit-resource-shell":
            print(
                emit_resource_shell(
                    source=args.source,
                    launch_map_path=args.launch_map,
                    phase=args.phase,
                    stateful=args.stateful,
                    model_names=args.model_names,
                )
            )
            return 0

        if args.command == "resolve-path":
            print(
                resolve_definition_path(
                    source=args.source,
                    kind=args.kind,
                ).as_posix()
            )
            return 0

        if args.command == "list-definitions":
            for name in list_definition_names():
                print(name)
            return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
