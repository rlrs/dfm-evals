from __future__ import annotations

import asyncio
import re
import shlex
import time
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Union, overload
from uuid import uuid4

import yaml
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.util._sandbox.environment import (
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
)
from inspect_ai.util._sandbox.limits import (
    OutputLimitExceededError,
    SandboxEnvironmentLimits,
)
from inspect_ai.util._sandbox.registry import sandboxenv
from inspect_ai.util._subprocess import ExecResult
from pydantic import BaseModel, Field, model_validator
from typing_extensions import override

DEFAULT_MODAL_CONFIG_FILES = [
    "modal-sandbox.yaml",
    "modal-sandbox.yml",
    "modal-sandbox.json",
]

_MODAL_PROVIDER_TAG_KEY = "inspect-provider"
_MODAL_PROVIDER_TAG_VALUE = "inspect-ai-modal"


class ModalSandboxConfig(BaseModel):
    app_name: str = "inspect-sandboxes"
    app_create_if_missing: bool = True
    environment_name: str | None = None

    name_prefix: str = "inspect"
    image: str = "python:3.11-slim"
    startup_command: str = "tail -f /dev/null"
    working_dir: str = "/workspace"

    timeout_seconds: int = 3600
    idle_timeout_seconds: int | None = 600
    cpu_cores: float | None = 1.0
    memory_mb: int | None = 2048
    gpu: str | None = None
    cloud: str | None = None
    region: str | list[str] | None = None

    block_network: bool = False
    cidr_allowlist: list[str] | None = None

    wait_for_start: bool = True
    wait_timeout_seconds: int = 900
    wait_poll_interval_seconds: float = 5.0

    keep_sandbox: bool = False
    environment_vars: dict[str, str] = Field(default_factory=dict)
    metadata_env_prefix: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_values(self) -> "ModalSandboxConfig":
        if not self.working_dir.startswith("/"):
            raise ValueError("`working_dir` must be an absolute path.")
        if self.wait_timeout_seconds < 1:
            raise ValueError("`wait_timeout_seconds` must be >= 1.")
        if self.wait_poll_interval_seconds <= 0:
            raise ValueError("`wait_poll_interval_seconds` must be > 0.")
        if self.timeout_seconds < 1:
            raise ValueError("`timeout_seconds` must be >= 1.")
        if self.idle_timeout_seconds is not None and self.idle_timeout_seconds < 1:
            raise ValueError("`idle_timeout_seconds` must be >= 1 when provided.")
        return self


def _require_modal() -> Any:
    try:
        import modal
    except ModuleNotFoundError as exc:
        raise PrerequisiteError(
            "Modal sandbox integration requires `modal` (install `modal`)."
        ) from exc
    return modal


def _load_config_from_path(path: str) -> ModalSandboxConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(
            f"Modal sandbox config file '{config_path.as_posix()}' does not exist."
        )

    text = config_path.read_text(encoding="utf-8")
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse Modal sandbox config file '{config_path.as_posix()}': {exc}"
        ) from exc

    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Modal sandbox config file '{config_path.as_posix()}' must contain a mapping."
        )

    return ModalSandboxConfig.model_validate(parsed)


def _resolve_modal_config(
    config: SandboxEnvironmentConfigType | None,
) -> ModalSandboxConfig:
    if config is None:
        return ModalSandboxConfig()
    if isinstance(config, ModalSandboxConfig):
        return config
    if isinstance(config, BaseModel):
        return ModalSandboxConfig.model_validate(config.model_dump())
    if isinstance(config, str):
        return _load_config_from_path(config)
    raise ValueError(f"Unsupported Modal sandbox config type: {type(config)!r}")


def _sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "inspect"


def _metadata_env(metadata: dict[str, str], prefix: str | None) -> dict[str, str]:
    if prefix is None:
        return {}

    env: dict[str, str] = {}
    for key, value in metadata.items():
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", key).upper()
        if not normalized:
            continue
        if normalized[0].isdigit():
            normalized = f"_{normalized}"
        env[f"{prefix}{normalized}"] = str(value)
    return env


def _build_sandbox_name(task_name: str, prefix: str) -> str:
    task_suffix = _sanitize_label(task_name.split("/")[-1])
    prefix = _sanitize_label(prefix)
    return f"{prefix}-{task_suffix}-{uuid4().hex[:8]}"


def _enforce_exec_output_limit(output: str) -> None:
    encoded = output.encode("utf-8", errors="replace")
    limit = SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE
    if len(encoded) <= limit:
        return

    truncated = encoded[-limit:].decode("utf-8", errors="replace")
    raise OutputLimitExceededError(
        limit_str=SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE_STR,
        truncated_output=truncated,
    )


def _enforce_read_output_limit(contents: bytes) -> None:
    if len(contents) <= SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
        return
    raise OutputLimitExceededError(
        limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
        truncated_output=None,
    )


def _to_text(value: str | bytes) -> str:
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


@sandboxenv(name="modal")
class ModalSandboxEnvironment(SandboxEnvironment):
    @classmethod
    def config_files(cls) -> list[str]:
        return DEFAULT_MODAL_CONFIG_FILES

    @classmethod
    def config_deserialize(cls, config: dict[str, Any]) -> BaseModel:
        return ModalSandboxConfig.model_validate(config)

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        cfg = _resolve_modal_config(config)
        modal = _require_modal()

        app = await modal.App.lookup.aio(
            cfg.app_name,
            create_if_missing=cfg.app_create_if_missing,
            environment_name=cfg.environment_name,
        )
        image = modal.Image.from_registry(cfg.image)
        startup_args = shlex.split(cfg.startup_command) if cfg.startup_command else []
        environment_vars = {
            **_metadata_env(metadata, cfg.metadata_env_prefix),
            **cfg.environment_vars,
        }

        sandbox = None
        try:
            sandbox = await modal.Sandbox.create.aio(
                *startup_args,
                app=app,
                name=_build_sandbox_name(task_name, cfg.name_prefix),
                image=image,
                env=environment_vars or None,
                timeout=cfg.timeout_seconds,
                idle_timeout=cfg.idle_timeout_seconds,
                workdir=cfg.working_dir,
                gpu=cfg.gpu,
                cloud=cfg.cloud,
                region=cfg.region,
                cpu=cfg.cpu_cores,
                memory=cfg.memory_mb,
                block_network=cfg.block_network,
                cidr_allowlist=cfg.cidr_allowlist,
            )
            tags = {
                _MODAL_PROVIDER_TAG_KEY: _MODAL_PROVIDER_TAG_VALUE,
                "inspect-task": _sanitize_label(task_name),
                **cfg.tags,
            }
            try:
                await sandbox.set_tags.aio(tags)
            except Exception:
                pass

            environment = ModalSandboxEnvironment(
                sandbox=sandbox,
                working_dir=cfg.working_dir,
                keep_sandbox=cfg.keep_sandbox,
            )

            if cfg.wait_for_start:
                await environment._wait_ready(
                    timeout_seconds=cfg.wait_timeout_seconds,
                    poll_interval_seconds=cfg.wait_poll_interval_seconds,
                )

            ensure_working_dir = await environment.exec(
                ["mkdir", "-p", cfg.working_dir],
                timeout=60,
            )
            if not ensure_working_dir.success:
                raise RuntimeError(
                    f"Failed to create working directory '{cfg.working_dir}' in Modal "
                    f"sandbox '{sandbox.object_id}': {ensure_working_dir.stderr}"
                )

            return {"default": environment}
        except BaseException:
            if sandbox is not None:
                try:
                    await sandbox.terminate.aio(wait=False)
                except Exception:
                    pass
            raise

    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        for environment in environments.values():
            modal_environment = environment.as_type(ModalSandboxEnvironment)
            await modal_environment._cleanup(delete=not modal_environment._keep_sandbox)

    @override
    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        modal = _require_modal()
        if id is not None:
            sandbox = modal.Sandbox.from_id(id)
            await sandbox.hydrate.aio()
            await sandbox.terminate.aio(wait=False)
            return

        async for sandbox in modal.Sandbox.list.aio(
            tags={_MODAL_PROVIDER_TAG_KEY: _MODAL_PROVIDER_TAG_VALUE}
        ):
            try:
                await sandbox.terminate.aio(wait=False)
            except Exception:
                pass

    def __init__(
        self,
        *,
        sandbox: Any,
        working_dir: str,
        keep_sandbox: bool = False,
    ) -> None:
        self._sandbox = sandbox
        self._working_dir = PurePosixPath(working_dir)
        self._keep_sandbox = keep_sandbox

    @override
    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        del timeout_retry
        del concurrency

        if len(cmd) == 0:
            raise ValueError("Modal sandbox `exec` requires a non-empty command list.")

        if user is not None:
            warnings.warn(
                "The 'user' parameter is ignored in ModalSandboxEnvironment.",
                UserWarning,
            )

        modal = _require_modal()
        working_dir = self._resolve_working_dir(cwd)
        try:
            process = await self._sandbox.exec.aio(
                *cmd,
                timeout=timeout,
                workdir=working_dir,
                env=env,
                text=False,
            )

            if input is not None:
                payload = input.encode("utf-8") if isinstance(input, str) else input
                process.stdin.write(payload)
                process.stdin.write_eof()
                await process.stdin.drain.aio()

            returncode = await process.wait.aio()
            stdout_raw = await process.stdout.read.aio()
            stderr_raw = await process.stderr.read.aio()
        except modal.exception.ExecTimeoutError as exc:
            raise TimeoutError(str(exc)) from exc
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        if returncode == -1 and timeout is not None:
            raise TimeoutError(
                f"Command timed out in Modal sandbox '{self._sandbox.object_id}' after {timeout}s."
            )

        stdout = _to_text(stdout_raw)
        stderr = _to_text(stderr_raw)
        _enforce_exec_output_limit(stdout)
        _enforce_exec_output_limit(stderr)

        return ExecResult(
            success=returncode == 0,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        file = self._resolve_file(file)
        parent = PurePosixPath(file).parent.as_posix()
        if parent and parent != ".":
            mkdir_result = await self.exec(["mkdir", "-p", parent], timeout=60)
            if not mkdir_result.success:
                raise RuntimeError(
                    f"Failed to create directory '{parent}' in Modal sandbox "
                    f"'{self._sandbox.object_id}': {mkdir_result.stderr}"
                )

        mode = "w" if isinstance(contents, str) else "wb"
        file_handle = await self._sandbox.open.aio(file, mode)
        try:
            await file_handle.write.aio(contents)
            await file_handle.flush.aio()
        finally:
            await file_handle.close.aio()

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> Union[str | bytes]:
        file = self._resolve_file(file)
        mode = "r" if text else "rb"
        file_handle = await self._sandbox.open.aio(file, mode)
        try:
            contents = await file_handle.read.aio()
        except Exception as exc:
            message = str(exc).casefold()
            if "not found" in message or "no such file" in message:
                raise FileNotFoundError(file) from exc
            if "permission denied" in message:
                raise PermissionError(f"Permission denied reading '{file}'.") from exc
            raise RuntimeError(f"Failed to read file '{file}': {exc}") from exc
        finally:
            await file_handle.close.aio()

        if isinstance(contents, str):
            raw = contents.encode("utf-8", errors="replace")
        else:
            raw = bytes(contents)
        _enforce_read_output_limit(raw)

        if text:
            if isinstance(contents, str):
                return contents
            return contents.decode("utf-8")
        else:
            if isinstance(contents, bytes):
                return contents
            return contents.encode("utf-8")

    def default_polling_interval(self) -> float:
        return 1.0

    async def _wait_ready(
        self,
        *,
        timeout_seconds: int,
        poll_interval_seconds: float,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            try:
                remaining = max(1, int(deadline - time.monotonic()))
                probe = await self.exec(
                    ["sh", "-lc", "echo modal-sandbox-ready"],
                    timeout=min(30, remaining),
                )
                if probe.success:
                    return
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(poll_interval_seconds)

        message = (
            f"Modal sandbox '{self._sandbox.object_id}' did not become ready within "
            f"{timeout_seconds}s."
        )
        if last_error is not None:
            message = f"{message} Last error: {last_error}"
        raise TimeoutError(message)

    async def _cleanup(self, *, delete: bool) -> None:
        if not delete:
            return
        try:
            await self._sandbox.terminate.aio(wait=False)
        except Exception:
            pass

    def _resolve_working_dir(self, cwd: str | None) -> str:
        path = self._working_dir if cwd is None else PurePosixPath(cwd)
        if not path.is_absolute():
            path = self._working_dir / path
        return path.as_posix()

    def _resolve_file(self, file: str) -> str:
        path = PurePosixPath(file)
        if not path.is_absolute():
            path = self._working_dir / path
        return path.as_posix()
