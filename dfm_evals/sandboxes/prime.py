from __future__ import annotations

import re
import shlex
import tempfile
import warnings
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Union, overload
from uuid import uuid4

import yaml
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.util._sandbox.environment import (
    SandboxConnection,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
)
from inspect_ai.util._sandbox.limits import (
    OutputLimitExceededError,
    SandboxEnvironmentLimits,
    verify_read_file_size,
)
from inspect_ai.util._sandbox.registry import sandboxenv
from inspect_ai.util._subprocess import ExecResult
from pydantic import BaseModel, Field, model_validator
from typing_extensions import override

if TYPE_CHECKING:
    from prime_sandboxes import AsyncSandboxClient


DEFAULT_PRIME_CONFIG_FILES = [
    "configs/sandboxes/prime-sandbox.yaml",
    "configs/sandboxes/prime-sandbox.yml",
    "configs/sandboxes/prime-sandbox.json",
    "prime-sandbox.yaml",
    "prime-sandbox.yml",
    "prime-sandbox.json",
]


class PrimeSandboxConfig(BaseModel):
    docker_image: str = "python:3.11-slim"
    start_command: str | None = "tail -f /dev/null"
    working_dir: str | None = None

    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    disk_size_gb: float = 5.0
    gpu_count: int = 0
    gpu_type: str | None = None
    network_access: bool = True
    timeout_minutes: int = 60

    wait_for_creation: bool = True
    wait_max_attempts: int = 60
    wait_stability_checks: int = 1

    name_prefix: str = "inspect"
    keep_sandbox: bool = False
    labels: list[str] = Field(default_factory=list)

    environment_vars: dict[str, str] = Field(default_factory=dict)
    metadata_env_prefix: str | None = None
    secrets: dict[str, str] | None = None

    team_id: str | None = None
    registry_credentials_id: str | None = None
    api_key: str | None = None

    @model_validator(mode="after")
    def validate_values(self) -> "PrimeSandboxConfig":
        if self.gpu_count > 0 and not self.gpu_type:
            raise ValueError("`gpu_type` is required when `gpu_count` is greater than 0.")
        if self.wait_max_attempts < 1:
            raise ValueError("`wait_max_attempts` must be >= 1.")
        if self.wait_stability_checks < 1:
            raise ValueError("`wait_stability_checks` must be >= 1.")
        return self


def _require_prime_sandboxes() -> Any:
    try:
        import prime_sandboxes
    except ModuleNotFoundError as exc:
        raise PrerequisiteError(
            "Prime sandbox integration requires `prime_sandboxes` (install `prime-sandboxes`)."
        ) from exc
    return prime_sandboxes


def _load_config_from_path(path: str) -> PrimeSandboxConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(
            f"Prime sandbox config file '{config_path.as_posix()}' does not exist."
        )

    text = config_path.read_text(encoding="utf-8")
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse Prime sandbox config file '{config_path.as_posix()}': {exc}"
        ) from exc

    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Prime sandbox config file '{config_path.as_posix()}' must contain a mapping."
        )

    return PrimeSandboxConfig.model_validate(parsed)


def _resolve_prime_config(
    config: SandboxEnvironmentConfigType | None,
) -> PrimeSandboxConfig:
    if config is None:
        return PrimeSandboxConfig()
    if isinstance(config, PrimeSandboxConfig):
        return config
    if isinstance(config, BaseModel):
        return PrimeSandboxConfig.model_validate(config.model_dump())
    if isinstance(config, str):
        return _load_config_from_path(config)
    raise ValueError(f"Unsupported Prime sandbox config type: {type(config)!r}")


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


@sandboxenv(name="prime")
class PrimeSandboxEnvironment(SandboxEnvironment):
    @classmethod
    def config_files(cls) -> list[str]:
        return DEFAULT_PRIME_CONFIG_FILES

    @classmethod
    def config_deserialize(cls, config: dict[str, Any]) -> BaseModel:
        return PrimeSandboxConfig.model_validate(config)

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        cfg = _resolve_prime_config(config)
        prime = _require_prime_sandboxes()

        client: AsyncSandboxClient = prime.AsyncSandboxClient(api_key=cfg.api_key)
        sandbox_id: str | None = None
        try:
            labels = ["inspect-ai", f"inspect-task-{_sanitize_label(task_name)}", *cfg.labels]
            environment_vars = {
                **_metadata_env(metadata, cfg.metadata_env_prefix),
                **cfg.environment_vars,
            }
            request = prime.CreateSandboxRequest(
                name=_build_sandbox_name(task_name, cfg.name_prefix),
                docker_image=cfg.docker_image,
                start_command=cfg.start_command,
                cpu_cores=cfg.cpu_cores,
                memory_gb=cfg.memory_gb,
                disk_size_gb=cfg.disk_size_gb,
                gpu_count=cfg.gpu_count,
                gpu_type=cfg.gpu_type,
                network_access=cfg.network_access,
                timeout_minutes=cfg.timeout_minutes,
                environment_vars=environment_vars or None,
                secrets=cfg.secrets,
                labels=labels,
                team_id=cfg.team_id,
                registry_credentials_id=cfg.registry_credentials_id,
            )
            sandbox = await client.create(request)
            sandbox_id = sandbox.id

            if cfg.wait_for_creation:
                await client.wait_for_creation(
                    sandbox_id,
                    max_attempts=cfg.wait_max_attempts,
                    stability_checks=cfg.wait_stability_checks,
                )

            working_dir = cfg.working_dir or sandbox.disk_mount_path or "/tmp"
            environment = PrimeSandboxEnvironment(
                client=client,
                sandbox_id=sandbox_id,
                working_dir=working_dir,
                keep_sandbox=cfg.keep_sandbox,
            )

            ensure_working_dir = await environment.exec(
                ["mkdir", "-p", working_dir],
                timeout=60,
            )
            if not ensure_working_dir.success:
                raise RuntimeError(
                    f"Failed to create working directory '{working_dir}' in Prime sandbox "
                    f"'{sandbox_id}': {ensure_working_dir.stderr}"
                )

            return {"default": environment}
        except BaseException:
            if sandbox_id is not None:
                try:
                    await client.delete(sandbox_id)
                except Exception:
                    pass
            await client.aclose()
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
            prime_environment = environment.as_type(PrimeSandboxEnvironment)
            await prime_environment._cleanup(delete=not prime_environment._keep_sandbox)

    @override
    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        prime = _require_prime_sandboxes()
        client = prime.AsyncSandboxClient()
        try:
            if id is not None:
                await client.delete(id)
                return

            page = 1
            while True:
                response = await client.list(
                    labels=["inspect-ai"],
                    page=page,
                    per_page=50,
                    exclude_terminated=True,
                )
                for sandbox in response.sandboxes:
                    try:
                        await client.delete(sandbox.id)
                    except Exception:
                        pass
                if not response.has_next:
                    break
                page += 1
        finally:
            await client.aclose()

    def __init__(
        self,
        *,
        client: AsyncSandboxClient,
        sandbox_id: str,
        working_dir: str,
        keep_sandbox: bool = False,
    ) -> None:
        self._client = client
        self._sandbox_id = sandbox_id
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
            raise ValueError("Prime sandbox `exec` requires a non-empty command list.")

        if user is not None:
            warnings.warn(
                "The 'user' parameter is ignored in PrimeSandboxEnvironment.",
                UserWarning,
            )

        prime = _require_prime_sandboxes()
        working_dir = self._resolve_working_dir(cwd)
        command = shlex.join(cmd)

        stdin_path: str | None = None
        if input is not None:
            stdin_path = f"/tmp/inspect-stdin-{uuid4().hex}"
            payload = input.encode("utf-8") if isinstance(input, str) else input
            await self._client.upload_bytes(
                sandbox_id=self._sandbox_id,
                file_path=stdin_path,
                file_bytes=payload,
                filename="stdin",
                timeout=120,
            )
            command = f"{command} < {shlex.quote(stdin_path)}"

        try:
            result = await self._client.execute_command(
                sandbox_id=self._sandbox_id,
                command=command,
                working_dir=working_dir,
                env=env,
                timeout=timeout,
            )
        except prime.CommandTimeoutError as exc:
            raise TimeoutError(str(exc)) from exc
        except prime.APIError as exc:
            raise RuntimeError(str(exc)) from exc
        finally:
            if stdin_path is not None:
                try:
                    await self._client.execute_command(
                        sandbox_id=self._sandbox_id,
                        command=f"rm -f {shlex.quote(stdin_path)}",
                        timeout=30,
                    )
                except Exception:
                    pass

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        _enforce_exec_output_limit(stdout)
        _enforce_exec_output_limit(stderr)

        return ExecResult(
            success=result.exit_code == 0,
            returncode=result.exit_code,
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
                    f"Failed to create directory '{parent}' in Prime sandbox "
                    f"'{self._sandbox_id}': {mkdir_result.stderr}"
                )

        payload = contents.encode("utf-8") if isinstance(contents, str) else contents
        filename = PurePosixPath(file).name or "file"

        prime = _require_prime_sandboxes()
        try:
            await self._client.upload_bytes(
                sandbox_id=self._sandbox_id,
                file_path=file,
                file_bytes=payload,
                filename=filename,
                timeout=120,
            )
        except prime.APIError as exc:
            message = str(exc).casefold()
            if "permission denied" in message or "forbidden" in message:
                raise PermissionError(f"Permission denied writing '{file}'.") from exc
            if "directory" in message and "exists" in message:
                raise IsADirectoryError(f"Cannot overwrite directory '{file}'.") from exc
            raise RuntimeError(f"Failed to write file '{file}': {exc}") from exc

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> Union[str | bytes]:
        file = self._resolve_file(file)
        prime = _require_prime_sandboxes()

        with tempfile.NamedTemporaryFile(delete=False) as local_file:
            local_path = Path(local_file.name)

        try:
            await self._client.download_file(
                sandbox_id=self._sandbox_id,
                file_path=file,
                local_file_path=local_path.as_posix(),
                timeout=120,
            )
            verify_read_file_size(local_path.as_posix())

            if text:
                with open(local_path, "r", newline="", encoding="utf-8") as f:
                    return f.read()
            else:
                with open(local_path, "rb") as f:
                    return f.read()
        except prime.APIError as exc:
            message = str(exc).casefold()
            if "404" in message or "not found" in message:
                raise FileNotFoundError(file) from exc
            if "permission denied" in message or "forbidden" in message:
                raise PermissionError(f"Permission denied reading '{file}'.") from exc
            raise RuntimeError(f"Failed to read file '{file}': {exc}") from exc
        finally:
            local_path.unlink(missing_ok=True)

    @override
    async def connection(self, *, user: str | None = None) -> SandboxConnection:
        if user is not None:
            warnings.warn(
                "The 'user' parameter is ignored in PrimeSandboxEnvironment connection.",
                UserWarning,
            )
        return SandboxConnection(
            type="prime",
            command=f"prime sandboxes ssh {self._sandbox_id}",
        )

    def default_polling_interval(self) -> float:
        return 1.0

    async def _cleanup(self, *, delete: bool) -> None:
        try:
            if delete:
                await self._client.delete(self._sandbox_id)
        finally:
            await self._client.aclose()

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
