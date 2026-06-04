from __future__ import annotations

import sys

from dfm_evals.vllm_qwen35_text_only import apply_qwen35_text_only_patch
from dfm_evals.vllm_patches import apply_runtime_thread_safety_patches


def apply_ministral3_config_patch() -> list[str]:
    try:
        from transformers import AutoConfig, MistralConfig
        try:
            from transformers.models.ministral3.configuration_ministral3 import (
                Ministral3Config,
            )
        except Exception:  # noqa: BLE001
            class Ministral3Config(MistralConfig):
                model_type = "ministral3"

        try:
            AutoConfig.register("ministral3", Ministral3Config, exist_ok=True)
        except TypeError:
            AutoConfig.register("ministral3", Ministral3Config)
    except Exception as exc:  # noqa: BLE001
        return [f"ministral3_config_patch_failed:{type(exc).__name__}"]
    return ["ministral3_config"]


def main() -> None:
    applied = apply_runtime_thread_safety_patches()
    applied.extend(apply_qwen35_text_only_patch())
    applied.extend(apply_ministral3_config_patch())
    print(
        "dfm_evals runtime patches: " + ", ".join(applied),
        file=sys.stderr,
        flush=True,
    )

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
