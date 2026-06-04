from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Any


def apply_qwen35_text_only_patch() -> list[str]:
    if os.environ.get("DFM_EVALS_VLLM_QWEN35_SKIP_VISUAL_WEIGHTS") != "1":
        return []

    applied: list[str] = []

    from vllm.config import ModelConfig
    from vllm.model_executor.model_loader import default_loader
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from vllm.model_executor.models.utils import AutoWeightsLoader

    if not getattr(Qwen3_5ForConditionalGeneration, "_dfm_skip_visual_weights", False):

        def load_weights(self, weights):
            loader = AutoWeightsLoader(self, skip_prefixes=["mtp.", "visual."])
            return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        Qwen3_5ForConditionalGeneration.load_weights = load_weights
        Qwen3_5ForConditionalGeneration._dfm_skip_visual_weights = True
        applied.append("qwen3.5-skip-visual-loader")

    default_loader_cls = default_loader.DefaultModelLoader
    if not getattr(default_loader_cls, "_dfm_skip_visual_weights", False):

        def load_weights(
            self,
            model: Any,
            model_config: ModelConfig,
        ) -> None:
            if model_config.quantization == "torchao":
                quant_config = default_loader.get_quant_config(
                    model_config,
                    self.load_config,
                )
                if (
                    hasattr(quant_config, "is_checkpoint_torchao_serialized")
                    and quant_config.is_checkpoint_torchao_serialized
                    and default_loader.torchao_version_at_least("0.15.0")
                ):
                    self.load_config.safetensors_load_strategy = "torchao"

            self._init_ep_weight_filter(model_config)

            weights_to_load = {
                name
                for name, _ in model.named_parameters()
                if not name.startswith("visual.")
            }
            loaded_weights = model.load_weights(
                self.get_all_weights(model_config, model)
            )

            self.counter_after_loading_weights = time.perf_counter()
            default_loader.logger.info_once(
                "Loading weights took %.2f seconds",
                self.counter_after_loading_weights
                - self.counter_before_loading_weights,
                scope="local",
            )
            if model_config.quantization is None and loaded_weights is not None:
                weights_not_loaded = weights_to_load - set(
                    _without_visual_weights(loaded_weights)
                )
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from "
                        f"checkpoint: {weights_not_loaded}"
                    )

        default_loader_cls.load_weights = load_weights
        default_loader_cls._dfm_skip_visual_weights = True
        applied.append("qwen3.5-skip-visual-strict-check")

    return applied


def _without_visual_weights(weights: Iterable[str]) -> Iterable[str]:
    for weight in weights:
        if not weight.startswith("visual."):
            yield weight
