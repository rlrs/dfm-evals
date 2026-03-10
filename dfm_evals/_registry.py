"""Inspect registry imports for dfm_evals."""

from __future__ import annotations

from .hf_eval_yaml import install_hf_eval_yaml_extensions
from .sandboxes import prime
from .scorers import comet, gleu
from .tasks import bfcl, bfcl_da, ifeval_da, multi_wiki_qa, piqa
from .tournament.scorer import decision_valid_rate, pairwise_judge

__all__ = [
    "multi_wiki_qa",
    "bfcl",
    "bfcl_da",
    "ifeval_da",
    "piqa",
    "gleu",
    "comet",
    "prime",
    "pairwise_judge",
    "decision_valid_rate",
]


# Ensure hf/... task specs use the dfm_evals extended eval.yaml loader.
install_hf_eval_yaml_extensions()
