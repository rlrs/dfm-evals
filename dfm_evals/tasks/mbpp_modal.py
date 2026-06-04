"""MBPP variant with a configurable sandbox backend."""

from __future__ import annotations

import textwrap

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, prompt_template
from inspect_evals.mbpp.mbpp import (
    DATASET_PATH,
    EVAL_VERSION,
    MBPP_DATASET_REVISION,
    NUM_EPOCHS,
    PROMPT_TEMPLATE,
    hf_dataset,
    load_dataset,
    record_to_sample,
    verify,
)


@task
def mbpp_modal(temperature: float = 0.5, sandbox: str = "modal") -> Task:
    """MBPP using the upstream task logic, with sandbox configurable.

    The upstream inspect_evals MBPP task hardcodes Docker even though scoring only
    needs to execute Python. This wrapper keeps the benchmark behavior intact while
    allowing LUMI jobs to run scoring via inspect-sandboxes Modal.
    """

    template = PROMPT_TEMPLATE
    template += "\n\nFor example:\n\n"

    few_shot_dataset = load_dataset(
        "google-research-datasets/mbpp",
        "full",
        split="prompt",
        revision=MBPP_DATASET_REVISION,
    )
    few_shot_ids = [2, 3, 4]
    few_shot_dataset = few_shot_dataset.filter(
        lambda row: row["task_id"] in few_shot_ids
    )

    for i, sample in enumerate(few_shot_dataset):
        test_cases = "\n".join(sample["test_list"])
        template += "".join(
            [
                f"## Prompt {i + 1}\n",
                "```python\n",
                f"{sample['text']}\n",
                "```\n\n",
                f"## Test Case {i + 1}\n",
                "```python\n",
                f"{test_cases}\n```\n\n",
                f"## Completion {i + 1}\n",
                "```python\n",
                f"{sample['code']}\n```\n\n",
            ]
        )

    template += textwrap.dedent(
        """
        # Now, do it for the following task.

        ## Prompt:
        ```python
        {prompt}
        ```

        ## Test Case:
        ```python
        {test_list_str}
        ```

        ## Completion:
        """
    )

    dataset = hf_dataset(
        path=DATASET_PATH,
        name="sanitized",
        sample_fields=record_to_sample,
        split="test",
        revision=MBPP_DATASET_REVISION,
    )

    return Task(
        dataset=dataset,
        epochs=Epochs(NUM_EPOCHS, ["mean", "pass_at_1", "pass_at_2", "pass_at_5"]),
        solver=[
            prompt_template(template),
            generate(),
        ],
        scorer=verify(),
        config=GenerateConfig(temperature=temperature),
        sandbox=sandbox,
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )
