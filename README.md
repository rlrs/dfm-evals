# dfm-evals

Minimal `inspect_ai` companion package with:

- `evals` CLI wrapper (forwarding to `inspect`)
- local tasks: `dfm_evals/multi_wiki_qa`, `dfm_evals/bfcl-v1`, `dfm_evals/bfcl-v1-da`

## Install

```bash
uv sync
```

## CLI

```bash
uv run evals tasks
uv run evals run dfm_evals/multi_wiki_qa --model openai/gpt-5-mini
uv run evals run dfm_evals/bfcl-v1 --model openai/gpt-5-mini
```

Suites default to the packaged file at `dfm_evals/eval-sets.yaml`.
Use `--file <path>` for custom suite files.
Packaged suites are provider-agnostic.

```bash
uv run evals suite fundamentals \
  --target-model openai/gpt-5-mini \
  --judge-model openai/gpt-5-mini
```

Suites use placeholders in `args`, for example:

- `--model`, `"{{target_model}}"`
- `-T`, `grader_model={{judge_model}}`

Supported placeholders:

- `{{target_model}}`
- `{{target_base_url}}`
- `{{judge_model}}`
- `{{judge_base_url}}`

## Lint

```bash
uv run --group dev ruff check dfm_evals
```

## Task layout

Local tasks live under `dfm_evals/tasks/`.

Current task:

- `dfm_evals/tasks/multi_wiki_qa.py`
- `dfm_evals/tasks/bfcl/`
