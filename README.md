# dfm-evals

Minimal `inspect_ai` companion package with:

- `evals` CLI wrapper (forwarding to `inspect`)
- one local task: `dfm_evals/multi_wiki_qa`

## Install

```bash
uv sync
```

## CLI

```bash
uv run evals tasks
uv run evals run dfm_evals/multi_wiki_qa --model openai/gpt-4o-mini
uv run evals suites
uv run evals suite multi-wiki-qa-smoke -- --model openai/gpt-4o-mini
```

Suites default to the packaged file at `dfm_evals/eval-sets.yaml`.
Use `--file <path>` for custom suite files.

## Lint

```bash
uv run --group dev ruff check dfm_evals
```

## Task layout

Local tasks live under `dfm_evals/tasks/`.

Current task:

- `dfm_evals/tasks/multi_wiki_qa.py`
