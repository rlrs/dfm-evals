# dfm-evals

Minimal `inspect_ai` companion package with:

- `evals` CLI wrapper (forwarding to `inspect`)
- local tasks: `dfm_evals/multi_wiki_qa`, `dfm_evals/bfcl-v1`, `dfm_evals/bfcl-v1-da`, `dfm_evals/ifeval-da`
- LUMI operational helpers under `lumi/`

## LUMI

For container/overlay/sbatch tooling on LUMI, see [`lumi/README.md`](lumi/README.md).
This includes both `dfm-evals` suite workflows and the dedicated 2-node EuroEval launcher.
Local overlay vLLM patch state is documented in the `Overlay vLLM Patches`
section there.

Quick notes:

- Inspect smoke:
  `OVERLAY_DIR=/path/to/overlay_vllm_minimal ./lumi/submit.sh --limit 1 --max-connections 2 --run-label inspect_smoke`
- Throughput tuning guidance (`TP/PP/DP` + `--max-connections`) is in:
  `lumi/README.md` under `Throughput Tuning` (e.g. 4B models can run `TP=1 PP=1 DP=8`).
- `lumi/submit.sh` now launches externally managed vLLM server(s) (target + optional judge)
  rather than relying on Inspect self-spawn.
- EuroEval (example: Danish + longer wall time):
  `OVERLAY_DIR=/path/to/overlay_vllm_minimal ./lumi/euroeval_submit.sh --model Qwen/Qwen3.5-397B-A17B --served-model-name Qwen/Qwen3.5-397B-A17B --euroeval-model Qwen/Qwen3.5-397B-A17B --generative-type reasoning --languages da --iterations 10 --time 12:00:00`
- Job logs:
  `logs/slurm/`
- Inspect eval logs:
  `logs/evals-logs/<run_label>/`
- Every Eval Ever shared data root (standard `data/<benchmark>/<developer>/<model>/` layout):
  `logs/every_eval_ever/data/`
- Inspect results in Every Eval Ever format:
  under `logs/every_eval_ever/data/` (`.json` aggregate + `.jsonl` instance-level)
- EuroEval results:
  `<OVERLAY_DIR>/euroeval-runs/<job_id>/euroeval_benchmark_results.jsonl`
- EuroEval results in Every Eval Ever format:
  under `logs/every_eval_ever/data/` (`.json` aggregate)
- For `openai/*` models, set both:
  `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

## Install

```bash
uv sync
```

## CLI

```bash
uv run evals tasks
uv run evals run dfm_evals/multi_wiki_qa --model openai/gpt-5-mini
uv run evals run dfm_evals/bfcl-v1 --model openai/gpt-5-mini
uv run evals run dfm_evals/ifeval-da --model openai/gpt-5-mini
uv run evals eee inspect --log-path logs/evals-logs/<run_label> --output-dir out/eee/data
uv run evals eee euroeval --results-file /path/to/euroeval_benchmark_results.jsonl --output-dir out/eee/data
# Optional: record inference endpoint/provider context
uv run evals eee inspect --log-path logs/evals-logs/<run_label> --output-dir out/eee/data --inference-base-url https://inference.example/v1 --inference-provider-name my-provider
uv run evals tournament --help
uv run evals tournament view logs/evals-logs/<tournament_run_label>/state --host 127.0.0.1 --port 7576
```

For tournament runs, `evals tournament view` is the main report surface for
standings, head-to-heads, prompt drilldown, and judged responses. `inspect
view` is still useful for raw generation/judge log debugging, but it does not
understand tournament state or final rankings.

Prime Sandbox provider (Inspect `--sandbox prime`):

```bash
PRIME_API_KEY=... \
uv run evals run dfm_evals/multi_wiki_qa \
  --model openai/gpt-5-mini \
  --sandbox prime
```

With explicit config file:

```bash
PRIME_API_KEY=... \
uv run evals run dfm_evals/multi_wiki_qa \
  --model openai/gpt-5-mini \
  --sandbox "prime:$(pwd)/prime-sandbox.yaml"
```

Example `prime-sandbox.yaml`:

```yaml
docker_image: python:3.11-slim
cpu_cores: 2
memory_gb: 4
disk_size_gb: 10
timeout_minutes: 120
working_dir: /workspace
metadata_env_prefix: SAMPLE_METADATA_
```

Cleanup stale Inspect-created Prime sandboxes:

```bash
uv run inspect sandbox cleanup prime
```

Modal Sandbox provider (Inspect `--sandbox modal`):

```bash
MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... \
uv run evals run dfm_evals/multi_wiki_qa \
  --model openai/gpt-5-mini \
  --sandbox modal
```

With explicit config file:

```bash
MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... \
uv run evals run dfm_evals/multi_wiki_qa \
  --model openai/gpt-5-mini \
  --sandbox "modal:$(pwd)/modal-sandbox.yaml"
```

Example `modal-sandbox.yaml`:

```yaml
app_name: inspect-sandboxes
image: python:3.11-slim
startup_command: tail -f /dev/null
working_dir: /workspace
cpu_cores: 1
memory_mb: 2048
timeout_seconds: 3600
wait_for_start: true
wait_timeout_seconds: 900
```

Cleanup stale Inspect-created Modal sandboxes:

```bash
uv run inspect sandbox cleanup modal
```

Suites default to the packaged file at `dfm_evals/eval-sets.yaml`.
Use `--file <path>` for custom suite files.
Packaged suites are provider-agnostic.

```bash
uv run evals suite fundamentals \
  --target-model openai/gpt-5-mini \
  --judge-model openai/gpt-5-mini
```

For external vLLM (recommended), set tool-calling flags on the vLLM server itself:

```bash
vllm serve ../../post \
  --served-model-name gemma3-4b-hermes \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000
```

Then run the suite against that endpoint (for LUMI launcher flags, see `lumi/README.md`):

```bash
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
uv run evals suite fundamentals \
  --target-model openai/gemma3-4b-hermes \
  --judge-model openai/gemma3-4b-hermes
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

Tournament modules live under `dfm_evals/tournament/`.
