# LUMI Toolkit

LUMI-specific helpers for running `dfm-evals` with inspect + vLLM in the LUMI container environment.
The suite launcher now manages vLLM servers explicitly (target server + optional judge server),
instead of relying on Inspect self-spawn.

## Files

- `lumi/build_overlay_minimal.sh`: build/update overlay venv with vLLM + dependencies.
- `lumi/submit.sh`: submit suite runs via `sbatch`.
- `lumi/run_suite.sbatch`: batch job entrypoint used by `submit.sh`.
- `lumi/tournament_submit.sh`: submit phase-based tournament runs via `sbatch`.
- `lumi/run_tournament.sbatch`: batch job entrypoint used by `tournament_submit.sh`.
- `lumi/tournament_launch.py`: validate tournament launch-maps and emit shell-safe runtime config.
- `lumi/euroeval_submit.sh`: submit 2-node vLLM + EuroEval jobs.
- `lumi/run_euroeval.sbatch`: 2-node vLLM MP launcher with optional EuroEval.
- `lumi/view.sh`: inspect-view helper (default log root: `logs/evals-logs`).
- `lumi/results_table.sh`: aggregate Every Eval Ever metrics into terminal table/CSV/JSON.

## Quick Start

From repository root:

```bash
# 1) Configure credentials (if using openai/* models)
cat > .env <<'EOF'
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
EOF

# 2) Build overlay (first time or after runtime updates)
./lumi/build_overlay_minimal.sh

# 3) Submit fundamentals suite (default: gemma target + judge)
./lumi/submit.sh --limit 100
```

## Update `dfm-evals` In The Overlay

When you change this checkout and want LUMI jobs to pick it up, reinstall the
repo inside the overlay venv before submitting. Use the helper script:

```bash
./lumi/install_overlay_dfm_evals.sh --extras harbor,sandboxes
```

Notes:

- For later code-only updates, `./lumi/install_overlay_dfm_evals.sh` is enough.
  That defaults to `pip install -e . --no-deps`.
- Use `--extras harbor,sandboxes` when you need `inspect_harbor` and
  `inspect_sandboxes` present in the overlay. That resolves dependencies.
- Do not run `<OVERLAY_DIR>/venv/vllm-min/bin/pip` directly from the host. The
  wrapper points at `/overlay/...` and only works from inside the container.
- The helper binds the repo into the container at the same absolute path the
  Slurm jobs use (`-B "$REPO_ROOT:$REPO_ROOT"`). That matters for editable
  installs: the old `/workspace` pattern is wrong because job containers do not
  mount the repo there.
- `lumi/submit.sh` expects `evals` to already be installed in the overlay venv;
  it does not reinstall this repo automatically on job start.

## Recommended Commands

Inspect smoke (fast validation, externally managed target vLLM):

```bash
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal \
./lumi/submit.sh --limit 1 --max-connections 2 --run-label inspect_smoke
```

Throughput-oriented run for small models (example: 4B on 1 node / 8 GPUs):

```bash
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal \
./lumi/submit.sh \
  --model ../../post/outputs/sft-16292768/final \
  --tp 1 --pp 1 --dp 8 \
  --max-connections 128 \
  --limit 100
```

Inspect with a local LoRA adapter directory (auto-detect adapter, serve base model, enable LoRA):

```bash
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal \
./lumi/submit.sh \
  --model ../../post/outputs/sft-16292768/final \
  --limit 100
```

If the adapter directory contains tokenizer artifacts (for example
`tokenizer_config.json` or `tokenizer.json`), the launch scripts pass that
directory as `--tokenizer` so vLLM uses the adapter tokenizer.

## Tournament

Tournament uses a separate launcher instead of `lumi/submit.sh` because it is
multi-contestant, stateful, and phase-based.

Committed tournament definitions live under `configs/tournaments/<name>/` and
typically include:

```text
configs/tournaments/creative-writing-da-smoke/
  tournament.yaml
  launch-map.yaml
```

The core tournament definition is YAML and can keep prompts external via
`prompt_source`, so you do not need to hand-edit large inline prompt arrays.

Minimal launch-map shape:

```yaml
defaults:
  tp: 1
  dp: 8
contestants:
  vllm/org/model-a:
    mode: local_vllm
    model: ../../post/outputs/sft-16292768/final
  openai/model-b:
    mode: external_openai
    base_url_env: MODEL_B_BASE_URL
    api_key_env: MODEL_B_API_KEY
judge:
  mode: local_vllm
  model: google/gemma-3-27b-it
```

Common flows:

```bash
# Generate contestant responses into a new run
./lumi/tournament_submit.sh \
  --phase generate \
  --definition creative-writing-da-smoke

# Full flow with an overridden contestant roster and judge model
./lumi/tournament_submit.sh \
  --phase all \
  --definition creative-writing-da-smoke \
  --contestant-model vllm/google/gemma-3-4b-it \
  --contestant-model vllm/google-gemma-3-4b-pt-hermes-final \
  --judge-model openai/qwen-235b

# Run or resume judging against an existing tournament run dir
./lumi/tournament_submit.sh \
  --phase run \
  --target ./logs/evals-logs/tournament__demo__job-123 \
  --launch-map ./configs/tournaments/creative-writing-da-smoke

# Export tournament artifacts plus Every Eval Ever JSON
./lumi/tournament_submit.sh \
  --phase export \
  --target ./logs/evals-logs/tournament__demo__job-123
```

Notes:

- `--phase all` runs `generate -> run -> export`.
- `--definition <name>` resolves committed definitions from `configs/tournaments/<name>/`.
- `--contestant-model` and `--judge-model` override the committed model lineup without editing the definition YAML.
- contestant endpoints are launched one model at a time from the launch-map.
- `run` and `resume` require complete generation coverage and will fail fast if responses are missing.
- raw contestant and judge server logs are written under `logs/evals-logs/<run_label>/services/vllm/`.

Hosted tournament viewer:

```bash
uv run evals tournament view \
  logs/evals-logs/tournament__demo__job-123 \
  --host 127.0.0.1 \
  --port 7576
```

Use the hosted tournament viewer for standings, pairwise results, prompts, and
response drilldown. `lumi/view.sh` still launches `inspect view`, which is
useful for raw generation/judge batch logs but is not the top-level tournament
report.

## Tool Calling Defaults

For suite runs (`lumi/submit.sh`), the target vLLM server now defaults to:

- `--enable-auto-tool-choice`
- `--tool-call-parser hermes`

This is only a baseline default. Tool-call parser choice is model-dependent and
must match the model's tool-call output format.

Override per run:

```bash
./lumi/submit.sh --target-tool-call-parser qwen3
./lumi/submit.sh --target-disable-auto-tool-choice
./lumi/submit.sh --target-enable-auto-tool-choice --target-tool-call-parser llama3_json
```

Judge server controls (when `--judge-server` is used):

```bash
./lumi/submit.sh --judge-enable-auto-tool-choice --judge-tool-call-parser hermes
```

For reliable tool calling, your model/template should explicitly handle tool
schemas and tool-call turns (`tools`, assistant `tool_calls`, and `tool` role
messages). A plain chat-only template will not be sufficient for robust
auto-tool-choice behavior.

Run with a dedicated judge replica (separate vLLM server + GPUs):

```bash
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal \
./lumi/submit.sh \
  --model ../../post/outputs/sft-16292768/final \
  --tp 1 --dp 6 --target-devices 0,1,2,3,4,5 \
  --judge-model openai/google/gemma-3-4b-pt \
  --judge-server --judge-server-model google/gemma-3-4b-pt \
  --judge-dp 2 --judge-devices 6,7 \
  --limit 100
```

EuroEval smoke on smaller model:

```bash
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal \
./lumi/euroeval_submit.sh \
  --model google/gemma-3-4b-it \
  --served-model-name google/gemma-3-4b-it \
  --euroeval-model google/gemma-3-4b-it \
  --tp 2 --pp 1 --ctx 4096 \
  --languages en --tasks knowledge --iterations 1
```

EuroEval on Qwen 3.5 397B, Danish, longer wall time:

```bash
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal \
./lumi/euroeval_submit.sh \
  --model Qwen/Qwen3.5-397B-A17B \
  --served-model-name Qwen/Qwen3.5-397B-A17B \
  --euroeval-model Qwen/Qwen3.5-397B-A17B \
  --generative-type reasoning \
  --languages da \
  --iterations 10 \
  --time 12:00:00
```

## Throughput Tuning

Choose `TP/PP/DP` and `--max-connections` per run, based on model size and node/GPU count.

- Small models (e.g. ~4B) on 1 node with 8 GPUs:
  Prefer data parallel fan-out (`TP=1`, `PP=1`, `DP=8`) for best throughput.
- Larger models that do not fit on one GPU:
  Increase `TP` (and `PP` when needed), then use remaining GPUs for `DP`.
- `--max-connections`:
  Set high enough to keep workers busy (typically `100+` on a full node), but not so high that request queues or memory become unstable.

Practical loop:

1. Start with `TP=1 PP=1 DP=<gpus_per_node>` for small models.
2. Increase `--max-connections` (e.g. `128`, `256`) while watching latency and error rate.
3. For large models, first satisfy memory with TP/PP, then allocate leftover GPUs to DP.

When using a separate judge server, reserve GPU slices explicitly with
`--target-devices` and `--judge-devices` to avoid resource contention.

## External Judge Example

```bash
./lumi/submit.sh \
  --target-model vllm/google/gemma-3-4b-it \
  --judge-model openai/<model> \
  --limit 100
```

For `openai/*`, `OPENAI_BASE_URL` must be configured (no fallback is used).
`submit.sh` now defaults judge model to `openai/<target-model-name>` unless
`--judge-model` is provided explicitly.

## EuroEval

Run the dedicated 2-node vLLM MP + EuroEval workflow:

```bash
./lumi/euroeval_submit.sh
```

Current defaults in this repo:

- `--languages da`
- `--iterations 1`
- `--datasets ifeval-da,danish-citizen-tests,danske-talemaader,multi-wiki-qa-da,dala`
- `--custom-datasets-file ./lumi/euroeval_custom_datasets.py` (adds `dala`)

Common examples:

```bash
./lumi/euroeval_submit.sh --datasets ifeval-da,danish-citizen-tests,danske-talemaader,multi-wiki-qa-da,dala
./lumi/euroeval_submit.sh --all-datasets --languages da
./lumi/euroeval_submit.sh --all-datasets --tasks "knowledge,summarization"
./lumi/euroeval_submit.sh --generative-type reasoning
./lumi/euroeval_submit.sh --time 12:00:00
./lumi/euroeval_submit.sh --no-euroeval
```

`--generative-type` accepts `auto|base|instruction_tuned|reasoning`.
When set to `auto` (default), Qwen3/Qwen3.5 models are auto-resolved to
`reasoning` to avoid short non-reasoning token budgets truncating outputs.

EuroEval artifacts are written to these default paths from `run_euroeval.sbatch`:

- `/overlay/euroeval-cache-<job_id>`
- `/overlay/euroeval-runs/<job_id>/euroeval_benchmark_results.jsonl`
- `<REPO_ROOT>/logs/every_eval_ever/data/` (shared converted EEE data root)

The `/overlay/...` paths are container paths bind-mounted from host `OVERLAY_DIR`.

EEE export is enabled by default in both launchers:

- `DFM_EVALS_EXPORT_EEE=1` in `run_suite.sbatch`
- `EUROEVAL_EXPORT_EEE=1` in `run_euroeval.sbatch`

Set either to `0` to disable conversion for a run.

When available, launcher-known inference endpoints are written to EEE metadata
(`generation_config.additional_details.inference_base_url` and `inference_host`).

## Inspect View

```bash
./lumi/view.sh list
./lumi/view.sh start --latest
./lumi/view.sh start --job-id <job_id>
```

By default, `view.sh` reads from `logs/evals-logs/`.

## Results Table

```bash
./lumi/results_table.sh --latest
./lumi/results_table.sh --compare-models --all-runs
./lumi/results_table.sh --compare-models --all-runs --task-rows
./lumi/results_table.sh --compare-models --all-runs --all-metrics --format csv
```

`results_table.sh` reads EEE `.json` records under `logs/every_eval_ever/data/`
and prints task/scorer/metric aggregates. Use `--compare-models` for model-vs-task tables
(default orientation: model rows, task columns; use `--task-rows` for the old layout).

## Log Locations

Default eval artifact roots:

- `logs/evals-logs/<run_label>/`
- `logs/evals-logs/<run_label>/config/runtime.json`
- `logs/evals-logs/<run_label>/inspect/` (generation + judge Inspect logs)
- `logs/evals-logs/<run_label>/services/vllm/` (launcher-managed vLLM raw logs)
- `logs/every_eval_ever/data/` (shared converted EEE JSON + Inspect instance-level JSONL)

Overlay still holds runtime environment assets (`venv`, source checkouts, cache).

Slurm stdout/stderr from `lumi/submit.sh` default to:

- `logs/slurm/<suite_or_run_label>-<job_id>.out`
- `logs/slurm/<suite_or_run_label>-<job_id>.err`

EuroEval submit/run logs default to:

- `logs/slurm/euroeval__<served_model_name>-<job_id>.out`
- `logs/slurm/euroeval__<served_model_name>-<job_id>.err`
- `logs/slurm/vllm-q35-mp1-rank-<rank>-<job_id>.log`
- `logs/slurm/completion-qwen35-mp1-<job_id>.json`

Override with:

- `--slurm-log-dir <path>` on `lumi/submit.sh`
- `--slurm-log-dir <path>` on `lumi/euroeval_submit.sh`
- `--eee-output-dir <path>` on `lumi/submit.sh` or `lumi/euroeval_submit.sh`
- or `SLURM_LOG_DIR=<path>` in the environment
- or `DFM_EVALS_EEE_OUTPUT_DIR=<path>` / `EUROEVAL_EEE_OUTPUT_DIR=<path>` in the environment

## Overlay vLLM Patches

The runtime overlay sources live in `<OVERLAY_DIR>/src/vllm` and can contain
local patches that are separate from this `dfm-evals` repo.

Current known patches:

- Upstream cherry-pick:
  `40f88d831` (`[Bugfix] Fix Qwen3/Qwen3.5 Reasoning Parser (#34779)`), touching:
  - `vllm/reasoning/qwen3_reasoning_parser.py`
  - `vllm/entrypoints/openai/chat_completion/serving.py`
  - `tests/reasoning/test_qwen3_reasoning_parser.py`
- Local multi-node SHM locality fix:
  - `vllm/distributed/device_communicators/shm_broadcast.py`
  - switches same-node detection from rank math to hostname-based process-group detection.
- Local Qwen3.5 config compatibility fixes:
  - `vllm/transformers_utils/configs/qwen3_5.py`
  - `vllm/transformers_utils/configs/qwen3_5_moe.py`
  - changes `ignore_keys_at_rope_validation` from list to set.

## Overlay EuroEval Patches

EuroEval is installed inside the overlay venv at:

- `<OVERLAY_DIR>/venv/vllm-min/lib/python3.12/site-packages/euroeval`

Current known local patches (against `euroeval==16.15.0` wheel):

- `euroeval/__init__.py`
  - allows `flash_attn` when `EUROEVAL_ALLOW_FLASH_ATTN=1` (default `1` in this overlay patch),
    instead of hard-failing import.
- `euroeval/benchmark_modules/litellm.py`
  - routes LiteLLM `Router` calls using the cleaned `model_id` consistently
    (`model_name=model_id`, `model=model_id`) to avoid model-ID mismatch issues.

Inspect patch state:

```bash
git -C "$OVERLAY_DIR/src/vllm" status --short
git -C "$OVERLAY_DIR/src/vllm" diff --name-status
git -C "$OVERLAY_DIR/src/vllm" diff --cached --name-status
```

Inspect EuroEval wheel drift (files modified from installed RECORD hashes):

```bash
python - <<'PY'
import base64, csv, hashlib
import os
from pathlib import Path
overlay_dir = Path(os.environ["OVERLAY_DIR"])
sp = overlay_dir / "venv/vllm-min/lib/python3.12/site-packages"
record = sp / "euroeval-16.15.0.dist-info" / "RECORD"
for path, hashfield, _ in csv.reader(record.read_text().splitlines()):
    if not hashfield:
        continue
    p = sp / path
    algo, expected = hashfield.split("=", 1)
    if algo != "sha256" or not p.exists():
        continue
    got = base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).rstrip(b"=").decode()
    if got != expected:
        print(path)
PY
```

If these should be tracked in git, commit them in their owning repos:

- vLLM changes in `$OVERLAY_DIR/src/vllm`
- EuroEval changes in a EuroEval source checkout (site-packages is not a git repo)

## Monitoring

Use these during execution:

```bash
squeue -j <job_id> -o '%i %T %M %D %R %j'
tail -f logs/slurm/<logfile>.out
```

After completion:

```bash
sacct -j <job_id> --format=JobID,JobName%30,State,Elapsed,ExitCode -P
```

Inspect success checks:

```bash
find logs/evals-logs/<run_label>/inspect -name '*.eval'
ls logs/evals-logs/<run_label>/services/vllm/
```

EuroEval success checks:

```bash
grep -E 'Server ready|EuroEval complete' logs/slurm/euroeval__*.out
ls /pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal/euroeval-runs/<job_id>/euroeval_benchmark_results.jsonl
ls logs/every_eval_ever/data/
```

`sacct` may show the `.0` step as cancelled during scripted shutdown while the top-level job is still `COMPLETED`; use the top-level job state/exit code as the source of truth.

## Common Failure Modes

- `overlay dir not found`: set `OVERLAY_DIR` explicitly to your existing overlay location.
- `openai/*` fails fast: set both `OPENAI_API_KEY` and `OPENAI_BASE_URL` (or pass `--openai-base-url`).
- `inspect_harbor/*` tasks fail on the current LUMI/Prime path: Harbor emits Docker/Compose sandbox specs, and this repo's built-in `prime` backend does not translate them. Use a Docker-capable runtime or a compose-aware provider such as `inspect_sandboxes` `modal` outside the current LUMI launcher flow.
- The packaged `openthoughts_tblite` suite now runs Harbor with `--no-fail-on-error --continue-on-fail`, so a single bad sample does not abort the full eval.
- Harbor sample parallelism follows `--max-connections` unless you explicitly pass `--max-samples`; a smoke run with `--max-connections 1` is intentionally single-sample-at-a-time.
- The LUMI launcher now exports model-info overrides for custom `vllm/*` served names, so Inspect compaction uses the actual `--ctx` value instead of falling back to `128000`.
- vLLM startup fails with low free GPU memory: lower `GPU_MEM`, reduce `TP/PP`, or retry on a cleaner node allocation.
- `view.sh list` shows no runs: new default root is `logs/evals-logs`; for older overlay runs, use `EVAL_LOG_ROOT_HOST=<overlay>/dfm-evals-logs ./lumi/view.sh list`.
- Wrong EuroEval model due inherited env: pass `--euroeval-model` explicitly (recommended for reproducibility).
