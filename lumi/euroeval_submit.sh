#!/bin/bash
# Friendly wrapper for submitting 2-node vLLM + EuroEval jobs on LUMI.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_SUBMIT_SCRIPT="$SCRIPT_DIR/run_euroeval.sbatch"
ENV_FILE=${ENV_FILE:-$REPO_ROOT/.env}

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

SUBMIT_SCRIPT=${SUBMIT_SCRIPT:-$DEFAULT_SUBMIT_SCRIPT}
OVERLAY_DIR=${OVERLAY_DIR:-$REPO_ROOT/overlay_vllm_minimal}
if [[ ! -d "$OVERLAY_DIR" && -d "$REPO_ROOT/../overlay_vllm_minimal" ]]; then
  OVERLAY_DIR="$REPO_ROOT/../overlay_vllm_minimal"
fi

MODEL=${MODEL:-Qwen/Qwen3.5-397B-A17B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
PORT=${PORT:-8000}
TP=${TP:-4}
PP=${PP:-4}
CTX=${CTX:-8192}
GPU_MEM=${GPU_MEM:-0.92}

RUN_EUROEVAL=${RUN_EUROEVAL:-1}
EUROEVAL_MODEL=${EUROEVAL_MODEL:-$SERVED_MODEL_NAME}
EUROEVAL_LANGUAGES=${EUROEVAL_LANGUAGES:-da}
DEFAULT_EUROEVAL_DATASETS="ifeval-da,danish-citizen-tests,danske-talemaader,multi-wiki-qa-da,dala"
EUROEVAL_DATASETS=${EUROEVAL_DATASETS:-$DEFAULT_EUROEVAL_DATASETS}
EUROEVAL_TASKS=${EUROEVAL_TASKS:-}
EUROEVAL_NUM_ITERATIONS=${EUROEVAL_NUM_ITERATIONS:-1}
EUROEVAL_CONCURRENCY_MODE=${EUROEVAL_CONCURRENCY_MODE:-adaptive}
EUROEVAL_MAX_CONCURRENT_CALLS=${EUROEVAL_MAX_CONCURRENT_CALLS:-1000}
EUROEVAL_GENERATIVE_TYPE=${EUROEVAL_GENERATIVE_TYPE:-auto}
EUROEVAL_EXTRA_ARGS=${EUROEVAL_EXTRA_ARGS:-}
EUROEVAL_CUSTOM_DATASETS_FILE=${EUROEVAL_CUSTOM_DATASETS_FILE:-$REPO_ROOT/lumi/euroeval_custom_datasets.py}
EUROEVAL_EEE_OUTPUT_DIR=${EUROEVAL_EEE_OUTPUT_DIR:-}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-$REPO_ROOT/logs/slurm}
TIME_LIMIT=${TIME_LIMIT:-}
NODES=${NODES:-}

DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./lumi/euroeval_submit.sh [options]

Options:
  --model <model>            Base served model (default: Qwen/Qwen3.5-397B-A17B)
  --served-model-name <id>   Served model name (default: <model>)
  --port <n>                 vLLM API port (default: 8000)
  --tp <n>                   Tensor parallel size (default: 4)
  --pp <n>                   Pipeline parallel size (default: 4)
  --ctx <n>                  Max model len (default: 8192)
  --gpu-mem <f>              GPU memory util (default: 0.92)
  --run-euroeval             Run EuroEval after server startup (default)
  --no-euroeval              Skip EuroEval and only launch/verify server
  --euroeval-model <id>      EuroEval model id (default: <served-model-name>)
  --languages <csv>          EuroEval languages (default: da)
  --datasets <csv>           EuroEval datasets (default: ifeval-da,danish-citizen-tests,danske-talemaader,multi-wiki-qa-da,dala)
  --all-datasets             Clear dataset filter (benchmark all datasets matching language/task filters)
  --tasks <csv>              EuroEval tasks (default: all)
  --iterations <n>           EuroEval num iterations (default: 1)
  --custom-datasets-file <p> Custom EuroEval dataset config file (default: ./lumi/euroeval_custom_datasets.py)
  --concurrency-mode <mode>  EuroEval concurrency mode: fixed|adaptive|auto (default: adaptive)
  --max-concurrent-calls <n> EuroEval max concurrent calls (default: 1000)
  --generative-type <type>   EuroEval generative type: auto|base|instruction_tuned|reasoning (default: auto)
  --extra-args <string>      Extra args appended to EuroEval CLI
  --eee-output-dir <path>    EEE root data dir override (default: ./logs/every_eval_ever/data)
  --nodes <n>                Slurm node count override (default: from sbatch file)
  --time <HH:MM:SS>          Slurm time limit override (default: from sbatch file)
  --slurm-log-dir <path>     Slurm stdout/err directory (default: ./logs/slurm)
  --script <path>            sbatch script path override
  --dry-run                  Print sbatch command/env and exit
  --help                     Show help

Examples:
  ./lumi/euroeval_submit.sh
  ./lumi/euroeval_submit.sh --languages da --iterations 1
  ./lumi/euroeval_submit.sh --datasets ifeval-da,danish-citizen-tests,danske-talemaader,multi-wiki-qa-da,dala
  ./lumi/euroeval_submit.sh --custom-datasets-file ./lumi/euroeval_custom_datasets.py
  ./lumi/euroeval_submit.sh --concurrency-mode adaptive --max-concurrent-calls 1000
  ./lumi/euroeval_submit.sh --tasks "knowledge,summarization"
  ./lumi/euroeval_submit.sh --all-datasets --tasks "knowledge,summarization"
  ./lumi/euroeval_submit.sh --eee-output-dir /path/to/every_eval_ever/data
  ./lumi/euroeval_submit.sh --nodes 1 --tp 1 --pp 1
  ./lumi/euroeval_submit.sh --time 12:00:00
  ./lumi/euroeval_submit.sh --slurm-log-dir /path/to/slurm-logs
  ./lumi/euroeval_submit.sh --no-euroeval
EOF
}

die() {
  echo "FATAL: $*" >&2
  exit 1
}

need_value() {
  local opt="$1"
  local remaining="$2"
  if [[ "$remaining" -lt 2 ]]; then
    die "missing value for $opt"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      need_value "$1" "$#"
      MODEL="$2"
      shift 2
      ;;
    --served-model-name)
      need_value "$1" "$#"
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --port)
      need_value "$1" "$#"
      PORT="$2"
      shift 2
      ;;
    --tp)
      need_value "$1" "$#"
      TP="$2"
      shift 2
      ;;
    --pp)
      need_value "$1" "$#"
      PP="$2"
      shift 2
      ;;
    --ctx)
      need_value "$1" "$#"
      CTX="$2"
      shift 2
      ;;
    --gpu-mem)
      need_value "$1" "$#"
      GPU_MEM="$2"
      shift 2
      ;;
    --run-euroeval)
      RUN_EUROEVAL=1
      shift
      ;;
    --no-euroeval)
      RUN_EUROEVAL=0
      shift
      ;;
    --euroeval-model)
      need_value "$1" "$#"
      EUROEVAL_MODEL="$2"
      shift 2
      ;;
    --languages)
      need_value "$1" "$#"
      EUROEVAL_LANGUAGES="$2"
      shift 2
      ;;
    --datasets)
      need_value "$1" "$#"
      EUROEVAL_DATASETS="$2"
      EUROEVAL_TASKS=""
      shift 2
      ;;
    --all-datasets)
      EUROEVAL_DATASETS=""
      shift
      ;;
    --tasks)
      need_value "$1" "$#"
      EUROEVAL_TASKS="$2"
      EUROEVAL_DATASETS=""
      shift 2
      ;;
    --iterations)
      need_value "$1" "$#"
      EUROEVAL_NUM_ITERATIONS="$2"
      shift 2
      ;;
    --custom-datasets-file)
      need_value "$1" "$#"
      EUROEVAL_CUSTOM_DATASETS_FILE="$2"
      shift 2
      ;;
    --concurrency-mode)
      need_value "$1" "$#"
      EUROEVAL_CONCURRENCY_MODE="$2"
      case "$EUROEVAL_CONCURRENCY_MODE" in
        fixed|adaptive|auto)
          ;;
        *)
          die "invalid --concurrency-mode: $EUROEVAL_CONCURRENCY_MODE (expected fixed|adaptive|auto)"
          ;;
      esac
      shift 2
      ;;
    --max-concurrent-calls)
      need_value "$1" "$#"
      EUROEVAL_MAX_CONCURRENT_CALLS="$2"
      shift 2
      ;;
    --generative-type)
      need_value "$1" "$#"
      EUROEVAL_GENERATIVE_TYPE="$2"
      case "$EUROEVAL_GENERATIVE_TYPE" in
        auto|base|instruction_tuned|reasoning)
          ;;
        instruction-tuned)
          EUROEVAL_GENERATIVE_TYPE="instruction_tuned"
          ;;
        *)
          die "invalid --generative-type: $EUROEVAL_GENERATIVE_TYPE (expected auto|base|instruction_tuned|reasoning)"
          ;;
      esac
      shift 2
      ;;
    --extra-args)
      need_value "$1" "$#"
      EUROEVAL_EXTRA_ARGS="$2"
      shift 2
      ;;
    --eee-output-dir)
      need_value "$1" "$#"
      EUROEVAL_EEE_OUTPUT_DIR="$2"
      shift 2
      ;;
    --nodes)
      need_value "$1" "$#"
      NODES="$2"
      shift 2
      ;;
    --time)
      need_value "$1" "$#"
      TIME_LIMIT="$2"
      shift 2
      ;;
    --slurm-log-dir)
      need_value "$1" "$#"
      SLURM_LOG_DIR="$2"
      shift 2
      ;;
    --script)
      need_value "$1" "$#"
      SUBMIT_SCRIPT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

if [[ -n "$EUROEVAL_TASKS" && -n "$EUROEVAL_DATASETS" ]]; then
  die "EUROEVAL_TASKS and EUROEVAL_DATASETS are mutually exclusive; use only one."
fi

[[ -f "$SUBMIT_SCRIPT" ]] || die "submit script not found: $SUBMIT_SCRIPT"
[[ -d "$OVERLAY_DIR" ]] || die "overlay dir not found: $OVERLAY_DIR"
mkdir -p "$SLURM_LOG_DIR"

raw_slurm_log_label="euroeval__${SERVED_MODEL_NAME}"
slurm_log_label="${raw_slurm_log_label//[^[:alnum:]._-]/_}"
slurm_out_path="${SLURM_LOG_DIR}/${slurm_log_label}-%j.out"
slurm_err_path="${SLURM_LOG_DIR}/${slurm_log_label}-%j.err"

env_kv=(
  "DFM_EVALS_REPO_ROOT=$REPO_ROOT"
  "OVERLAY_DIR=$OVERLAY_DIR"
  "MODEL=$MODEL"
  "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
  "PORT=$PORT"
  "TP=$TP"
  "PP=$PP"
  "CTX=$CTX"
  "GPU_MEM=$GPU_MEM"
  "RUN_EUROEVAL=$RUN_EUROEVAL"
  "EUROEVAL_MODEL=$EUROEVAL_MODEL"
  "EUROEVAL_LANGUAGES=$EUROEVAL_LANGUAGES"
  "EUROEVAL_NUM_ITERATIONS=$EUROEVAL_NUM_ITERATIONS"
  "EUROEVAL_CONCURRENCY_MODE=$EUROEVAL_CONCURRENCY_MODE"
  "EUROEVAL_MAX_CONCURRENT_CALLS=$EUROEVAL_MAX_CONCURRENT_CALLS"
  "EUROEVAL_GENERATIVE_TYPE=$EUROEVAL_GENERATIVE_TYPE"
  "EUROEVAL_CUSTOM_DATASETS_FILE=$EUROEVAL_CUSTOM_DATASETS_FILE"
  "EUROEVAL_SLURM_LOG_DIR=$SLURM_LOG_DIR"
)
if [[ -n "$EUROEVAL_TASKS" ]]; then
  env_kv+=("EUROEVAL_TASKS=$EUROEVAL_TASKS")
fi
if [[ -n "$EUROEVAL_DATASETS" ]]; then
  env_kv+=("EUROEVAL_DATASETS=$EUROEVAL_DATASETS")
fi
if [[ -n "$EUROEVAL_EXTRA_ARGS" ]]; then
  env_kv+=("EUROEVAL_EXTRA_ARGS=$EUROEVAL_EXTRA_ARGS")
fi
if [[ -n "$EUROEVAL_EEE_OUTPUT_DIR" ]]; then
  env_kv+=("EUROEVAL_EEE_OUTPUT_DIR=$EUROEVAL_EEE_OUTPUT_DIR")
fi

echo "Submit script: $SUBMIT_SCRIPT"
echo "Overlay: $OVERLAY_DIR"
echo "Model: $MODEL"
echo "Served model name: $SERVED_MODEL_NAME"
echo "TP/PP: $TP/$PP"
echo "CTX: $CTX"
echo "GPU_MEM: $GPU_MEM"
echo "Port: $PORT"
echo "Run EuroEval: $RUN_EUROEVAL"
echo "EuroEval model: $EUROEVAL_MODEL"
echo "EuroEval languages: $EUROEVAL_LANGUAGES"
echo "EuroEval datasets: ${EUROEVAL_DATASETS:-<all>}"
echo "EuroEval tasks: ${EUROEVAL_TASKS:-<all>}"
echo "EuroEval iterations: $EUROEVAL_NUM_ITERATIONS"
echo "EuroEval custom datasets file: $EUROEVAL_CUSTOM_DATASETS_FILE"
echo "EuroEval concurrency mode: $EUROEVAL_CONCURRENCY_MODE"
echo "EuroEval max concurrent calls: $EUROEVAL_MAX_CONCURRENT_CALLS"
echo "EuroEval generative type: $EUROEVAL_GENERATIVE_TYPE"
if [[ -n "$TIME_LIMIT" ]]; then
  echo "Slurm time limit override: $TIME_LIMIT"
fi
echo "Slurm stdout path pattern: $slurm_out_path"
echo "Slurm stderr path pattern: $slurm_err_path"
if [[ -n "$EUROEVAL_EXTRA_ARGS" ]]; then
  echo "EuroEval extra args: $EUROEVAL_EXTRA_ARGS"
fi
if [[ -n "$EUROEVAL_EEE_OUTPUT_DIR" ]]; then
  echo "EEE output dir override: $EUROEVAL_EEE_OUTPUT_DIR"
fi
if [[ -n "$NODES" ]]; then
  echo "Slurm node override: $NODES"
fi

sbatch_args=(--output "$slurm_out_path" --error "$slurm_err_path")
if [[ -n "$NODES" ]]; then
  sbatch_args+=(--nodes "$NODES")
fi
if [[ -n "$TIME_LIMIT" ]]; then
  sbatch_args+=(--time "$TIME_LIMIT")
fi
cmd=(env "${env_kv[@]}" sbatch "${sbatch_args[@]}" "$SUBMIT_SCRIPT")
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'Dry run command: '
  printf '(cd %q && ' "$REPO_ROOT"
  printf '%q ' "${cmd[@]}"
  printf ')'
  echo
  exit 0
fi

submit_out="$(cd "$REPO_ROOT" && "${cmd[@]}")"
echo "$submit_out"

job_id="$(awk '/Submitted batch job/{print $4}' <<<"$submit_out")"
if [[ -n "$job_id" ]]; then
  echo "Job id: $job_id"
  echo "Stdout: ${slurm_out_path//%j/$job_id}"
  echo "Stderr: ${slurm_err_path//%j/$job_id}"
  echo "Rank logs: $SLURM_LOG_DIR/vllm-q35-mp1-rank-*-${job_id}.log"
fi
