#!/bin/bash
# Multi-node vLLM MP launch using a single srun step (one task per node),
# followed by EuroEval against the in-job OpenAI-compatible vLLM endpoint.

#SBATCH --job-name=vllm-q35-2n-euroeval
#SBATCH --account=project_465002183
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output=vllm-q35-mp1-%j.out
#SBATCH --error=vllm-q35-mp1-%j.err

set -euo pipefail

REPO_ROOT="${DFM_EVALS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_ROOT/lumi"

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
SIF=${SIF:-$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif}
OVERLAY_DIR=${OVERLAY_DIR:-$REPO_ROOT/overlay_vllm_minimal}
if [[ ! -d "$OVERLAY_DIR" && -d "$REPO_ROOT/../overlay_vllm_minimal" ]]; then
  OVERLAY_DIR="$REPO_ROOT/../overlay_vllm_minimal"
fi

MODEL=${MODEL:-Qwen/Qwen3.5-397B-A17B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
HF_HOME=${HF_HOME:-/flash/project_465002183/.cache/huggingface/}

PORT=${PORT:-8000}
TP=${TP:-4}
PP=${PP:-4}
CTX=${CTX:-8192}
GPU_MEM=${GPU_MEM:-0.92}
VLLM_SERVER_API_KEY=${VLLM_SERVER_API_KEY:-${VLLM_API_KEY:-}}

RUN_EUROEVAL=${RUN_EUROEVAL:-1}
EUROEVAL_MODEL=${EUROEVAL_MODEL:-$SERVED_MODEL_NAME}
EUROEVAL_API_BASE=${EUROEVAL_API_BASE:-http://127.0.0.1:$PORT/v1}
EUROEVAL_LANGUAGES=${EUROEVAL_LANGUAGES:-da}
DEFAULT_EUROEVAL_DATASETS="ifeval-da,danish-citizen-tests,danske-talemaader,multi-wiki-qa-da,dala"
EUROEVAL_DATASETS=${EUROEVAL_DATASETS:-$DEFAULT_EUROEVAL_DATASETS}
EUROEVAL_TASKS=${EUROEVAL_TASKS:-}
EUROEVAL_NUM_ITERATIONS=${EUROEVAL_NUM_ITERATIONS:-1}
EUROEVAL_CONCURRENCY_MODE=${EUROEVAL_CONCURRENCY_MODE:-adaptive}
EUROEVAL_MAX_CONCURRENT_CALLS=${EUROEVAL_MAX_CONCURRENT_CALLS:-1000}
EUROEVAL_GENERATIVE_TYPE=${EUROEVAL_GENERATIVE_TYPE:-auto}
EUROEVAL_CUSTOM_DATASETS_FILE=${EUROEVAL_CUSTOM_DATASETS_FILE:-$REPO_ROOT/lumi/euroeval_custom_datasets.py}
EUROEVAL_CACHE_DIR=${EUROEVAL_CACHE_DIR:-/overlay/euroeval-cache-${SLURM_JOB_ID:-manual}}
EUROEVAL_RUN_DIR=${EUROEVAL_RUN_DIR:-/overlay/euroeval-runs/${SLURM_JOB_ID:-manual}}
EUROEVAL_EXPORT_EEE=${EUROEVAL_EXPORT_EEE:-1}
EUROEVAL_EEE_OUTPUT_DIR=${EUROEVAL_EEE_OUTPUT_DIR:-}
EUROEVAL_EEE_SOURCE_ORGANIZATION_NAME=${EUROEVAL_EEE_SOURCE_ORGANIZATION_NAME:-dfm-evals}
EUROEVAL_EEE_EVALUATOR_RELATIONSHIP=${EUROEVAL_EEE_EVALUATOR_RELATIONSHIP:-third_party}
EUROEVAL_RAISE_ERRORS=${EUROEVAL_RAISE_ERRORS:-1}
EUROEVAL_EXTRA_ARGS=${EUROEVAL_EXTRA_ARGS:-}
SLURM_JOB_ID_SAFE=${SLURM_JOB_ID:-manual}
EUROEVAL_SLURM_LOG_DIR=${EUROEVAL_SLURM_LOG_DIR:-$REPO_ROOT/logs/slurm}
RANK_LOG_PATH="${EUROEVAL_SLURM_LOG_DIR}/vllm-q35-mp1-rank-%t-${SLURM_JOB_ID_SAFE}.log"
COMPLETION_FILE="${EUROEVAL_SLURM_LOG_DIR}/completion-qwen35-mp1-${SLURM_JOB_ID_SAFE}.json"
if [[ -z "$EUROEVAL_EEE_OUTPUT_DIR" ]]; then
  EUROEVAL_EEE_OUTPUT_DIR="${REPO_ROOT}/logs/every_eval_ever/data"
fi

MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-29501}
LAUNCH_TIMEOUT=${LAUNCH_TIMEOUT:-7200}
LAUNCH_STEP=${LAUNCH_STEP:-10}

FI_PROVIDER_NAME=${FI_PROVIDER_NAME:-cxi}
FI_CXI_DISABLE_HMEM_FLAG=${FI_CXI_DISABLE_HMEM_FLAG:-1}
GLOO_IFNAME=${GLOO_IFNAME:-hsn0}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}
NCCL_DEBUG_LEVEL=${NCCL_DEBUG_LEVEL:-WARN}
NCCL_DEBUG_SUBSYS_LEVEL=${NCCL_DEBUG_SUBSYS_LEVEL:-INIT,NET}
NCCL_NET_BACKEND=${NCCL_NET_BACKEND:-}
NCCL_NET_PLUGIN_NAME=${NCCL_NET_PLUGIN_NAME:-}
AWS_OFI_RCCL_LIBDIR=${AWS_OFI_RCCL_LIBDIR:-}
RCCL_NET_SO_SRC=${RCCL_NET_SO_SRC:-}
LIBFABRIC_EXTRACT_DIR=${LIBFABRIC_EXTRACT_DIR:-$SCRIPT_DIR/libfabric_extract_20260216}
LIBFABRIC_SO=${LIBFABRIC_SO:-$LIBFABRIC_EXTRACT_DIR/libfabric.so.1.27.0}
LIBCXI_SO=${LIBCXI_SO:-$LIBFABRIC_EXTRACT_DIR/libcxi.so.1.5.0}

if [[ ! -d "$LIBFABRIC_EXTRACT_DIR" && -d "$REPO_ROOT/../libfabric_extract_20260216" ]]; then
  LIBFABRIC_EXTRACT_DIR="$REPO_ROOT/../libfabric_extract_20260216"
  LIBFABRIC_SO="${LIBFABRIC_EXTRACT_DIR}/libfabric.so.1.27.0"
  LIBCXI_SO="${LIBFABRIC_EXTRACT_DIR}/libcxi.so.1.5.0"
fi

TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-}
ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE:-0}
TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
REASONING_PARSER=${REASONING_PARSER:-auto}
STRUCTURED_OUTPUTS_CONFIG=${STRUCTURED_OUTPUTS_CONFIG:-}
LORA_MAX_RANK_OVERRIDE=${LORA_MAX_RANK_OVERRIDE:-}

SRUN_PID=""
HEAD_NODE=""

die() {
  echo "FATAL: $*" >&2
  exit 1
}

model_label_from_ref() {
  local model_ref="$1"
  local ref="$model_ref"
  local base_name
  local parent_name
  local label

  ref="${ref#vllm/}"
  ref="${ref#openai/}"

  if [[ "$ref" == */* ]]; then
    base_name="$(basename "$ref")"
    parent_name="$(basename "$(dirname "$ref")")"
    case "$base_name" in
      final|latest|last|checkpoint-*|step-*|epoch-*)
        if [[ -n "$parent_name" && "$parent_name" != "." && "$parent_name" != "/" ]]; then
          label="${parent_name}-${base_name}"
        else
          label="$base_name"
        fi
        ;;
      *)
        label="$base_name"
        ;;
    esac
  else
    label="$ref"
  fi

  label="${label//[^[:alnum:]._-]/_}"
  [[ -n "$label" ]] || label="model"
  echo "$label"
}

model_developer_from_ref() {
  local model_ref="$1"
  local ref="$model_ref"
  ref="${ref#vllm/}"
  ref="${ref#openai/}"
  if [[ "$ref" == */* ]]; then
    echo "${ref%%/*}"
  else
    echo ""
  fi
}

case "$EUROEVAL_GENERATIVE_TYPE" in
  auto|base|instruction_tuned|reasoning)
    ;;
  instruction-tuned)
    EUROEVAL_GENERATIVE_TYPE="instruction_tuned"
    ;;
  *)
    die "invalid EUROEVAL_GENERATIVE_TYPE=$EUROEVAL_GENERATIVE_TYPE (expected auto|base|instruction_tuned|reasoning)"
    ;;
esac

case "$EUROEVAL_CONCURRENCY_MODE" in
  fixed|adaptive|auto)
    ;;
  *)
    die "invalid EUROEVAL_CONCURRENCY_MODE=$EUROEVAL_CONCURRENCY_MODE (expected fixed|adaptive|auto)"
    ;;
esac

if [[ -n "$EUROEVAL_TASKS" && -n "$EUROEVAL_DATASETS" ]]; then
  die "EUROEVAL_TASKS and EUROEVAL_DATASETS are mutually exclusive; set at most one."
fi

ADAPTER_CONFIG_PATH="$MODEL/adapter_config.json"
ADAPTER_WEIGHTS_PATH="$MODEL/adapter_model.safetensors"
ADAPTER_CHAT_TEMPLATE_PATH="$MODEL/chat_template.jinja"
USE_LORA_ADAPTER=0
LORA_BASE_MODEL=""
LORA_MODULE_NAME=""
LORA_MODULE_SPEC=""
LORA_ADAPTER_PATH="$MODEL"
LORA_CHAT_TEMPLATE_PATH=""
LORA_TOKENIZER_PATH=""
LORA_MAX_RANK="${LORA_MAX_RANK_OVERRIDE:-}"
VLLM_MODEL_TAG="$MODEL"
VLLM_SERVED_MODEL_NAME="$SERVED_MODEL_NAME"
STARTUP_CHECK_MODEL="$EUROEVAL_MODEL"

if [[ -d "$MODEL" && -f "$ADAPTER_CONFIG_PATH" && -f "$ADAPTER_WEIGHTS_PATH" ]]; then
  adapter_meta="$(
    ADAPTER_CONFIG_PATH="$ADAPTER_CONFIG_PATH" python3 - <<'PY'
import json
import os

p = os.environ["ADAPTER_CONFIG_PATH"]
with open(p, "r", encoding="utf-8") as f:
    cfg = json.load(f)

base = str(cfg.get("base_model_name_or_path") or "").strip()
if not base:
    raise SystemExit("adapter_config.json missing base_model_name_or_path")

rank = cfg.get("r", 16)
try:
    rank = int(rank)
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"invalid adapter rank r={rank!r}") from exc

if rank <= 0:
    raise SystemExit(f"invalid adapter rank r={rank}")

print(f"{base}\t{rank}")
PY
  )" || die "failed to parse adapter metadata from $ADAPTER_CONFIG_PATH"

  IFS=$'\t' read -r LORA_BASE_MODEL LORA_MAX_RANK <<<"$adapter_meta"
  [[ -n "$LORA_BASE_MODEL" ]] || die "empty base model in $ADAPTER_CONFIG_PATH"
  [[ -n "$LORA_MAX_RANK" ]] || die "empty adapter rank in $ADAPTER_CONFIG_PATH"

  USE_LORA_ADAPTER=1
  VLLM_MODEL_TAG="$LORA_BASE_MODEL"
  VLLM_SERVED_MODEL_NAME="$LORA_BASE_MODEL"
  LORA_BASE_DEVELOPER="$(model_developer_from_ref "$LORA_BASE_MODEL")"

  if [[ "$SERVED_MODEL_NAME" == "$MODEL" || -z "$SERVED_MODEL_NAME" ]]; then
    LORA_MODULE_NAME="$(model_label_from_ref "$MODEL")"
    if [[ -n "$LORA_BASE_DEVELOPER" ]]; then
      LORA_MODULE_NAME="${LORA_BASE_DEVELOPER}/${LORA_MODULE_NAME}"
    fi
  else
    LORA_MODULE_NAME="$SERVED_MODEL_NAME"
  fi
  [[ -n "$LORA_MODULE_NAME" ]] || LORA_MODULE_NAME="adapter"

  # Avoid serving base and LoRA with the same model id.
  if [[ "$LORA_MODULE_NAME" == "$VLLM_SERVED_MODEL_NAME" ]]; then
    LORA_MODULE_NAME="${LORA_MODULE_NAME}--lora"
  fi

  if resolved_adapter_path="$(readlink -f "$MODEL" 2>/dev/null)"; then
    LORA_ADAPTER_PATH="$resolved_adapter_path"
  fi
  LORA_MODULE_SPEC="${LORA_MODULE_NAME}=${LORA_ADAPTER_PATH}"
  if [[ -f "$ADAPTER_CHAT_TEMPLATE_PATH" ]]; then
    if resolved_chat_template_path="$(readlink -f "$ADAPTER_CHAT_TEMPLATE_PATH" 2>/dev/null)"; then
      LORA_CHAT_TEMPLATE_PATH="$resolved_chat_template_path"
    else
      LORA_CHAT_TEMPLATE_PATH="$ADAPTER_CHAT_TEMPLATE_PATH"
    fi
  fi

  # If adapter export contains tokenizer artifacts, force vLLM to use them.
  if [[ -f "$MODEL/tokenizer_config.json" || -f "$MODEL/tokenizer.json" || -f "$MODEL/tokenizer.model" || -f "$MODEL/vocab.json" ]]; then
    if resolved_tokenizer_path="$(readlink -f "$MODEL" 2>/dev/null)"; then
      LORA_TOKENIZER_PATH="$resolved_tokenizer_path"
    else
      LORA_TOKENIZER_PATH="$MODEL"
    fi
  fi

  # If EUROEVAL_MODEL still points to the adapter path/default served id, map it to the LoRA id.
  if [[ "$EUROEVAL_MODEL" == "$MODEL" || "$EUROEVAL_MODEL" == "$SERVED_MODEL_NAME" || "$EUROEVAL_MODEL" == "$VLLM_SERVED_MODEL_NAME" ]]; then
    EUROEVAL_MODEL="$LORA_MODULE_NAME"
  fi
  STARTUP_CHECK_MODEL="$EUROEVAL_MODEL"
fi

resolve_euroeval_generative_type() {
  local resolved="${EUROEVAL_GENERATIVE_TYPE}"

  if [[ "$resolved" == "auto" ]]; then
    case "$EUROEVAL_MODEL" in
      Qwen/Qwen3*|Qwen/Qwen3.5*|openai/Qwen/Qwen3*|openai/Qwen/Qwen3.5*|openai/Qwen3*|openai/Qwen3.5*)
        resolved="reasoning"
        ;;
    esac
  fi

  echo "$resolved"
}

build_euroeval_cli() {
  local -n out_cmd_ref=$1
  local -a euroeval_args
  local resolved_generative_type

  euroeval_args=(
    --model "$EUROEVAL_MODEL"
    --api-base "$EUROEVAL_API_BASE"
    --cache-dir "$EUROEVAL_CACHE_DIR"
    --num-iterations "$EUROEVAL_NUM_ITERATIONS"
    --concurrency-mode "$EUROEVAL_CONCURRENCY_MODE"
  )
  if [[ -n "$EUROEVAL_MAX_CONCURRENT_CALLS" ]]; then
    euroeval_args+=(--max-concurrent-calls "$EUROEVAL_MAX_CONCURRENT_CALLS")
  fi

  resolved_generative_type="$(resolve_euroeval_generative_type)"
  if [[ "$resolved_generative_type" != "auto" && -n "$resolved_generative_type" ]]; then
    euroeval_args+=(--generative-type "$resolved_generative_type")
  fi

  if [[ "$EUROEVAL_RAISE_ERRORS" == "1" ]]; then
    euroeval_args+=(--raise-errors)
  else
    euroeval_args+=(--no-raise-errors)
  fi

  local raw_languages="${EUROEVAL_LANGUAGES//,/ }"
  local lang
  local languages_added=0
  for lang in $raw_languages; do
    [[ -n "$lang" ]] || continue
    euroeval_args+=(--language "$lang")
    languages_added=1
  done
  if [[ "$languages_added" -eq 0 ]]; then
    euroeval_args+=(--language all)
  fi

  local raw_tasks="${EUROEVAL_TASKS//,/ }"
  local task
  for task in $raw_tasks; do
    [[ -n "$task" ]] || continue
    euroeval_args+=(--task "$task")
  done

  local raw_datasets="${EUROEVAL_DATASETS//,/ }"
  local dataset
  for dataset in $raw_datasets; do
    [[ -n "$dataset" ]] || continue
    euroeval_args+=(--dataset "$dataset")
  done

  if [[ -n "$EUROEVAL_CUSTOM_DATASETS_FILE" ]]; then
    euroeval_args+=(--custom-datasets-file "$EUROEVAL_CUSTOM_DATASETS_FILE")
  fi

  local escaped_args=""
  printf -v escaped_args '%q ' "${euroeval_args[@]}"
  out_cmd_ref="euroeval ${escaped_args}"
  if [[ -n "$EUROEVAL_EXTRA_ARGS" ]]; then
    out_cmd_ref+="$EUROEVAL_EXTRA_ARGS"
  fi
}

container_env_prefix() {
  cat <<'EOC'
set -euo pipefail
source /overlay/venv/vllm-min/bin/activate
OVERLAY_SITE=/overlay/venv/vllm-min/lib/python3.12/site-packages
if [[ -d /overlay/src/transformers/src ]]; then
  export PYTHONPATH=/overlay/src/transformers/src:/overlay/src/vllm:${OVERLAY_SITE}${PYTHONPATH:+:$PYTHONPATH}
else
  export PYTHONPATH=/overlay/src/vllm:${OVERLAY_SITE}${PYTHONPATH:+:$PYTHONPATH}
fi
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi
unset ROCR_VISIBLE_DEVICES
export HF_HOME="__HF_HOME__"
export HF_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
EOC
}

render_env() {
  container_env_prefix | sed "s|__HF_HOME__|$HF_HOME|g"
}

cleanup() {
  set +e
  stop_srun
}
trap cleanup EXIT

stop_srun() {
  if [[ -z "${SRUN_PID}" ]]; then
    return
  fi
  if ! kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
    wait "${SRUN_PID}" >/dev/null 2>&1 || true
    SRUN_PID=""
    return
  fi

  kill "${SRUN_PID}" >/dev/null 2>&1 || true
  for _ in {1..20}; do
    if ! kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
    scancel --signal=TERM "${SLURM_JOB_ID}.0" >/dev/null 2>&1 || true
    sleep 2
  fi
  if kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
    kill -9 "${SRUN_PID}" >/dev/null 2>&1 || true
  fi
  wait "${SRUN_PID}" >/dev/null 2>&1 || true
  SRUN_PID=""
}

[[ -f "$SIF" ]] || die "SIF not found: $SIF"
[[ -d "$OVERLAY_DIR/venv/vllm-min" ]] || die "overlay venv missing: $OVERLAY_DIR/venv/vllm-min"
[[ "${SLURM_NNODES:-0}" -ge 1 ]] || die "this script requires at least 1 node"
mkdir -p "$EUROEVAL_SLURM_LOG_DIR"

readarray -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#NODES[@]}" -ge 1 ]] || die "expected at least 1 node, got ${#NODES[@]}"
HEAD_NODE="${NODES[0]}"

if [[ -z "$MASTER_ADDR" ]]; then
  MASTER_ADDR="$HEAD_NODE"
fi

echo "SIF: $SIF"
echo "Overlay: $OVERLAY_DIR"
echo "Model: $MODEL"
echo "vLLM model tag: $VLLM_MODEL_TAG"
echo "vLLM served model name: $VLLM_SERVED_MODEL_NAME"
echo "HF_HOME: $HF_HOME"
echo "Nodes: ${NODES[*]}"
echo "Head node: $HEAD_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "NCCL_NET_GDR_LEVEL: $NCCL_NET_GDR_LEVEL"
echo "NCCL_NET_PLUGIN_NAME: ${NCCL_NET_PLUGIN_NAME:-<unset>}"
echo "AWS_OFI_RCCL_LIBDIR: $AWS_OFI_RCCL_LIBDIR"
echo "RCCL_NET_SO_SRC: $RCCL_NET_SO_SRC"
echo "LIBFABRIC_SO: $LIBFABRIC_SO"
echo "LIBCXI_SO: $LIBCXI_SO"
echo "TORCHDYNAMO_DISABLE: $TORCHDYNAMO_DISABLE"
echo "REASONING_PARSER: $REASONING_PARSER"
echo "STRUCTURED_OUTPUTS_CONFIG: ${STRUCTURED_OUTPUTS_CONFIG:-<default-auto>}"
echo "RUN_EUROEVAL: $RUN_EUROEVAL"
echo "EUROEVAL_MODEL: $EUROEVAL_MODEL"
echo "EUROEVAL_API_BASE: $EUROEVAL_API_BASE"
echo "EUROEVAL_LANGUAGES: $EUROEVAL_LANGUAGES"
echo "EUROEVAL_DATASETS: ${EUROEVAL_DATASETS:-<all-datasets>}"
echo "EUROEVAL_TASKS: ${EUROEVAL_TASKS:-<all-tasks>}"
echo "EUROEVAL_NUM_ITERATIONS: $EUROEVAL_NUM_ITERATIONS"
echo "EUROEVAL_CUSTOM_DATASETS_FILE: $EUROEVAL_CUSTOM_DATASETS_FILE"
echo "EUROEVAL_CONCURRENCY_MODE: $EUROEVAL_CONCURRENCY_MODE"
echo "EUROEVAL_MAX_CONCURRENT_CALLS: $EUROEVAL_MAX_CONCURRENT_CALLS"
echo "EUROEVAL_GENERATIVE_TYPE: $EUROEVAL_GENERATIVE_TYPE"
echo "EUROEVAL_GENERATIVE_TYPE_RESOLVED: $(resolve_euroeval_generative_type)"
echo "STARTUP_CHECK_MODEL: $STARTUP_CHECK_MODEL"
if [[ -n "$VLLM_SERVER_API_KEY" ]]; then
  echo "VLLM_SERVER_API_KEY: <set>"
else
  echo "VLLM_SERVER_API_KEY: <unset>"
fi
echo "USE_LORA_ADAPTER: $USE_LORA_ADAPTER"
if [[ "$USE_LORA_ADAPTER" == "1" ]]; then
  echo "LoRA adapter path: $LORA_ADAPTER_PATH"
  echo "LoRA base model: $LORA_BASE_MODEL"
  echo "LoRA module id: $LORA_MODULE_NAME"
  echo "LoRA max rank: $LORA_MAX_RANK"
  echo "LoRA chat template: ${LORA_CHAT_TEMPLATE_PATH:-<none>}"
  echo "LoRA tokenizer path: ${LORA_TOKENIZER_PATH:-<none>}"
fi
echo "EUROEVAL_CACHE_DIR: $EUROEVAL_CACHE_DIR"
echo "EUROEVAL_RUN_DIR: $EUROEVAL_RUN_DIR"
echo "EUROEVAL_EXPORT_EEE: $EUROEVAL_EXPORT_EEE"
if [[ "$EUROEVAL_EXPORT_EEE" == "1" ]]; then
  echo "EUROEVAL_EEE_OUTPUT_DIR: $EUROEVAL_EEE_OUTPUT_DIR"
  echo "EUROEVAL_EEE_SOURCE_ORGANIZATION_NAME: $EUROEVAL_EEE_SOURCE_ORGANIZATION_NAME"
  echo "EUROEVAL_EEE_EVALUATOR_RELATIONSHIP: $EUROEVAL_EEE_EVALUATOR_RELATIONSHIP"
fi
echo "Slurm log dir: $EUROEVAL_SLURM_LOG_DIR"
echo "Rank log path pattern: $RANK_LOG_PATH"

SING_BIND_ARGS=(
  -B "$BASE_DIR:$LAIFS_APPL_DIR"
  -B "$OVERLAY_DIR:/overlay"
  -B "$REPO_ROOT:$REPO_ROOT"
)
if [[ -d /flash ]]; then
  SING_BIND_ARGS+=(-B /flash:/flash)
fi
if [[ -L /flash/project_465002183 ]]; then
  FLASH_TARGET="$(readlink -f /flash/project_465002183 || true)"
  if [[ -n "$FLASH_TARGET" && -d "$FLASH_TARGET" ]]; then
    SING_BIND_ARGS+=(-B "$FLASH_TARGET:$FLASH_TARGET")
  fi
fi
if [[ -f "$RCCL_NET_SO_SRC" ]]; then
  SING_BIND_ARGS+=(
    -B "$RCCL_NET_SO_SRC:/lib/librccl-net.so"
    -B "$RCCL_NET_SO_SRC:/usr/lib/librccl-net.so"
  )
fi
if [[ -f "$LIBFABRIC_SO" ]]; then
  SING_BIND_ARGS+=(
    -B "$LIBFABRIC_SO:/lib/x86_64-linux-gnu/libfabric.so.1"
    -B "$LIBFABRIC_SO:/usr/lib/libfabric.so.1"
  )
fi
if [[ -f "$LIBCXI_SO" ]]; then
  SING_BIND_ARGS+=(
    -B "$LIBCXI_SO:/lib/libcxi.so.1"
    -B "$LIBCXI_SO:/usr/lib/libcxi.so.1"
  )
fi
if [[ "$USE_LORA_ADAPTER" == "1" && -d "$LORA_ADAPTER_PATH" ]]; then
  SING_BIND_ARGS+=(-B "$LORA_ADAPTER_PATH:$LORA_ADAPTER_PATH")
fi

RANK_CMD="$(render_env)"
RANK_CMD="${RANK_CMD}
export NODE_RANK=\${SLURM_PROCID}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_PLUGINS=''
export VLLM_DISABLE_PYNCCL=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_PROVIDER='$FI_PROVIDER_NAME'
export FI_CXI_DISABLE_HMEM='$FI_CXI_DISABLE_HMEM_FLAG'
export GLOO_SOCKET_IFNAME='$GLOO_IFNAME'
export NCCL_SOCKET_IFNAME='$NCCL_SOCKET_IFNAME'
export NCCL_NET_GDR_LEVEL='$NCCL_NET_GDR_LEVEL'
export NCCL_DEBUG='$NCCL_DEBUG_LEVEL'
export NCCL_DEBUG_SUBSYS='$NCCL_DEBUG_SUBSYS_LEVEL'
export TORCHDYNAMO_DISABLE='$TORCHDYNAMO_DISABLE'
if [[ -d '$AWS_OFI_RCCL_LIBDIR' ]]; then
  export LD_LIBRARY_PATH='$AWS_OFI_RCCL_LIBDIR':\${LD_LIBRARY_PATH:-}
fi
if [[ -n '$NCCL_NET_PLUGIN_NAME' ]]; then
  export NCCL_NET_PLUGIN='$NCCL_NET_PLUGIN_NAME'
else
  unset NCCL_NET_PLUGIN
fi
if [[ -n '$NCCL_NET_BACKEND' ]]; then
  export NCCL_NET='$NCCL_NET_BACKEND'
else
  unset NCCL_NET
fi
EXTRA_VLLM_FLAGS=()
if [[ '$REASONING_PARSER' == 'auto' ]]; then
  case '$VLLM_MODEL_TAG' in
    Qwen/Qwen3*|Qwen/Qwen3.5*) EXTRA_VLLM_FLAGS+=(--reasoning-parser qwen3) ;;
  esac
elif [[ -n '$REASONING_PARSER' && '$REASONING_PARSER' != 'none' ]]; then
  EXTRA_VLLM_FLAGS+=(--reasoning-parser '$REASONING_PARSER')
fi
if [[ -n '$STRUCTURED_OUTPUTS_CONFIG' ]]; then
  EXTRA_VLLM_FLAGS+=(--structured-outputs-config '$STRUCTURED_OUTPUTS_CONFIG')
fi
if [[ '$ENABLE_AUTO_TOOL_CHOICE' == '1' ]]; then
  EXTRA_VLLM_FLAGS+=(--enable-auto-tool-choice)
fi
if [[ -n '$TOOL_CALL_PARSER' ]]; then
  EXTRA_VLLM_FLAGS+=(--tool-call-parser '$TOOL_CALL_PARSER')
fi
if [[ -n '$VLLM_SERVER_API_KEY' ]]; then
  EXTRA_VLLM_FLAGS+=(--api-key '$VLLM_SERVER_API_KEY')
fi
if [[ '$USE_LORA_ADAPTER' == '1' ]]; then
  EXTRA_VLLM_FLAGS+=(--enable-lora --lora-modules '$LORA_MODULE_SPEC' --max-lora-rank '$LORA_MAX_RANK')
  if [[ -n '$LORA_CHAT_TEMPLATE_PATH' ]]; then
    EXTRA_VLLM_FLAGS+=(--chat-template '$LORA_CHAT_TEMPLATE_PATH')
  fi
  if [[ -n '$LORA_TOKENIZER_PATH' ]]; then
    EXTRA_VLLM_FLAGS+=(--tokenizer '$LORA_TOKENIZER_PATH')
  fi
fi
if [[ \${NODE_RANK} -ne 0 ]]; then
  EXTRA_VLLM_FLAGS+=(--headless)
fi
vllm serve '$VLLM_MODEL_TAG' \
  --served-model-name '$VLLM_SERVED_MODEL_NAME' \
  --host 0.0.0.0 \
  --port '$PORT' \
  --dtype bfloat16 \
  --distributed-executor-backend mp \
  --data-parallel-backend mp \
  --tensor-parallel-size '$TP' \
  --pipeline-parallel-size '$PP' \
  --nnodes '${#NODES[@]}' \
  --node-rank \${NODE_RANK} \
  --master-addr '$MASTER_ADDR' \
  --master-port '$MASTER_PORT' \
  --gpu-memory-utilization '$GPU_MEM' \
  --max-model-len '$CTX' \
  --safetensors-load-strategy eager \
  --enforce-eager \
  \${EXTRA_VLLM_FLAGS[@]}"

echo "Starting single-step srun MP launch..."
srun --overlap \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-56}" \
  --output="$RANK_LOG_PATH" \
  --error="$RANK_LOG_PATH" \
  singularity exec --rocm "${SING_BIND_ARGS[@]}" "$SIF" \
  bash -lc "$RANK_CMD" &
SRUN_PID=$!

READY=0
CURL_AUTH_ARGS=()
if [[ -n "$VLLM_SERVER_API_KEY" ]]; then
  CURL_AUTH_ARGS=(-H "Authorization: Bearer $VLLM_SERVER_API_KEY")
fi
for ((i=0; i<LAUNCH_TIMEOUT; i+=LAUNCH_STEP)); do
  if ! kill -0 "$SRUN_PID" >/dev/null 2>&1; then
    die "srun exited early; check ${EUROEVAL_SLURM_LOG_DIR}/vllm-q35-mp1-rank-*-${SLURM_JOB_ID_SAFE}.log"
  fi
  if curl -sf "${CURL_AUTH_ARGS[@]}" "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep "$LAUNCH_STEP"
done
[[ "$READY" -eq 1 ]] || die "vLLM server did not become ready"
echo "Server ready."

curl -sS "${CURL_AUTH_ARGS[@]}" -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$STARTUP_CHECK_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Write one short sentence about LUMI.\"}], \"temperature\": 0.2, \"max_tokens\": 256, \"skip_special_tokens\": false}" \
  > "$COMPLETION_FILE"

COMPLETION_FILE="$COMPLETION_FILE" python3 - <<'PY'
import json
import os
p = os.environ["COMPLETION_FILE"]
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)
print("Startup completion check:")
if "choices" in data and data["choices"]:
    message = data["choices"][0].get("message", {})
    reasoning = (message.get("reasoning") or "").strip()
    content = (message.get("content") or "").strip()
    print(f"Reasoning detected: {'yes' if reasoning else 'no'}")
    if reasoning:
        print("Reasoning preview:")
        print(reasoning[:800])
    print("Content:")
    print(content)
else:
    print(json.dumps(data, indent=2))
    raise SystemExit(1)
PY

echo "Run complete."

if [[ "$RUN_EUROEVAL" == "1" ]]; then
  EUROEVAL_CMD=""
  build_euroeval_cli EUROEVAL_CMD

  printf -v EUROEVAL_CACHE_DIR_Q '%q' "$EUROEVAL_CACHE_DIR"
  printf -v EUROEVAL_RUN_DIR_Q '%q' "$EUROEVAL_RUN_DIR"
  printf -v VLLM_SERVER_API_KEY_Q '%q' "$VLLM_SERVER_API_KEY"
  EVAL_CMD="$(render_env)
export EUROEVAL_ALLOW_FLASH_ATTN=1
if [[ -n ${VLLM_SERVER_API_KEY_Q} ]]; then
  export OPENAI_API_KEY=${VLLM_SERVER_API_KEY_Q}
fi
mkdir -p ${EUROEVAL_RUN_DIR_Q}
mkdir -p ${EUROEVAL_CACHE_DIR_Q}
cd ${EUROEVAL_RUN_DIR_Q}
command -v euroeval >/dev/null 2>&1 || { echo 'FATAL: euroeval not installed in overlay venv'; exit 1; }
${EUROEVAL_CMD}"

  echo "Starting EuroEval against ${EUROEVAL_API_BASE} ..."
  singularity exec --rocm "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$EVAL_CMD"
  echo "EuroEval results file: ${EUROEVAL_RUN_DIR}/euroeval_benchmark_results.jsonl"
  echo "EuroEval complete."

  if [[ "$EUROEVAL_EXPORT_EEE" == "1" ]]; then
    EUROEVAL_RESULTS_FILE="${EUROEVAL_RUN_DIR}/euroeval_benchmark_results.jsonl"
    printf -v EUROEVAL_RESULTS_FILE_Q '%q' "$EUROEVAL_RESULTS_FILE"
    printf -v EUROEVAL_EEE_OUTPUT_DIR_Q '%q' "$EUROEVAL_EEE_OUTPUT_DIR"
    printf -v EUROEVAL_EEE_SOURCE_ORGANIZATION_NAME_Q '%q' "$EUROEVAL_EEE_SOURCE_ORGANIZATION_NAME"
    printf -v EUROEVAL_EEE_EVALUATOR_RELATIONSHIP_Q '%q' "$EUROEVAL_EEE_EVALUATOR_RELATIONSHIP"
    EUROEVAL_EEE_INFERENCE_ARGS=""
    if [[ -n "${EUROEVAL_API_BASE:-}" ]]; then
      printf -v EUROEVAL_EEE_INFERENCE_ARGS ' --inference-base-url %q' "$EUROEVAL_API_BASE"
    fi

    EXPORT_EEE_CMD="$(render_env)
command -v evals >/dev/null 2>&1 || { echo 'FATAL: evals not installed in overlay venv'; exit 1; }
if [[ ! -f ${EUROEVAL_RESULTS_FILE_Q} ]]; then
  echo 'FATAL: missing EuroEval results file.'
  exit 1
fi
mkdir -p ${EUROEVAL_EEE_OUTPUT_DIR_Q}
evals eee euroeval \
  --results-file ${EUROEVAL_RESULTS_FILE_Q} \
  --output-dir ${EUROEVAL_EEE_OUTPUT_DIR_Q} \
  --source-organization-name ${EUROEVAL_EEE_SOURCE_ORGANIZATION_NAME_Q} \
  --evaluator-relationship ${EUROEVAL_EEE_EVALUATOR_RELATIONSHIP_Q}${EUROEVAL_EEE_INFERENCE_ARGS}"

    echo "Exporting EuroEval results to every_eval_ever format..."
    singularity exec --rocm "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$EXPORT_EEE_CMD"
    echo "EEE export complete: $EUROEVAL_EEE_OUTPUT_DIR"
  else
    echo "EUROEVAL_EXPORT_EEE=0, skipping EEE export."
  fi
else
  echo "RUN_EUROEVAL=0, skipping EuroEval."
fi

stop_srun
