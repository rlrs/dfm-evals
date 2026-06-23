#!/bin/bash
# Run dfm-evals suite with explicitly managed vLLM server(s).
# This avoids Inspect self-spawn and gives full control over server topology.

#SBATCH --job-name=dfm-evals-vllm-suite
#SBATCH --account=project_465002183
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output=dfm-evals-vllm-suite-%j.out
#SBATCH --error=dfm-evals-vllm-suite-%j.err

set -euo pipefail

REPO_ROOT="${DFM_EVALS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_ROOT/lumi"
source "$SCRIPT_DIR/laif_runtime.sh"
source "$SCRIPT_DIR/artifact_root.sh"
POST_ARTIFACT_ROOT="$(resolve_post_artifact_root "$REPO_ROOT")"
export POST_ARTIFACT_ROOT

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
SIF=${SIF:-$(dfm_lumi_default_sif)}
OVERLAY_DIR=${OVERLAY_DIR:-$(dfm_lumi_default_overlay_dir)}
SINGULARITY_GPU_MODE=${SINGULARITY_GPU_MODE:-$(dfm_lumi_default_gpu_mode "$SIF")}

MODEL=${MODEL:-google/gemma-3-4b-it}
DFM_EVALS_MODEL=${DFM_EVALS_MODEL:-vllm/$MODEL}
DFM_EVALS_TARGET_MODEL=${DFM_EVALS_TARGET_MODEL:-$DFM_EVALS_MODEL}
JUDGE_MODEL=${JUDGE_MODEL:-}
DFM_EVALS_JUDGE_MODEL=${DFM_EVALS_JUDGE_MODEL:-${JUDGE_MODEL:-$DFM_EVALS_TARGET_MODEL}}
DFM_EVALS_OPENAI_BASE_URL=${DFM_EVALS_OPENAI_BASE_URL:-${OPENAI_BASE_URL:-}}
if [[ -z "${JUDGE_MODEL_AUTO:-}" ]]; then
  if [[ -n "$JUDGE_MODEL" || -n "${DFM_EVALS_JUDGE_MODEL:-}" ]]; then
    JUDGE_MODEL_AUTO=0
  else
    JUDGE_MODEL_AUTO=1
  fi
fi
DFM_EVALS_SUITE=${DFM_EVALS_SUITE:-fundamentals}
DFM_EVALS_SUITES_FILE=${DFM_EVALS_SUITES_FILE:-}
DFM_EVALS_LIMIT=${DFM_EVALS_LIMIT-20}
case "${DFM_EVALS_LIMIT}" in
  none|off|all)
    DFM_EVALS_LIMIT=""
    ;;
esac
DFM_EVALS_MODAL_ENABLE_OUTPUT=${DFM_EVALS_MODAL_ENABLE_OUTPUT:-0}
DFM_EVALS_MODAL_SANDBOX_TIMEOUT=${DFM_EVALS_MODAL_SANDBOX_TIMEOUT:-}
DFM_EVALS_MODAL_SANDBOX_IDLE_TIMEOUT=${DFM_EVALS_MODAL_SANDBOX_IDLE_TIMEOUT:-}
DFM_EVALS_RUNNER_IDLE_TIMEOUT=${DFM_EVALS_RUNNER_IDLE_TIMEOUT:-900}
DFM_EVALS_RUNNER_POLL_INTERVAL=${DFM_EVALS_RUNNER_POLL_INTERVAL:-30}
DFM_EVALS_RUNNER_KILL_GRACE=${DFM_EVALS_RUNNER_KILL_GRACE:-30}
DFM_EVALS_SALVAGE_PARTIAL_RESULTS=${DFM_EVALS_SALVAGE_PARTIAL_RESULTS:-1}
DFM_EVALS_TEMPERATURE=${DFM_EVALS_TEMPERATURE:-0}
DFM_EVALS_EXTRA_ARGS=${DFM_EVALS_EXTRA_ARGS:-}
DFM_EVALS_EXPORT_EEE=${DFM_EVALS_EXPORT_EEE:-1}
DFM_EVALS_EEE_OUTPUT_DIR=${DFM_EVALS_EEE_OUTPUT_DIR:-$POST_ARTIFACT_ROOT/evals/eee/data}
DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME=${DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME:-dfm-evals}
DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP=${DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP:-third_party}
DFM_EVALS_EEE_MODEL_ID=${DFM_EVALS_EEE_MODEL_ID:-}

TP=${TP:-8}
PP=${PP:-1}
DP=${DP:-1}
CTX=${CTX:-}
GPU_MEM=${GPU_MEM:-0.92}
MAX_CONNECTIONS=${MAX_CONNECTIONS:-128}
MAX_TASKS=${MAX_TASKS:-4}
DFM_EVALS_CLIENT_TIMEOUT=${DFM_EVALS_CLIENT_TIMEOUT:-7200}
DFM_EVALS_CLIENT_MAX_RETRIES=${DFM_EVALS_CLIENT_MAX_RETRIES:-0}
TARGET_PORT=${TARGET_PORT:-8000}
TARGET_VISIBLE_DEVICES=${TARGET_VISIBLE_DEVICES:-}
TARGET_ENABLE_AUTO_TOOL_CHOICE=${TARGET_ENABLE_AUTO_TOOL_CHOICE:-1}
TARGET_TOOL_CALL_PARSER=${TARGET_TOOL_CALL_PARSER:-hermes}
TARGET_TOOL_PARSER_PLUGIN=${TARGET_TOOL_PARSER_PLUGIN:-}
TARGET_REASONING_PARSER=${TARGET_REASONING_PARSER:-}
TARGET_TOKENIZER_MODE=${TARGET_TOKENIZER_MODE:-}
TARGET_CHAT_TEMPLATE_KWARGS_JSON=${TARGET_CHAT_TEMPLATE_KWARGS_JSON:-}
TARGET_ENFORCE_EAGER=${TARGET_ENFORCE_EAGER:-0}
TARGET_ATTENTION_BACKEND=${TARGET_ATTENTION_BACKEND:-}

JUDGE_SERVER_ENABLED=${JUDGE_SERVER_ENABLED:-0}
JUDGE_SERVER_MODEL=${JUDGE_SERVER_MODEL:-$MODEL}
JUDGE_SERVER_SERVED_MODEL_NAME=${JUDGE_SERVER_SERVED_MODEL_NAME:-}
JUDGE_PORT=${JUDGE_PORT:-8001}
JUDGE_TP=${JUDGE_TP:-1}
JUDGE_PP=${JUDGE_PP:-1}
JUDGE_DP=${JUDGE_DP:-1}
JUDGE_CTX=${JUDGE_CTX:-$CTX}
JUDGE_GPU_MEM=${JUDGE_GPU_MEM:-0.85}
JUDGE_VISIBLE_DEVICES=${JUDGE_VISIBLE_DEVICES:-}
JUDGE_ENABLE_AUTO_TOOL_CHOICE=${JUDGE_ENABLE_AUTO_TOOL_CHOICE:-0}
JUDGE_TOOL_CALL_PARSER=${JUDGE_TOOL_CALL_PARSER:-}
JUDGE_TOOL_PARSER_PLUGIN=${JUDGE_TOOL_PARSER_PLUGIN:-}
JUDGE_REASONING_PARSER=${JUDGE_REASONING_PARSER:-}
JUDGE_TOKENIZER_MODE=${JUDGE_TOKENIZER_MODE:-}
JUDGE_CHAT_TEMPLATE_KWARGS_JSON=${JUDGE_CHAT_TEMPLATE_KWARGS_JSON:-}
JUDGE_ENFORCE_EAGER=${JUDGE_ENFORCE_EAGER:-0}
JUDGE_ATTENTION_BACKEND=${JUDGE_ATTENTION_BACKEND:-}

case "${CTX}" in
  ""|auto|none)
    CTX=""
    ;;
esac
case "${JUDGE_CTX}" in
  ""|auto|none)
    JUDGE_CTX=""
    ;;
esac

SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-900}
SERVER_START_STEP=${SERVER_START_STEP:-5}

LORA_MAX_RANK_OVERRIDE=${LORA_MAX_RANK_OVERRIDE:-}

HF_HOME_HOST=${HF_HOME:-/scratch/project_465002183/.cache/huggingface}
HF_HOME_HOST=${HF_HOME_HOST%/}
HF_HOME_CONTAINER=/hf-home
mkdir -p "$HF_HOME_HOST"
HF_HOME_HOST="$(readlink -f "$HF_HOME_HOST" 2>/dev/null || printf '%s' "$HF_HOME_HOST")"
SLURM_JOB_ID_SAFE=${SLURM_JOB_ID:-manual}
DFM_EVALS_RUN_ROOT=${DFM_EVALS_RUN_ROOT:-$POST_ARTIFACT_ROOT/evals/runs}
DFM_EVALS_LOG_ROOT=${DFM_EVALS_LOG_ROOT:-$POST_ARTIFACT_ROOT/evals/logs}

TARGET_VLLM_URL=""
TARGET_SERVER_PID=""
TARGET_SERVER_LOG=""
JUDGE_VLLM_URL=""
JUDGE_SERVER_PID=""
JUDGE_SERVER_LOG=""
EVAL_RUN_PID=""
EVAL_RUN_EXIT_CODE=0
EVAL_RUN_STALLED=0

ORIG_DFM_EVALS_MODEL="$DFM_EVALS_MODEL"
ORIG_DFM_EVALS_TARGET_MODEL="$DFM_EVALS_TARGET_MODEL"
ORIG_DFM_EVALS_JUDGE_MODEL="$DFM_EVALS_JUDGE_MODEL"

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

TARGET_SERVER_MODEL_TAG=""
TARGET_SERVER_SERVED_MODEL_NAME=""
TARGET_SERVER_API_KEY="${TARGET_SERVER_API_KEY:-${VLLM_API_KEY:-inspectai}}"

JUDGE_USE_LORA_ADAPTER=0
JUDGE_LORA_BASE_MODEL=""
JUDGE_LORA_MODULE_NAME=""
JUDGE_LORA_MODULE_SPEC=""
JUDGE_LORA_ADAPTER_PATH="$JUDGE_SERVER_MODEL"
JUDGE_LORA_CHAT_TEMPLATE_PATH=""
JUDGE_LORA_TOKENIZER_PATH=""
JUDGE_LORA_MAX_RANK="${LORA_MAX_RANK_OVERRIDE:-}"
JUDGE_SERVER_MODEL_TAG="$JUDGE_SERVER_MODEL"
JUDGE_SERVER_API_KEY="${JUDGE_SERVER_API_KEY:-${OPENAI_API_KEY:-inspectai}}"

die() {
  echo "FATAL: $*" >&2
  exit 1
}

if [[ "$TARGET_TOOL_CALL_PARSER" == "none" ]]; then
  TARGET_TOOL_CALL_PARSER=""
fi
if [[ "$TARGET_REASONING_PARSER" == "none" ]]; then
  TARGET_REASONING_PARSER=""
fi
if [[ "$TARGET_TOOL_PARSER_PLUGIN" == "none" ]]; then
  TARGET_TOOL_PARSER_PLUGIN=""
fi
if [[ "$TARGET_TOKENIZER_MODE" == "none" ]]; then
  TARGET_TOKENIZER_MODE=""
fi
if [[ "$JUDGE_TOOL_CALL_PARSER" == "none" ]]; then
  JUDGE_TOOL_CALL_PARSER=""
fi
if [[ "$JUDGE_REASONING_PARSER" == "none" ]]; then
  JUDGE_REASONING_PARSER=""
fi
if [[ "$JUDGE_TOOL_PARSER_PLUGIN" == "none" ]]; then
  JUDGE_TOOL_PARSER_PLUGIN=""
fi
if [[ "$JUDGE_TOKENIZER_MODE" == "none" ]]; then
  JUDGE_TOKENIZER_MODE=""
fi
if [[ "$TARGET_ATTENTION_BACKEND" == "none" ]]; then
  TARGET_ATTENTION_BACKEND=""
fi
if [[ "$JUDGE_ATTENTION_BACKEND" == "none" ]]; then
  JUDGE_ATTENTION_BACKEND=""
fi

case "$TARGET_ENABLE_AUTO_TOOL_CHOICE" in
  0|1) ;;
  *) die "TARGET_ENABLE_AUTO_TOOL_CHOICE must be 0 or 1 (got: $TARGET_ENABLE_AUTO_TOOL_CHOICE)" ;;
esac
case "$TARGET_ENFORCE_EAGER" in
  0|1) ;;
  *) die "TARGET_ENFORCE_EAGER must be 0 or 1 (got: $TARGET_ENFORCE_EAGER)" ;;
esac
case "$JUDGE_ENABLE_AUTO_TOOL_CHOICE" in
  0|1) ;;
  *) die "JUDGE_ENABLE_AUTO_TOOL_CHOICE must be 0 or 1 (got: $JUDGE_ENABLE_AUTO_TOOL_CHOICE)" ;;
esac
case "$JUDGE_ENFORCE_EAGER" in
  0|1) ;;
  *) die "JUDGE_ENFORCE_EAGER must be 0 or 1 (got: $JUDGE_ENFORCE_EAGER)" ;;
esac

if [[ "$TARGET_ENABLE_AUTO_TOOL_CHOICE" == "1" && -z "$TARGET_TOOL_CALL_PARSER" ]]; then
  die "target auto tool choice enabled but target tool call parser is empty"
fi
if [[ "$JUDGE_ENABLE_AUTO_TOOL_CHOICE" == "1" && -z "$JUDGE_TOOL_CALL_PARSER" ]]; then
  die "judge auto tool choice enabled but judge tool call parser is empty"
fi

parse_adapter_metadata() {
  local adapter_config="$1"
  ADAPTER_CONFIG_PATH="$adapter_config" python3 - <<'PY'
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

if [[ "$DFM_EVALS_MODEL" == vllm/* && -d "$MODEL" && -f "$ADAPTER_CONFIG_PATH" && -f "$ADAPTER_WEIGHTS_PATH" ]]; then
  adapter_meta="$(parse_adapter_metadata "$ADAPTER_CONFIG_PATH")" || die "failed to parse adapter metadata from $ADAPTER_CONFIG_PATH"

  IFS=$'\t' read -r LORA_BASE_MODEL LORA_MAX_RANK <<<"$adapter_meta"
  [[ -n "$LORA_BASE_MODEL" ]] || die "empty base model in $ADAPTER_CONFIG_PATH"
  [[ -n "$LORA_MAX_RANK" ]] || die "empty adapter rank in $ADAPTER_CONFIG_PATH"

  USE_LORA_ADAPTER=1

  # Preserve a stable LoRA-serving model id for eval metadata/export; do not
  # collapse it to the base model id.
  LORA_MODULE_NAME="${DFM_EVALS_MODEL#vllm/}"
  LORA_BASE_DEVELOPER="$(model_developer_from_ref "$LORA_BASE_MODEL")"
  if [[ "$LORA_MODULE_NAME" == "$DFM_EVALS_MODEL" || "$LORA_MODULE_NAME" == "$MODEL" || -z "$LORA_MODULE_NAME" ]]; then
    LORA_MODULE_NAME="$(model_label_from_ref "$MODEL")"
    if [[ -n "$LORA_BASE_DEVELOPER" ]]; then
      LORA_MODULE_NAME="${LORA_BASE_DEVELOPER}/${LORA_MODULE_NAME}"
    fi
  fi
  [[ -n "$LORA_MODULE_NAME" ]] || LORA_MODULE_NAME="$LORA_BASE_MODEL"
  [[ -n "$LORA_MODULE_NAME" ]] || LORA_MODULE_NAME="adapter"
  if [[ "$LORA_MODULE_NAME" == "$LORA_BASE_MODEL" ]]; then
    LORA_MODULE_NAME="${LORA_MODULE_NAME}--lora"
  fi

  if [[ "$DFM_EVALS_MODEL" == "vllm/$MODEL" || "$DFM_EVALS_MODEL" == "$MODEL" ]]; then
    DFM_EVALS_MODEL="vllm/$LORA_MODULE_NAME"
  fi
  if [[ "$DFM_EVALS_TARGET_MODEL" == "$ORIG_DFM_EVALS_MODEL" || "$DFM_EVALS_TARGET_MODEL" == "vllm/$MODEL" || "$DFM_EVALS_TARGET_MODEL" == "$MODEL" ]]; then
    DFM_EVALS_TARGET_MODEL="$DFM_EVALS_MODEL"
  fi
  if [[ "$DFM_EVALS_JUDGE_MODEL" == "$ORIG_DFM_EVALS_TARGET_MODEL" || "$DFM_EVALS_JUDGE_MODEL" == "$ORIG_DFM_EVALS_MODEL" || "$DFM_EVALS_JUDGE_MODEL" == "vllm/$MODEL" || "$DFM_EVALS_JUDGE_MODEL" == "$MODEL" ]]; then
    DFM_EVALS_JUDGE_MODEL="$DFM_EVALS_TARGET_MODEL"
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

  if [[ -f "$MODEL/tokenizer_config.json" || -f "$MODEL/tokenizer.json" || -f "$MODEL/tokenizer.model" || -f "$MODEL/vocab.json" ]]; then
    if resolved_tokenizer_path="$(readlink -f "$MODEL" 2>/dev/null)"; then
      LORA_TOKENIZER_PATH="$resolved_tokenizer_path"
    else
      LORA_TOKENIZER_PATH="$MODEL"
    fi
  fi
fi

if [[ "$JUDGE_SERVER_ENABLED" == "1" ]]; then
  if [[ -z "$JUDGE_SERVER_SERVED_MODEL_NAME" ]]; then
    if [[ "$DFM_EVALS_JUDGE_MODEL" == openai/* ]]; then
      JUDGE_SERVER_SERVED_MODEL_NAME="${DFM_EVALS_JUDGE_MODEL#openai/}"
    elif [[ "$DFM_EVALS_JUDGE_MODEL" == vllm/* ]]; then
      JUDGE_SERVER_SERVED_MODEL_NAME="${DFM_EVALS_JUDGE_MODEL#vllm/}"
    else
      JUDGE_SERVER_SERVED_MODEL_NAME="$DFM_EVALS_JUDGE_MODEL"
    fi
  fi

  [[ -n "$JUDGE_SERVER_SERVED_MODEL_NAME" ]] || die "JUDGE_SERVER_SERVED_MODEL_NAME resolved empty"

  if [[ "$DFM_EVALS_JUDGE_MODEL" != openai/* ]]; then
    DFM_EVALS_JUDGE_MODEL="openai/$JUDGE_SERVER_SERVED_MODEL_NAME"
  fi

  judge_adapter_config="$JUDGE_SERVER_MODEL/adapter_config.json"
  judge_adapter_weights="$JUDGE_SERVER_MODEL/adapter_model.safetensors"
  judge_adapter_chat_template="$JUDGE_SERVER_MODEL/chat_template.jinja"

  if [[ -d "$JUDGE_SERVER_MODEL" && -f "$judge_adapter_config" && -f "$judge_adapter_weights" ]]; then
    judge_adapter_meta="$(parse_adapter_metadata "$judge_adapter_config")" || die "failed to parse adapter metadata from $judge_adapter_config"

    IFS=$'\t' read -r JUDGE_LORA_BASE_MODEL JUDGE_LORA_MAX_RANK <<<"$judge_adapter_meta"
    [[ -n "$JUDGE_LORA_BASE_MODEL" ]] || die "empty base model in $judge_adapter_config"
    [[ -n "$JUDGE_LORA_MAX_RANK" ]] || die "empty adapter rank in $judge_adapter_config"

    JUDGE_USE_LORA_ADAPTER=1
    JUDGE_SERVER_MODEL_TAG="$JUDGE_LORA_BASE_MODEL"

    JUDGE_LORA_MODULE_NAME="$JUDGE_SERVER_SERVED_MODEL_NAME"
    [[ -n "$JUDGE_LORA_MODULE_NAME" ]] || JUDGE_LORA_MODULE_NAME="$JUDGE_LORA_BASE_MODEL"
    [[ -n "$JUDGE_LORA_MODULE_NAME" ]] || JUDGE_LORA_MODULE_NAME="adapter"

    if resolved_judge_adapter_path="$(readlink -f "$JUDGE_SERVER_MODEL" 2>/dev/null)"; then
      JUDGE_LORA_ADAPTER_PATH="$resolved_judge_adapter_path"
    fi
    JUDGE_LORA_MODULE_SPEC="${JUDGE_LORA_MODULE_NAME}=${JUDGE_LORA_ADAPTER_PATH}"

    if [[ -f "$judge_adapter_chat_template" ]]; then
      if resolved_judge_chat_template="$(readlink -f "$judge_adapter_chat_template" 2>/dev/null)"; then
        JUDGE_LORA_CHAT_TEMPLATE_PATH="$resolved_judge_chat_template"
      else
        JUDGE_LORA_CHAT_TEMPLATE_PATH="$judge_adapter_chat_template"
      fi
    fi

    if [[ -f "$JUDGE_SERVER_MODEL/tokenizer_config.json" || -f "$JUDGE_SERVER_MODEL/tokenizer.json" || -f "$JUDGE_SERVER_MODEL/tokenizer.model" || -f "$JUDGE_SERVER_MODEL/vocab.json" ]]; then
      if resolved_judge_tokenizer_path="$(readlink -f "$JUDGE_SERVER_MODEL" 2>/dev/null)"; then
        JUDGE_LORA_TOKENIZER_PATH="$resolved_judge_tokenizer_path"
      else
        JUDGE_LORA_TOKENIZER_PATH="$JUDGE_SERVER_MODEL"
      fi
    fi
  fi
fi

# If judge model was auto-selected in submit.sh, derive it from the
# normalized target model (important for local LoRA adapter paths).
if [[ "$JUDGE_MODEL_AUTO" == "1" ]]; then
  if [[ "$DFM_EVALS_TARGET_MODEL" == openai/* ]]; then
    DFM_EVALS_JUDGE_MODEL="$DFM_EVALS_TARGET_MODEL"
  elif [[ "$DFM_EVALS_TARGET_MODEL" == vllm/* ]]; then
    DFM_EVALS_JUDGE_MODEL="openai/${DFM_EVALS_TARGET_MODEL#vllm/}"
  else
    DFM_EVALS_JUDGE_MODEL="openai/$DFM_EVALS_TARGET_MODEL"
  fi
fi

# Inspect's get_model requires <api>/<model>; accept plain judge names
# (e.g., qwen-235b) from submit env and normalize here.
if [[ "$DFM_EVALS_JUDGE_MODEL" != */* ]]; then
  DFM_EVALS_JUDGE_MODEL="openai/$DFM_EVALS_JUDGE_MODEL"
fi

TARGET_SERVER_MODEL_TAG="$MODEL"
if [[ "$DFM_EVALS_MODEL" == vllm/* ]]; then
  TARGET_SERVER_MODEL_TAG="${DFM_EVALS_MODEL#vllm/}"
fi
TARGET_SERVER_SERVED_MODEL_NAME="$TARGET_SERVER_MODEL_TAG"
if [[ "$USE_LORA_ADAPTER" == "1" ]]; then
  TARGET_SERVER_MODEL_TAG="$LORA_BASE_MODEL"
  TARGET_SERVER_SERVED_MODEL_NAME="$LORA_MODULE_NAME"
fi

if [[ -z "$JUDGE_SERVER_MODEL_TAG" ]]; then
  JUDGE_SERVER_MODEL_TAG="$JUDGE_SERVER_MODEL"
fi

if [[ "$DFM_EVALS_MODEL" == vllm/* ]]; then
  if [[ "$DFM_EVALS_TARGET_MODEL" == vllm/* && "${DFM_EVALS_TARGET_MODEL#vllm/}" != "${DFM_EVALS_MODEL#vllm/}" ]]; then
    die "DFM_EVALS_TARGET_MODEL ($DFM_EVALS_TARGET_MODEL) must match DFM_EVALS_MODEL ($DFM_EVALS_MODEL) when using a single target vLLM server"
  fi
  if [[ "$DFM_EVALS_JUDGE_MODEL" == vllm/* && "${DFM_EVALS_JUDGE_MODEL#vllm/}" != "${DFM_EVALS_MODEL#vllm/}" ]]; then
    die "DFM_EVALS_JUDGE_MODEL ($DFM_EVALS_JUDGE_MODEL) must match DFM_EVALS_MODEL ($DFM_EVALS_MODEL) unless judge server is configured as openai/*"
  fi
fi

if [[ "$JUDGE_SERVER_ENABLED" == "1" && "$DFM_EVALS_JUDGE_MODEL" != openai/* ]]; then
  die "JUDGE_SERVER_ENABLED=1 requires DFM_EVALS_JUDGE_MODEL to resolve to openai/*"
fi

label_model_ref="$DFM_EVALS_TARGET_MODEL"
if [[ "$USE_LORA_ADAPTER" == "1" ]]; then
  label_model_ref="$MODEL"
fi
default_model_label="$(model_label_from_ref "$label_model_ref")"
if [[ -n "${DFM_EVALS_RUN_LABEL:-}" ]]; then
  _raw_dfm_label="$DFM_EVALS_RUN_LABEL"
else
  _raw_dfm_label="${DFM_EVALS_SUITE}__${default_model_label}__job-${SLURM_JOB_ID_SAFE}"
fi
DFM_EVALS_RUN_LABEL=${_raw_dfm_label//[^[:alnum:]._-]/_}
DFM_EVALS_CACHE_DIR=${DFM_EVALS_CACHE_DIR:-/overlay/cache/dfm-evals-${DFM_EVALS_RUN_LABEL}}
DFM_EVALS_RUN_DIR=${DFM_EVALS_RUN_DIR:-${DFM_EVALS_RUN_ROOT}/${DFM_EVALS_RUN_LABEL}}
DFM_EVALS_LOG_DIR=${DFM_EVALS_LOG_DIR:-${DFM_EVALS_LOG_ROOT}/${DFM_EVALS_RUN_LABEL}}
DFM_EVALS_SERVER_LOG_DIR=${DFM_EVALS_SERVER_LOG_DIR:-${DFM_EVALS_LOG_DIR}/_vllm_server}
DFM_EVALS_XDG_CACHE_HOME=${DFM_EVALS_XDG_CACHE_HOME:-${DFM_EVALS_CACHE_DIR}/xdg-cache}
DFM_EVALS_XDG_DATA_HOME=${DFM_EVALS_XDG_DATA_HOME:-${DFM_EVALS_CACHE_DIR}/xdg-data}
DFM_EVALS_TRITON_CACHE_DIR=${DFM_EVALS_TRITON_CACHE_DIR:-${DFM_EVALS_CACHE_DIR}/triton}
DFM_EVALS_TORCHINDUCTOR_CACHE_DIR=${DFM_EVALS_TORCHINDUCTOR_CACHE_DIR:-${DFM_EVALS_CACHE_DIR}/torchinductor}
DFM_EVALS_RULER_QA_CACHE=${DFM_EVALS_RULER_QA_CACHE:-${POST_ARTIFACT_ROOT}/evals/ruler_qa_cache}
DFM_EVALS_RULER_QA_OFFLINE=${DFM_EVALS_RULER_QA_OFFLINE:-1}
TARGET_SERVER_CACHE_ROOT=${TARGET_SERVER_CACHE_ROOT:-${DFM_EVALS_CACHE_DIR}/target-server}
JUDGE_SERVER_CACHE_ROOT=${JUDGE_SERVER_CACHE_ROOT:-${DFM_EVALS_CACHE_DIR}/judge-server}
if [[ -z "$DFM_EVALS_EEE_OUTPUT_DIR" ]]; then
  DFM_EVALS_EEE_OUTPUT_DIR="${POST_ARTIFACT_ROOT}/evals/eee/data"
fi

case "${DFM_EVALS_RULER_QA_OFFLINE,,}" in
  1|true|yes|on)
    [[ -s "$DFM_EVALS_RULER_QA_CACHE/squad.json" ]] || die "missing cached RULER QA dataset: $DFM_EVALS_RULER_QA_CACHE/squad.json"
    [[ -s "$DFM_EVALS_RULER_QA_CACHE/hotpotqa.json" ]] || die "missing cached RULER QA dataset: $DFM_EVALS_RULER_QA_CACHE/hotpotqa.json"
    ;;
esac

normalize_tokenizer_dir() {
  local source_dir="$1"
  local normalized_dir="$2"

  [[ -d "$source_dir" ]] || die "tokenizer dir not found: $source_dir"
  mkdir -p "$normalized_dir"
  cp -a "$source_dir"/. "$normalized_dir"/

  if [[ -f "$normalized_dir/tokenizer_config.json" ]]; then
    TOKENIZER_CONFIG_PATH="$normalized_dir/tokenizer_config.json" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["TOKENIZER_CONFIG_PATH"])
with path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

extra = cfg.get("extra_special_tokens")
if isinstance(extra, list):
    existing = cfg.get("additional_special_tokens")
    if isinstance(existing, list):
        merged = list(existing)
    elif isinstance(existing, tuple):
        merged = list(existing)
    elif existing in (None, ""):
        merged = []
    else:
        merged = [existing]

    for token in extra:
        if token not in merged:
            merged.append(token)

    cfg["additional_special_tokens"] = merged

    model_specific = cfg.get("model_specific_special_tokens")
    if isinstance(model_specific, dict):
        cfg["extra_special_tokens"] = model_specific
    else:
        cfg.pop("extra_special_tokens", None)

with path.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
  fi
}

if [[ -n "$LORA_TOKENIZER_PATH" ]]; then
  TARGET_NORMALIZED_TOKENIZER_DIR="${DFM_EVALS_RUN_DIR}/_normalized_tokenizers/target"
  normalize_tokenizer_dir "$LORA_TOKENIZER_PATH" "$TARGET_NORMALIZED_TOKENIZER_DIR"
  LORA_TOKENIZER_PATH="$TARGET_NORMALIZED_TOKENIZER_DIR"
fi

if [[ -n "$JUDGE_LORA_TOKENIZER_PATH" ]]; then
  JUDGE_NORMALIZED_TOKENIZER_DIR="${DFM_EVALS_RUN_DIR}/_normalized_tokenizers/judge"
  normalize_tokenizer_dir "$JUDGE_LORA_TOKENIZER_PATH" "$JUDGE_NORMALIZED_TOKENIZER_DIR"
  JUDGE_LORA_TOKENIZER_PATH="$JUDGE_NORMALIZED_TOKENIZER_DIR"
fi

build_model_info_overrides() {
  local -n out_json_ref=$1
  out_json_ref="$(
    DFM_EVALS_MODEL="$DFM_EVALS_MODEL" \
    DFM_EVALS_TARGET_MODEL="$DFM_EVALS_TARGET_MODEL" \
    DFM_EVALS_JUDGE_MODEL="$DFM_EVALS_JUDGE_MODEL" \
    CTX="$CTX" \
    JUDGE_CTX="$JUDGE_CTX" \
    python3 - <<'PY'
import json
import os

overrides = {}


def maybe_add(model_env, ctx_env):
    model_name = os.environ.get(model_env, "").strip()
    ctx = os.environ.get(ctx_env, "").strip()
    if not model_name or not ctx or not model_name.startswith("vllm/"):
        return

    try:
        context_length = int(ctx)
    except ValueError as exc:
        raise SystemExit(f"{ctx_env} must be an integer (got {ctx!r})") from exc

    if context_length <= 0:
        raise SystemExit(f"{ctx_env} must be positive (got {context_length})")

    overrides[model_name] = {"context_length": context_length}


maybe_add("DFM_EVALS_MODEL", "CTX")
maybe_add("DFM_EVALS_TARGET_MODEL", "CTX")
maybe_add("DFM_EVALS_JUDGE_MODEL", "JUDGE_CTX")

print(json.dumps(overrides, separators=(",", ":")))
PY
  )"
}

build_evals_cli() {
  local -n out_cmd_ref=$1
  local -a evals_args

  evals_args=(
    suite
    "$DFM_EVALS_SUITE"
    --mode "run"
    --target-model "$DFM_EVALS_TARGET_MODEL"
    --judge-model "$DFM_EVALS_JUDGE_MODEL"
  )

  if [[ -n "$DFM_EVALS_SUITES_FILE" ]]; then
    evals_args+=(--file "$DFM_EVALS_SUITES_FILE")
  fi

  evals_args+=(
    --
    --log-dir "$DFM_EVALS_LOG_DIR"
    --model "$DFM_EVALS_MODEL"
    --temperature "$DFM_EVALS_TEMPERATURE"
    --max-connections "$MAX_CONNECTIONS"
    --max-tasks "$MAX_TASKS"
    -M "timeout=$DFM_EVALS_CLIENT_TIMEOUT"
    -M "max_retries=$DFM_EVALS_CLIENT_MAX_RETRIES"
  )
  if [[ -n "${DFM_EVALS_LIMIT:-}" ]]; then
    evals_args+=(--limit "$DFM_EVALS_LIMIT")
  fi

  local escaped_args=""
  printf -v escaped_args '%q ' "${evals_args[@]}"
  out_cmd_ref="python -m dfm_evals.cli ${escaped_args}"
  if [[ -n "$DFM_EVALS_EXTRA_ARGS" ]]; then
    out_cmd_ref+="$DFM_EVALS_EXTRA_ARGS"
  fi
}

container_env_prefix() {
  cat <<'EOC'
set -euo pipefail
source /overlay/venv/vllm-min/bin/activate
DFM_EVALS_REPO_ROOT="__DFM_EVALS_REPO_ROOT__"
export DFM_EVALS_REPO_ROOT
if [[ -f /overlay/overlay-runtime.env ]]; then
  # shellcheck disable=SC1091
  source /overlay/overlay-runtime.env
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${DFM_EVALS_REPO_ROOT}:${PYTHONPATH}"
  else
    export PYTHONPATH="${DFM_EVALS_REPO_ROOT}"
  fi
else
  BASE_SITE=/opt/venv/lib/python3.12/site-packages
  OVERLAY_SITE=/overlay/venv/vllm-min/lib/python3.12/site-packages
  # Keep overlay packages ahead of the base container, but still surface shared
  # dependencies that live only in /opt/venv (for example PyYAML).
  export PYTHONPATH=${DFM_EVALS_REPO_ROOT}:/overlay/src/transformers/src:/overlay/src/vllm:${OVERLAY_SITE}:${BASE_SITE}${PYTHONPATH:+:$PYTHONPATH}
fi
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi
unset ROCR_VISIBLE_DEVICES
export HF_HOME="__HF_HOME_CONTAINER__"
EOC
}

render_env() {
  container_env_prefix \
    | sed "s|__HF_HOME_CONTAINER__|$HF_HOME_CONTAINER|g" \
    | sed "s|__DFM_EVALS_REPO_ROOT__|$REPO_ROOT|g"
}

start_vllm_server() {
  local name="$1"
  local model_tag="$2"
  local served_model_name="$3"
  local port="$4"
  local api_key="$5"
  local tp="$6"
  local pp="$7"
  local dp="$8"
  local ctx="$9"
  local gpu_mem="${10}"
  local visible_devices="${11}"
  local use_lora="${12}"
  local lora_spec="${13}"
  local lora_rank="${14}"
  local chat_template="${15}"
  local tokenizer_path="${16}"
  local enable_auto_tool_choice="${17}"
  local tool_call_parser="${18}"
  local reasoning_parser="${19}"
  local tokenizer_mode="${20}"
  local chat_template_kwargs_json="${21}"
  local enforce_eager="${22}"
  local attention_backend="${23}"
  local tool_parser_plugin="${24}"
  local log_file="${25}"
  local cache_root="${26}"

  local server_cmd
server_cmd="$(render_env)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_PLUGINS=''
export XDG_CACHE_HOME=\"\${CACHE_ROOT%/}/xdg-cache\"
export XDG_DATA_HOME=\"\${CACHE_ROOT%/}/xdg-data\"
export TRITON_CACHE_DIR=\"\${CACHE_ROOT%/}/triton\"
export TORCHINDUCTOR_CACHE_DIR=\"\${CACHE_ROOT%/}/torchinductor\"
mkdir -p \"\${XDG_CACHE_HOME}\" \"\${XDG_DATA_HOME}\" \"\${TRITON_CACHE_DIR}\" \"\${TORCHINDUCTOR_CACHE_DIR}\"
if [[ -n \"\${VISIBLE_DEVICES:-}\" ]]; then
  export HIP_VISIBLE_DEVICES=\"\${VISIBLE_DEVICES}\"
  export CUDA_VISIBLE_DEVICES=\"\${VISIBLE_DEVICES}\"
fi
EXTRA_VLLM_FLAGS=()
EXTRA_VLLM_FLAGS+=(--enable-prefix-caching)
if [[ \"\${ENABLE_AUTO_TOOL_CHOICE:-0}\" == \"1\" ]]; then
  EXTRA_VLLM_FLAGS+=(--enable-auto-tool-choice)
fi
  if [[ -n \"\${TOOL_PARSER_PLUGIN:-}\" ]]; then
    EXTRA_VLLM_FLAGS+=(--tool-parser-plugin \"\${TOOL_PARSER_PLUGIN}\")
  fi
  if [[ -n \"\${TOOL_CALL_PARSER:-}\" ]]; then
    EXTRA_VLLM_FLAGS+=(--tool-call-parser \"\${TOOL_CALL_PARSER}\")
  fi
  if [[ -n \"\${REASONING_PARSER:-}\" ]]; then
    EXTRA_VLLM_FLAGS+=(--reasoning-parser \"\${REASONING_PARSER}\")
  fi
  if [[ -n \"\${TOKENIZER_MODE:-}\" ]]; then
    EXTRA_VLLM_FLAGS+=(--tokenizer-mode \"\${TOKENIZER_MODE}\")
  fi
  if [[ -n \"\${DEFAULT_CHAT_TEMPLATE_KWARGS:-}\" ]]; then
    EXTRA_VLLM_FLAGS+=(--default-chat-template-kwargs \"\${DEFAULT_CHAT_TEMPLATE_KWARGS}\")
  fi
  if [[ \"\${ENFORCE_EAGER:-0}\" == \"1\" ]]; then
    EXTRA_VLLM_FLAGS+=(--enforce-eager)
  fi
  if [[ -n \"\${ATTENTION_BACKEND:-}\" ]]; then
    EXTRA_VLLM_FLAGS+=(--attention-backend \"\${ATTENTION_BACKEND}\")
  fi
  if [[ -n \"\${CTX:-}\" ]]; then
    EXTRA_VLLM_FLAGS+=(--max-model-len \"\${CTX}\")
  fi
if [[ \"\${USE_LORA_ADAPTER:-0}\" == \"1\" ]]; then
  EXTRA_VLLM_FLAGS+=(--enable-lora --lora-modules \"\${LORA_MODULE_SPEC}\" --max-lora-rank \"\${LORA_MAX_RANK}\")
  if [[ -n \"\${LORA_CHAT_TEMPLATE_PATH:-}\" ]]; then
    EXTRA_VLLM_FLAGS+=(--chat-template \"\${LORA_CHAT_TEMPLATE_PATH}\")
  fi
  if [[ -n \"\${LORA_TOKENIZER_PATH:-}\" ]]; then
    EXTRA_VLLM_FLAGS+=(--tokenizer \"\${LORA_TOKENIZER_PATH}\")
  fi
fi
python -m dfm_evals.vllm_serve serve \"\${MODEL_TAG}\" \\
  --served-model-name \"\${SERVED_MODEL_NAME}\" \\
  --host 0.0.0.0 \\
  --port \"\${PORT}\" \\
  --api-key \"\${API_KEY}\" \\
  --dtype bfloat16 \\
  --tensor-parallel-size \"\${TP}\" \\
  --pipeline-parallel-size \"\${PP}\" \\
  --data-parallel-size \"\${DP}\" \\
  --gpu-memory-utilization \"\${GPU_MEM}\" \\
  --safetensors-load-strategy eager \\
  \"\${EXTRA_VLLM_FLAGS[@]}\""

  echo "Starting $name server on :$port (log: $log_file)"
  singularity exec "${SINGULARITY_GPU_ARGS[@]}" "${SING_BIND_ARGS[@]}" "$SIF" \
    env \
      MODEL_TAG="$model_tag" \
      SERVED_MODEL_NAME="$served_model_name" \
      PORT="$port" \
      API_KEY="$api_key" \
      TP="$tp" \
      PP="$pp" \
      DP="$dp" \
      CTX="$ctx" \
      GPU_MEM="$gpu_mem" \
      VISIBLE_DEVICES="$visible_devices" \
      USE_LORA_ADAPTER="$use_lora" \
      LORA_MODULE_SPEC="$lora_spec" \
      LORA_MAX_RANK="$lora_rank" \
      LORA_CHAT_TEMPLATE_PATH="$chat_template" \
      LORA_TOKENIZER_PATH="$tokenizer_path" \
      ENABLE_AUTO_TOOL_CHOICE="$enable_auto_tool_choice" \
    TOOL_PARSER_PLUGIN="$tool_parser_plugin" \
    TOOL_CALL_PARSER="$tool_call_parser" \
    REASONING_PARSER="$reasoning_parser" \
    TOKENIZER_MODE="$tokenizer_mode" \
    DEFAULT_CHAT_TEMPLATE_KWARGS="$chat_template_kwargs_json" \
    ENFORCE_EAGER="$enforce_eager" \
    ATTENTION_BACKEND="$attention_backend" \
      CACHE_ROOT="$cache_root" \
      bash -lc "$server_cmd" \
      >"$log_file" 2>&1 &

  local server_pid=$!
  local ready=0
  local -a readiness_curl_args
  readiness_curl_args=(curl -sf)
  if [[ -n "$api_key" ]]; then
    readiness_curl_args+=(-H "Authorization: Bearer $api_key")
  fi

  for ((i=0; i<SERVER_START_TIMEOUT; i+=SERVER_START_STEP)); do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      tail -n 120 "$log_file" >&2 || true
      die "$name server exited during startup (pid=$server_pid)"
    fi
    if "${readiness_curl_args[@]}" "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep "$SERVER_START_STEP"
  done

  if [[ "$ready" != "1" ]]; then
    tail -n 120 "$log_file" >&2 || true
    die "$name server did not become ready within ${SERVER_START_TIMEOUT}s"
  fi

  if [[ "$name" == "target" ]]; then
    TARGET_SERVER_PID="$server_pid"
    TARGET_VLLM_URL="http://127.0.0.1:$port/v1"
  else
    JUDGE_SERVER_PID="$server_pid"
    JUDGE_VLLM_URL="http://127.0.0.1:$port/v1"
  fi

  echo "$name server ready at http://127.0.0.1:$port/v1"
}

cleanup() {
  set +e
  if [[ -n "$EVAL_RUN_PID" ]] && kill -0 "$EVAL_RUN_PID" >/dev/null 2>&1; then
    kill "$EVAL_RUN_PID" >/dev/null 2>&1 || true
    wait "$EVAL_RUN_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$JUDGE_SERVER_PID" ]] && kill -0 "$JUDGE_SERVER_PID" >/dev/null 2>&1; then
    kill "$JUDGE_SERVER_PID" >/dev/null 2>&1 || true
    wait "$JUDGE_SERVER_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$TARGET_SERVER_PID" ]] && kill -0 "$TARGET_SERVER_PID" >/dev/null 2>&1; then
    kill "$TARGET_SERVER_PID" >/dev/null 2>&1 || true
    wait "$TARGET_SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

progress_epoch() {
  local newest=""
  {
    find "$DFM_EVALS_LOG_DIR" -maxdepth 1 -type f \
      \( -name '*.eval' -o -name '*.json' \) -printf '%T@\n' 2>/dev/null || true
    find "$DFM_EVALS_XDG_DATA_HOME/inspect_ai/traces" -maxdepth 1 -type f \
      -printf '%T@\n' 2>/dev/null || true
    if [[ -f "$TARGET_SERVER_LOG" ]]; then
      stat -c '%Y' "$TARGET_SERVER_LOG" 2>/dev/null || true
    fi
    if [[ -n "$JUDGE_SERVER_LOG" && -f "$JUDGE_SERVER_LOG" ]]; then
      stat -c '%Y' "$JUDGE_SERVER_LOG" 2>/dev/null || true
    fi
  } | sort -nr | head -n1 || true
}

salvage_partial_results() {
  if [[ "$DFM_EVALS_SALVAGE_PARTIAL_RESULTS" != "1" ]]; then
    return 0
  fi

  local salvage_cmd
  printf -v salvage_cmd '%q %q' \
    "$REPO_ROOT/lumi/salvage_eval.py" \
    "$DFM_EVALS_LOG_DIR"

  singularity exec "${SINGULARITY_GPU_ARGS[@]}" "${SING_BIND_ARGS[@]}" "$SIF" \
    bash -lc "$(render_env)
cd $(printf '%q' "$REPO_ROOT")
python ${salvage_cmd}" || true
}

run_eval_with_watchdog() {
  singularity exec "${SINGULARITY_GPU_ARGS[@]}" "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$RUN_CMD" &
  EVAL_RUN_PID=$!

  local last_progress_wall
  local last_progress_epoch
  local current_progress_epoch
  local now

  last_progress_wall="$(date +%s)"
  last_progress_epoch="$(progress_epoch)"
  if [[ -z "$last_progress_epoch" ]]; then
    last_progress_epoch=0
  fi

  while kill -0 "$EVAL_RUN_PID" >/dev/null 2>&1; do
    sleep "$DFM_EVALS_RUNNER_POLL_INTERVAL"
    current_progress_epoch="$(progress_epoch)"
    if [[ -n "$current_progress_epoch" && "$current_progress_epoch" != "$last_progress_epoch" ]]; then
      last_progress_epoch="$current_progress_epoch"
      last_progress_wall="$(date +%s)"
    fi

    if [[ "$DFM_EVALS_RUNNER_IDLE_TIMEOUT" =~ ^[0-9]+$ ]] && (( DFM_EVALS_RUNNER_IDLE_TIMEOUT > 0 )); then
      now="$(date +%s)"
      if (( now - last_progress_wall >= DFM_EVALS_RUNNER_IDLE_TIMEOUT )); then
        echo "WARNING: dfm-evals runner observed no progress for ${DFM_EVALS_RUNNER_IDLE_TIMEOUT}s; terminating eval subprocess pid=$EVAL_RUN_PID" >&2
        EVAL_RUN_STALLED=1
        kill "$EVAL_RUN_PID" >/dev/null 2>&1 || true
        local deadline=$(( now + DFM_EVALS_RUNNER_KILL_GRACE ))
        while kill -0 "$EVAL_RUN_PID" >/dev/null 2>&1 && (( $(date +%s) < deadline )); do
          sleep 1
        done
        if kill -0 "$EVAL_RUN_PID" >/dev/null 2>&1; then
          kill -9 "$EVAL_RUN_PID" >/dev/null 2>&1 || true
        fi
        break
      fi
    fi
  done

  if wait "$EVAL_RUN_PID"; then
    EVAL_RUN_EXIT_CODE=0
  else
    EVAL_RUN_EXIT_CODE=$?
  fi
}

[[ -f "$SIF" ]] || die "SIF not found: $SIF"
[[ -d "$OVERLAY_DIR/venv/vllm-min" ]] || die "overlay venv missing: $OVERLAY_DIR/venv/vllm-min"
dfm_lumi_init_gpu_args
if [[ -n "$DFM_EVALS_SUITES_FILE" ]]; then
  DFM_EVALS_SUITES_FILE="$(readlink -f "$DFM_EVALS_SUITES_FILE" 2>/dev/null || true)"
  [[ -n "$DFM_EVALS_SUITES_FILE" ]] || die "unable to resolve suites file path"
  [[ -f "$DFM_EVALS_SUITES_FILE" ]] || die "suites file not found: $DFM_EVALS_SUITES_FILE"
fi

SING_BIND_ARGS=(
  -B "$BASE_DIR:$LAIFS_APPL_DIR"
  -B "$OVERLAY_DIR:/overlay"
  -B "$REPO_ROOT:$REPO_ROOT"
  -B "$HF_HOME_HOST:$HF_HOME_CONTAINER"
)
if [[ -f "$OVERLAY_DIR/overlay-runtime.binds" ]]; then
  while IFS= read -r bind_spec; do
    [[ -n "$bind_spec" ]] || continue
    SING_BIND_ARGS+=(-B "$bind_spec")
  done < "$OVERLAY_DIR/overlay-runtime.binds"
fi
if [[ -e "$POST_ARTIFACT_ROOT" ]]; then
  ARTIFACT_ROOT_TARGET="$(readlink -f "$POST_ARTIFACT_ROOT" || true)"
  if [[ -n "$ARTIFACT_ROOT_TARGET" && -d "$ARTIFACT_ROOT_TARGET" ]]; then
    SING_BIND_ARGS+=(-B "$ARTIFACT_ROOT_TARGET:$ARTIFACT_ROOT_TARGET")
  elif [[ -d "$POST_ARTIFACT_ROOT" ]]; then
    SING_BIND_ARGS+=(-B "$POST_ARTIFACT_ROOT:$POST_ARTIFACT_ROOT")
  fi
fi
if [[ "$USE_LORA_ADAPTER" == "1" && -d "$LORA_ADAPTER_PATH" ]]; then
  SING_BIND_ARGS+=(-B "$LORA_ADAPTER_PATH:$LORA_ADAPTER_PATH")
fi
if [[ "$JUDGE_USE_LORA_ADAPTER" == "1" && -d "$JUDGE_LORA_ADAPTER_PATH" ]]; then
  SING_BIND_ARGS+=(-B "$JUDGE_LORA_ADAPTER_PATH:$JUDGE_LORA_ADAPTER_PATH")
fi
if [[ -d "$MODEL" ]]; then
  MODEL_BIND_PATH="$(readlink -f "$MODEL" 2>/dev/null || true)"
  if [[ -n "$MODEL_BIND_PATH" && -d "$MODEL_BIND_PATH" && "$MODEL_BIND_PATH" != "$REPO_ROOT"* ]]; then
    SING_BIND_ARGS+=(-B "$MODEL_BIND_PATH:$MODEL_BIND_PATH")
  fi
fi
if [[ -d "$JUDGE_SERVER_MODEL" ]]; then
  JUDGE_MODEL_BIND_PATH="$(readlink -f "$JUDGE_SERVER_MODEL" 2>/dev/null || true)"
  if [[ -n "$JUDGE_MODEL_BIND_PATH" && -d "$JUDGE_MODEL_BIND_PATH" && "$JUDGE_MODEL_BIND_PATH" != "$REPO_ROOT"* ]]; then
    SING_BIND_ARGS+=(-B "$JUDGE_MODEL_BIND_PATH:$JUDGE_MODEL_BIND_PATH")
  fi
fi
if [[ -n "$DFM_EVALS_SUITES_FILE" ]]; then
  DFM_EVALS_SUITES_DIR="$(dirname "$DFM_EVALS_SUITES_FILE")"
  if [[ "$DFM_EVALS_SUITES_DIR" != "$REPO_ROOT"* ]]; then
    SING_BIND_ARGS+=(-B "$DFM_EVALS_SUITES_DIR:$DFM_EVALS_SUITES_DIR")
  fi
fi

mkdir -p \
  "$DFM_EVALS_RUN_DIR" \
  "$DFM_EVALS_LOG_DIR" \
  "$DFM_EVALS_SERVER_LOG_DIR"
TARGET_SERVER_LOG="$DFM_EVALS_SERVER_LOG_DIR/target-vllm-${SLURM_JOB_ID_SAFE}.log"
JUDGE_SERVER_LOG="$DFM_EVALS_SERVER_LOG_DIR/judge-vllm-${SLURM_JOB_ID_SAFE}.log"

if [[ "$DFM_EVALS_MODEL" == vllm/* || "$DFM_EVALS_TARGET_MODEL" == vllm/* || "$DFM_EVALS_JUDGE_MODEL" == vllm/* ]]; then
  start_vllm_server \
    target \
    "$TARGET_SERVER_MODEL_TAG" \
    "$TARGET_SERVER_SERVED_MODEL_NAME" \
    "$TARGET_PORT" \
    "$TARGET_SERVER_API_KEY" \
    "$TP" \
    "$PP" \
    "$DP" \
    "$CTX" \
    "$GPU_MEM" \
    "$TARGET_VISIBLE_DEVICES" \
    "$USE_LORA_ADAPTER" \
    "$LORA_MODULE_SPEC" \
    "$LORA_MAX_RANK" \
    "$LORA_CHAT_TEMPLATE_PATH" \
    "$LORA_TOKENIZER_PATH" \
    "$TARGET_ENABLE_AUTO_TOOL_CHOICE" \
    "$TARGET_TOOL_CALL_PARSER" \
    "$TARGET_REASONING_PARSER" \
    "$TARGET_TOKENIZER_MODE" \
    "$TARGET_CHAT_TEMPLATE_KWARGS_JSON" \
    "$TARGET_ENFORCE_EAGER" \
    "$TARGET_ATTENTION_BACKEND" \
    "$TARGET_TOOL_PARSER_PLUGIN" \
    "$TARGET_SERVER_LOG" \
    "$TARGET_SERVER_CACHE_ROOT"
fi

if [[ "$JUDGE_SERVER_ENABLED" == "1" ]]; then
  [[ -n "$JUDGE_VISIBLE_DEVICES" ]] || echo "WARNING: JUDGE_VISIBLE_DEVICES is empty; judge server will compete for GPUs with target server"

  start_vllm_server \
    judge \
    "$JUDGE_SERVER_MODEL_TAG" \
    "$JUDGE_SERVER_SERVED_MODEL_NAME" \
    "$JUDGE_PORT" \
    "$JUDGE_SERVER_API_KEY" \
    "$JUDGE_TP" \
    "$JUDGE_PP" \
    "$JUDGE_DP" \
    "$JUDGE_CTX" \
    "$JUDGE_GPU_MEM" \
    "$JUDGE_VISIBLE_DEVICES" \
    "$JUDGE_USE_LORA_ADAPTER" \
    "$JUDGE_LORA_MODULE_SPEC" \
    "$JUDGE_LORA_MAX_RANK" \
    "$JUDGE_LORA_CHAT_TEMPLATE_PATH" \
    "$JUDGE_LORA_TOKENIZER_PATH" \
    "$JUDGE_ENABLE_AUTO_TOOL_CHOICE" \
    "$JUDGE_TOOL_CALL_PARSER" \
    "$JUDGE_REASONING_PARSER" \
    "$JUDGE_TOKENIZER_MODE" \
    "$JUDGE_CHAT_TEMPLATE_KWARGS_JSON" \
    "$JUDGE_ENFORCE_EAGER" \
    "$JUDGE_ATTENTION_BACKEND" \
    "$JUDGE_TOOL_PARSER_PLUGIN" \
    "$JUDGE_SERVER_LOG" \
    "$JUDGE_SERVER_CACHE_ROOT"
fi

if [[ -n "$TARGET_VLLM_URL" ]]; then
  export VLLM_BASE_URL="$TARGET_VLLM_URL"
  export VLLM_API_KEY="$TARGET_SERVER_API_KEY"
fi

if [[ -n "$JUDGE_VLLM_URL" ]]; then
  DFM_EVALS_OPENAI_BASE_URL="$JUDGE_VLLM_URL"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-$JUDGE_SERVER_API_KEY}"
fi

if [[ -n "$DFM_EVALS_OPENAI_BASE_URL" ]]; then
  export OPENAI_BASE_URL="$DFM_EVALS_OPENAI_BASE_URL"
fi

if [[ "$DFM_EVALS_TARGET_MODEL" == openai/* || "$DFM_EVALS_JUDGE_MODEL" == openai/* || "$DFM_EVALS_MODEL" == openai/* ]]; then
  [[ -n "${OPENAI_API_KEY:-}" ]] || export OPENAI_API_KEY="inspectai"
  [[ -n "${OPENAI_BASE_URL:-}" ]] || die "OPENAI_BASE_URL (or DFM_EVALS_OPENAI_BASE_URL) is required for openai/* models"
fi

EVALS_CMD=""
build_evals_cli EVALS_CMD
DFM_EVALS_MODEL_INFO_OVERRIDES=""
build_model_info_overrides DFM_EVALS_MODEL_INFO_OVERRIDES

printf -v DFM_EVALS_CACHE_DIR_Q '%q' "$DFM_EVALS_CACHE_DIR"
printf -v DFM_EVALS_RUN_DIR_Q '%q' "$DFM_EVALS_RUN_DIR"
printf -v DFM_EVALS_LOG_DIR_Q '%q' "$DFM_EVALS_LOG_DIR"
printf -v DFM_EVALS_XDG_CACHE_HOME_Q '%q' "$DFM_EVALS_XDG_CACHE_HOME"
printf -v DFM_EVALS_XDG_DATA_HOME_Q '%q' "$DFM_EVALS_XDG_DATA_HOME"
printf -v DFM_EVALS_TRITON_CACHE_DIR_Q '%q' "$DFM_EVALS_TRITON_CACHE_DIR"
printf -v DFM_EVALS_TORCHINDUCTOR_CACHE_DIR_Q '%q' "$DFM_EVALS_TORCHINDUCTOR_CACHE_DIR"
printf -v DFM_EVALS_RULER_QA_CACHE_Q '%q' "$DFM_EVALS_RULER_QA_CACHE"
printf -v DFM_EVALS_RULER_QA_OFFLINE_Q '%q' "$DFM_EVALS_RULER_QA_OFFLINE"
printf -v DFM_EVALS_MODAL_ENABLE_OUTPUT_Q '%q' "$DFM_EVALS_MODAL_ENABLE_OUTPUT"
printf -v DFM_EVALS_MODAL_SANDBOX_TIMEOUT_Q '%q' "$DFM_EVALS_MODAL_SANDBOX_TIMEOUT"
printf -v DFM_EVALS_MODAL_SANDBOX_IDLE_TIMEOUT_Q '%q' "$DFM_EVALS_MODAL_SANDBOX_IDLE_TIMEOUT"
printf -v TARGET_VLLM_URL_Q '%q' "${TARGET_VLLM_URL:-}"
printf -v TARGET_SERVER_API_KEY_Q '%q' "${TARGET_SERVER_API_KEY:-}"
printf -v OPENAI_BASE_URL_Q '%q' "${OPENAI_BASE_URL:-}"
printf -v OPENAI_API_KEY_Q '%q' "${OPENAI_API_KEY:-}"
MODEL_INFO_EXPORT_CMD=""
if [[ -n "$DFM_EVALS_MODEL_INFO_OVERRIDES" && "$DFM_EVALS_MODEL_INFO_OVERRIDES" != "{}" ]]; then
  printf -v DFM_EVALS_MODEL_INFO_OVERRIDES_Q '%q' "$DFM_EVALS_MODEL_INFO_OVERRIDES"
  MODEL_INFO_EXPORT_CMD="export DFM_EVALS_MODEL_INFO_OVERRIDES=${DFM_EVALS_MODEL_INFO_OVERRIDES_Q}"
fi

RUN_CMD="$(render_env)
export XDG_CACHE_HOME=${DFM_EVALS_XDG_CACHE_HOME_Q}
export XDG_DATA_HOME=${DFM_EVALS_XDG_DATA_HOME_Q}
export TRITON_CACHE_DIR=${DFM_EVALS_TRITON_CACHE_DIR_Q}
export TORCHINDUCTOR_CACHE_DIR=${DFM_EVALS_TORCHINDUCTOR_CACHE_DIR_Q}
export DFM_EVALS_RULER_QA_CACHE=${DFM_EVALS_RULER_QA_CACHE_Q}
export DFM_EVALS_RULER_QA_OFFLINE=${DFM_EVALS_RULER_QA_OFFLINE_Q}
export DFM_EVALS_MODAL_ENABLE_OUTPUT=${DFM_EVALS_MODAL_ENABLE_OUTPUT_Q}
export DFM_EVALS_MODAL_SANDBOX_TIMEOUT=${DFM_EVALS_MODAL_SANDBOX_TIMEOUT_Q}
export DFM_EVALS_MODAL_SANDBOX_IDLE_TIMEOUT=${DFM_EVALS_MODAL_SANDBOX_IDLE_TIMEOUT_Q}
mkdir -p ${DFM_EVALS_CACHE_DIR_Q}
mkdir -p ${DFM_EVALS_XDG_CACHE_HOME_Q}
mkdir -p ${DFM_EVALS_XDG_DATA_HOME_Q}
mkdir -p ${DFM_EVALS_TRITON_CACHE_DIR_Q}
mkdir -p ${DFM_EVALS_TORCHINDUCTOR_CACHE_DIR_Q}
mkdir -p ${DFM_EVALS_RULER_QA_CACHE_Q}
mkdir -p ${DFM_EVALS_RUN_DIR_Q}
mkdir -p ${DFM_EVALS_LOG_DIR_Q}
if [[ -n ${TARGET_VLLM_URL_Q} ]]; then
  export VLLM_BASE_URL=${TARGET_VLLM_URL_Q}
  export VLLM_API_KEY=${TARGET_SERVER_API_KEY_Q}
fi
if [[ -n ${OPENAI_BASE_URL_Q} ]]; then
  export OPENAI_BASE_URL=${OPENAI_BASE_URL_Q}
  export OPENAI_API_KEY=${OPENAI_API_KEY_Q}
fi
${MODEL_INFO_EXPORT_CMD}
cd ${DFM_EVALS_RUN_DIR_Q}
python - <<'PY'
import inspect_ai
import sys
import yaml

import dfm_evals.cli

print('dfm_evals_python\t' + sys.executable)
print('dfm_evals_prefix\t' + sys.prefix)
print('dfm_evals_yaml\t' + yaml.__file__)
print('dfm_evals_inspect_ai\t' + inspect_ai.__file__)
print('dfm_evals_cli\t' + dfm_evals.cli.__file__)
PY
${EVALS_CMD}"

echo "SIF: $SIF"
echo "Overlay: $OVERLAY_DIR"
echo "Singularity GPU mode: $SINGULARITY_GPU_MODE"
echo "Model: $MODEL"
echo "Eval model: $DFM_EVALS_MODEL"
echo "Eval target model: $DFM_EVALS_TARGET_MODEL"
echo "Eval judge model: $DFM_EVALS_JUDGE_MODEL"
echo "Judge model auto-derived: $JUDGE_MODEL_AUTO"
echo "Suite: $DFM_EVALS_SUITE"
echo "Run label: $DFM_EVALS_RUN_LABEL"
echo "Suites file: ${DFM_EVALS_SUITES_FILE:-<packaged default>}"
echo "Limit: $DFM_EVALS_LIMIT"
echo "Temperature: $DFM_EVALS_TEMPERATURE"
echo "MAX_CONNECTIONS: $MAX_CONNECTIONS"
echo "MAX_TASKS: $MAX_TASKS"
echo "Client timeout: $DFM_EVALS_CLIENT_TIMEOUT"
echo "Client max retries: $DFM_EVALS_CLIENT_MAX_RETRIES"
echo "Target server model tag: $TARGET_SERVER_MODEL_TAG"
echo "Target served model name: $TARGET_SERVER_SERVED_MODEL_NAME"
echo "Target URL: ${TARGET_VLLM_URL:-<none>}"
echo "Target TP/PP/DP: $TP/$PP/$DP"
echo "Target CTX: ${CTX:-<auto>}"
echo "Target GPU_MEM: $GPU_MEM"
echo "Target visible devices: ${TARGET_VISIBLE_DEVICES:-<all>}"
echo "Target auto tool choice: $TARGET_ENABLE_AUTO_TOOL_CHOICE"
echo "Target tool call parser: ${TARGET_TOOL_CALL_PARSER:-<none>}"
echo "Target tool parser plugin: ${TARGET_TOOL_PARSER_PLUGIN:-<none>}"
echo "Target reasoning parser: ${TARGET_REASONING_PARSER:-<none>}"
echo "Target tokenizer mode: ${TARGET_TOKENIZER_MODE:-<auto>}"
echo "Target chat template kwargs: ${TARGET_CHAT_TEMPLATE_KWARGS_JSON:-<none>}"
echo "Target enforce eager: $TARGET_ENFORCE_EAGER"
echo "Target attention backend: ${TARGET_ATTENTION_BACKEND:-<auto>}"
echo "Judge server enabled: $JUDGE_SERVER_ENABLED"
if [[ "$JUDGE_SERVER_ENABLED" == "1" ]]; then
  echo "Judge server model: $JUDGE_SERVER_MODEL"
  echo "Judge served model name: $JUDGE_SERVER_SERVED_MODEL_NAME"
  echo "Judge URL: ${JUDGE_VLLM_URL:-<starting>}"
  echo "Judge TP/PP/DP: $JUDGE_TP/$JUDGE_PP/$JUDGE_DP"
  echo "Judge CTX: ${JUDGE_CTX:-<auto>}"
  echo "Judge GPU_MEM: $JUDGE_GPU_MEM"
  echo "Judge visible devices: ${JUDGE_VISIBLE_DEVICES:-<all>}"
  echo "Judge auto tool choice: $JUDGE_ENABLE_AUTO_TOOL_CHOICE"
  echo "Judge tool call parser: ${JUDGE_TOOL_CALL_PARSER:-<none>}"
  echo "Judge tool parser plugin: ${JUDGE_TOOL_PARSER_PLUGIN:-<none>}"
  echo "Judge chat template kwargs: ${JUDGE_CHAT_TEMPLATE_KWARGS_JSON:-<none>}"
  echo "Judge enforce eager: $JUDGE_ENFORCE_EAGER"
  echo "Judge attention backend: ${JUDGE_ATTENTION_BACKEND:-<auto>}"
fi
echo "Target LoRA enabled: $USE_LORA_ADAPTER"
if [[ "$USE_LORA_ADAPTER" == "1" ]]; then
  echo "Target LoRA adapter path: $LORA_ADAPTER_PATH"
  echo "Target LoRA base model: $LORA_BASE_MODEL"
  echo "Target LoRA module id: $LORA_MODULE_NAME"
  echo "Target LoRA max rank: $LORA_MAX_RANK"
  echo "Target LoRA chat template: ${LORA_CHAT_TEMPLATE_PATH:-<none>}"
  echo "Target LoRA tokenizer path: ${LORA_TOKENIZER_PATH:-<none>}"
fi
echo "Judge LoRA enabled: $JUDGE_USE_LORA_ADAPTER"
if [[ "$JUDGE_USE_LORA_ADAPTER" == "1" ]]; then
  echo "Judge LoRA adapter path: $JUDGE_LORA_ADAPTER_PATH"
  echo "Judge LoRA base model: $JUDGE_LORA_BASE_MODEL"
  echo "Judge LoRA module id: $JUDGE_LORA_MODULE_NAME"
  echo "Judge LoRA max rank: $JUDGE_LORA_MAX_RANK"
  echo "Judge LoRA chat template: ${JUDGE_LORA_CHAT_TEMPLATE_PATH:-<none>}"
  echo "Judge LoRA tokenizer path: ${JUDGE_LORA_TOKENIZER_PATH:-<none>}"
fi
echo "HF_HOME host: $HF_HOME_HOST"
echo "HF_HOME container: $HF_HOME_CONTAINER"
echo "XDG cache home: $DFM_EVALS_XDG_CACHE_HOME"
echo "XDG data home: $DFM_EVALS_XDG_DATA_HOME"
echo "Triton cache dir: $DFM_EVALS_TRITON_CACHE_DIR"
echo "TorchInductor cache dir: $DFM_EVALS_TORCHINDUCTOR_CACHE_DIR"
echo "RULER QA cache: $DFM_EVALS_RULER_QA_CACHE"
echo "RULER QA offline: $DFM_EVALS_RULER_QA_OFFLINE"
echo "Run dir: $DFM_EVALS_RUN_DIR"
echo "Log dir: $DFM_EVALS_LOG_DIR"
echo "vLLM server log dir: $DFM_EVALS_SERVER_LOG_DIR"
echo "OpenAI base URL: ${OPENAI_BASE_URL:-<unset>}"
echo "Target server log: $TARGET_SERVER_LOG"
if [[ -n "$DFM_EVALS_MODEL_INFO_OVERRIDES" && "$DFM_EVALS_MODEL_INFO_OVERRIDES" != "{}" ]]; then
  echo "Model info overrides: $DFM_EVALS_MODEL_INFO_OVERRIDES"
fi
if [[ "$JUDGE_SERVER_ENABLED" == "1" ]]; then
  echo "Judge server log: $JUDGE_SERVER_LOG"
fi
echo "Export EEE: $DFM_EVALS_EXPORT_EEE"
if [[ "$DFM_EVALS_EXPORT_EEE" == "1" ]]; then
  echo "EEE output dir: $DFM_EVALS_EEE_OUTPUT_DIR"
echo "EEE source org: $DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME"
echo "EEE evaluator relationship: $DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP"
fi
echo "Modal enable output: $DFM_EVALS_MODAL_ENABLE_OUTPUT"
echo "Modal sandbox timeout: ${DFM_EVALS_MODAL_SANDBOX_TIMEOUT:-<default>}"
echo "Modal sandbox idle timeout: ${DFM_EVALS_MODAL_SANDBOX_IDLE_TIMEOUT:-<default>}"
echo "Runner idle timeout: ${DFM_EVALS_RUNNER_IDLE_TIMEOUT:-<disabled>}"
echo "Runner poll interval: ${DFM_EVALS_RUNNER_POLL_INTERVAL}s"
echo "Runner kill grace: ${DFM_EVALS_RUNNER_KILL_GRACE}s"
echo "Salvage partial results: $DFM_EVALS_SALVAGE_PARTIAL_RESULTS"

echo "Running dfm-evals suite with externally managed vLLM backend(s)..."
run_eval_with_watchdog
salvage_partial_results
if [[ "$EVAL_RUN_EXIT_CODE" -ne 0 ]]; then
  if [[ "$EVAL_RUN_STALLED" == "1" ]]; then
    die "dfm-evals suite stalled and was terminated after ${DFM_EVALS_RUNNER_IDLE_TIMEOUT}s without progress"
  fi
  exit "$EVAL_RUN_EXIT_CODE"
fi
echo "dfm-evals suite complete."

if [[ "$DFM_EVALS_EXPORT_EEE" == "1" ]]; then
  printf -v DFM_EVALS_EEE_OUTPUT_DIR_Q '%q' "$DFM_EVALS_EEE_OUTPUT_DIR"
  printf -v DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME_Q '%q' "$DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME"
  printf -v DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP_Q '%q' "$DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP"
  DFM_EVALS_EEE_INFERENCE_ARGS=""
  if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
    printf -v DFM_EVALS_EEE_INFERENCE_ARGS ' --inference-base-url %q' "$OPENAI_BASE_URL"
  fi
  if [[ -n "$DFM_EVALS_EEE_MODEL_ID" ]]; then
    printf -v DFM_EVALS_EEE_MODEL_ID_ARG ' --model-id-override %q' "$DFM_EVALS_EEE_MODEL_ID"
    DFM_EVALS_EEE_INFERENCE_ARGS+="$DFM_EVALS_EEE_MODEL_ID_ARG"
  fi

  EXPORT_EEE_CMD="$(render_env)
export XDG_CACHE_HOME=${DFM_EVALS_XDG_CACHE_HOME_Q}
export XDG_DATA_HOME=${DFM_EVALS_XDG_DATA_HOME_Q}
export TRITON_CACHE_DIR=${DFM_EVALS_TRITON_CACHE_DIR_Q}
export TORCHINDUCTOR_CACHE_DIR=${DFM_EVALS_TORCHINDUCTOR_CACHE_DIR_Q}
mkdir -p ${DFM_EVALS_XDG_CACHE_HOME_Q}
mkdir -p ${DFM_EVALS_XDG_DATA_HOME_Q}
mkdir -p ${DFM_EVALS_TRITON_CACHE_DIR_Q}
mkdir -p ${DFM_EVALS_TORCHINDUCTOR_CACHE_DIR_Q}
mkdir -p ${DFM_EVALS_EEE_OUTPUT_DIR_Q}
python - <<'PY'
import inspect_ai
import sys
import yaml

import dfm_evals.cli

print('dfm_evals_python\t' + sys.executable)
print('dfm_evals_prefix\t' + sys.prefix)
print('dfm_evals_yaml\t' + yaml.__file__)
print('dfm_evals_inspect_ai\t' + inspect_ai.__file__)
print('dfm_evals_cli\t' + dfm_evals.cli.__file__)
PY
python -m dfm_evals.cli eee inspect \
  --log-path ${DFM_EVALS_LOG_DIR_Q} \
  --output-dir ${DFM_EVALS_EEE_OUTPUT_DIR_Q} \
  --source-organization-name ${DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME_Q} \
  --evaluator-relationship ${DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP_Q}${DFM_EVALS_EEE_INFERENCE_ARGS}"

  echo "Exporting Inspect logs to every_eval_ever format..."
  singularity exec "${SINGULARITY_GPU_ARGS[@]}" "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$EXPORT_EEE_CMD"
  echo "EEE export complete: $DFM_EVALS_EEE_OUTPUT_DIR"
else
  echo "DFM_EVALS_EXPORT_EEE=0, skipping EEE export."
fi
