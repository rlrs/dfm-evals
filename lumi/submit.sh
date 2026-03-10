#!/bin/bash
# Friendly wrapper for submitting dfm-evals jobs without manual env var juggling.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_SUBMIT_SCRIPT="$SCRIPT_DIR/run_suite.sbatch"
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
DFM_EVALS_RUN_ROOT=${DFM_EVALS_RUN_ROOT:-$REPO_ROOT/logs/evals-runs}
DFM_EVALS_LOG_ROOT=${DFM_EVALS_LOG_ROOT:-$REPO_ROOT/logs/evals-logs}

MODEL=${MODEL:-google/gemma-3-4b-it}
EVAL_MODEL=${EVAL_MODEL:-}
TARGET_MODEL=${TARGET_MODEL:-}
JUDGE_MODEL=${JUDGE_MODEL:-}
JUDGE_MODEL_SET=0
OPENAI_BASE_URL_OVERRIDE=${OPENAI_BASE_URL_OVERRIDE:-}
SUITE=${SUITE:-fundamentals}
LIMIT=${LIMIT:-100}
TP=${TP:-1}
PP=${PP:-1}
DP=${DP:-8}
CTX=${CTX:-4096}
GPU_MEM=${GPU_MEM:-0.92}
TARGET_PORT=${TARGET_PORT:-8000}
TARGET_VISIBLE_DEVICES=${TARGET_VISIBLE_DEVICES:-}
MAX_CONNECTIONS=${MAX_CONNECTIONS:-128}
TARGET_ENABLE_AUTO_TOOL_CHOICE=${TARGET_ENABLE_AUTO_TOOL_CHOICE:-1}
TARGET_TOOL_CALL_PARSER=${TARGET_TOOL_CALL_PARSER:-hermes}
TARGET_CHAT_TEMPLATE_KWARGS_JSON=${TARGET_CHAT_TEMPLATE_KWARGS_JSON:-}
TARGET_ENFORCE_EAGER=${TARGET_ENFORCE_EAGER:-0}
JUDGE_SERVER_ENABLED=${JUDGE_SERVER_ENABLED:-0}
JUDGE_SERVER_MODEL=${JUDGE_SERVER_MODEL:-}
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
JUDGE_CHAT_TEMPLATE_KWARGS_JSON=${JUDGE_CHAT_TEMPLATE_KWARGS_JSON:-}
JUDGE_ENFORCE_EAGER=${JUDGE_ENFORCE_EAGER:-0}
JUDGE_CTX_SET=0
DFM_EVALS_MODAL_ENABLE_OUTPUT=${DFM_EVALS_MODAL_ENABLE_OUTPUT:-0}
RUN_LABEL=${RUN_LABEL:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
DFM_EVALS_EEE_OUTPUT_DIR=${DFM_EVALS_EEE_OUTPUT_DIR:-}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-$REPO_ROOT/logs/slurm}
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./lumi/submit.sh [options]

Options:
  --model <model>            Base model id or local LoRA adapter dir (default: google/gemma-3-4b-it)
  --eval-model <model>       Eval runtime model, default: vllm/<model>
  --target-model <model>     Suite target model, default: <eval-model>
  --judge-model <model>      Suite judge model (accepts plain name like qwen-235b; default: openai/<target-model-name>)
  --openai-base-url <url>    Override OPENAI_BASE_URL inside job/container
  --suite <name>             Eval suite (default: fundamentals)
  --limit <n|none>           Sample limit (default: 100, use 'none' to omit)
  --tp <n>                   Target server tensor parallel size (default: 1)
  --pp <n>                   Target server pipeline parallel size (default: 1)
  --dp <n>                   Target server data parallel size (default: 8)
  --ctx <n>                  Target server max model len (default: 4096)
  --gpu-mem <f>              Target server GPU memory utilization (default: 0.92)
  --target-port <n>          Target vLLM server port (default: 8000)
  --target-devices <csv>     Target HIP/CUDA visible devices (default: all)
  --judge-server             Launch separate judge vLLM server
  --no-judge-server          Do not launch separate judge vLLM server (default)
  --judge-server-model <m>   Judge server model path/id (default: --model)
  --judge-served-name <id>   Judge server served model name override
  --judge-port <n>           Judge vLLM server port (default: 8001)
  --judge-tp <n>             Judge server tensor parallel size (default: 1)
  --judge-pp <n>             Judge server pipeline parallel size (default: 1)
  --judge-dp <n>             Judge server data parallel size (default: 1)
  --judge-ctx <n>            Judge server max model len (default: 4096)
  --judge-gpu-mem <f>        Judge server GPU memory utilization (default: 0.85)
  --judge-devices <csv>      Judge HIP/CUDA visible devices (default: all)
  --max-connections <n>      Concurrency for inspect eval (default: 128)
  --target-enable-auto-tool-choice   Enable target vLLM --enable-auto-tool-choice (default)
  --target-disable-auto-tool-choice  Disable target vLLM --enable-auto-tool-choice
  --target-tool-call-parser <name>   Target vLLM --tool-call-parser (default: hermes; use 'none' to unset)
  --target-chat-template-kwargs-json <json>  Target vLLM --default-chat-template-kwargs JSON
  --target-enforce-eager             Enable target vLLM --enforce-eager
  --target-disable-enforce-eager     Disable target vLLM --enforce-eager (default)
  --judge-enable-auto-tool-choice    Enable judge vLLM --enable-auto-tool-choice
  --judge-disable-auto-tool-choice   Disable judge vLLM --enable-auto-tool-choice (default)
  --judge-tool-call-parser <name>    Judge vLLM --tool-call-parser (default: unset; use 'none' to unset)
  --judge-chat-template-kwargs-json <json>   Judge vLLM --default-chat-template-kwargs JSON
  --judge-enforce-eager              Enable judge vLLM --enforce-eager
  --judge-disable-enforce-eager      Disable judge vLLM --enforce-eager (default)
  --modal-enable-output      Wrap eval execution in `modal.enable_output()` for Modal logs
  --modal-disable-output     Disable Modal SDK output (default)
  --run-label <label>        Optional DFM_EVALS_RUN_LABEL override (default: <suite>__<model-slug>__job-<jobid>)
  --extra-args <string>      Extra args appended to evals CLI
  --eee-output-dir <path>    EEE root data dir override (default: ./logs/every_eval_ever/data)
  --slurm-log-dir <path>     Slurm stdout/err directory (default: ./logs/slurm)
  --script <path>            sbatch script to submit
  --dry-run                  Print sbatch command/env and exit
  --help                     Show help

Examples:
  ./lumi/submit.sh
  ./lumi/submit.sh --limit 100 --max-connections 128
  ./lumi/submit.sh --tp 1 --dp 8 --target-port 8000 --max-connections 128
  ./lumi/submit.sh --judge-model openai/google/gemma-3-4b-pt --judge-server --judge-devices 6,7 --judge-dp 2
  ./lumi/submit.sh --target-tool-call-parser qwen3
  ./lumi/submit.sh --model ../../post/outputs/sft-16292768/final --limit 100
  ./lumi/submit.sh --target-model vllm/google/gemma-3-4b-it --judge-model vllm/google/gemma-3-4b-it
  ./lumi/submit.sh --eee-output-dir /path/to/every_eval_ever/data
  ./lumi/submit.sh --slurm-log-dir /path/to/slurm-logs
  ./lumi/submit.sh --run-label fundamentals_gemma64c --dry-run
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      need_value "$1" "$#"
      MODEL="$2"
      shift 2
      ;;
    --eval-model)
      need_value "$1" "$#"
      EVAL_MODEL="$2"
      shift 2
      ;;
    --target-model)
      need_value "$1" "$#"
      TARGET_MODEL="$2"
      shift 2
      ;;
    --judge-model)
      need_value "$1" "$#"
      JUDGE_MODEL="$2"
      JUDGE_MODEL_SET=1
      shift 2
      ;;
    --openai-base-url)
      need_value "$1" "$#"
      OPENAI_BASE_URL_OVERRIDE="$2"
      shift 2
      ;;
    --suite)
      need_value "$1" "$#"
      SUITE="$2"
      shift 2
      ;;
    --limit)
      need_value "$1" "$#"
      case "$2" in
        none|off|all)
          LIMIT=""
          ;;
        *)
          LIMIT="$2"
          ;;
      esac
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
    --dp)
      need_value "$1" "$#"
      DP="$2"
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
    --target-port)
      need_value "$1" "$#"
      TARGET_PORT="$2"
      shift 2
      ;;
    --target-devices)
      need_value "$1" "$#"
      TARGET_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --judge-server)
      JUDGE_SERVER_ENABLED=1
      shift
      ;;
    --no-judge-server)
      JUDGE_SERVER_ENABLED=0
      shift
      ;;
    --judge-server-model)
      need_value "$1" "$#"
      JUDGE_SERVER_MODEL="$2"
      shift 2
      ;;
    --judge-served-name)
      need_value "$1" "$#"
      JUDGE_SERVER_SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --judge-port)
      need_value "$1" "$#"
      JUDGE_PORT="$2"
      shift 2
      ;;
    --judge-tp)
      need_value "$1" "$#"
      JUDGE_TP="$2"
      shift 2
      ;;
    --judge-pp)
      need_value "$1" "$#"
      JUDGE_PP="$2"
      shift 2
      ;;
    --judge-dp)
      need_value "$1" "$#"
      JUDGE_DP="$2"
      shift 2
      ;;
    --judge-ctx)
      need_value "$1" "$#"
      JUDGE_CTX="$2"
      JUDGE_CTX_SET=1
      shift 2
      ;;
    --judge-gpu-mem)
      need_value "$1" "$#"
      JUDGE_GPU_MEM="$2"
      shift 2
      ;;
    --judge-devices)
      need_value "$1" "$#"
      JUDGE_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --max-connections)
      need_value "$1" "$#"
      MAX_CONNECTIONS="$2"
      shift 2
      ;;
    --target-enable-auto-tool-choice)
      TARGET_ENABLE_AUTO_TOOL_CHOICE=1
      shift
      ;;
    --target-disable-auto-tool-choice)
      TARGET_ENABLE_AUTO_TOOL_CHOICE=0
      shift
      ;;
    --target-tool-call-parser)
      need_value "$1" "$#"
      TARGET_TOOL_CALL_PARSER="$2"
      shift 2
      ;;
    --target-chat-template-kwargs-json)
      need_value "$1" "$#"
      TARGET_CHAT_TEMPLATE_KWARGS_JSON="$2"
      shift 2
      ;;
    --target-enforce-eager)
      TARGET_ENFORCE_EAGER=1
      shift
      ;;
    --target-disable-enforce-eager)
      TARGET_ENFORCE_EAGER=0
      shift
      ;;
    --judge-enable-auto-tool-choice)
      JUDGE_ENABLE_AUTO_TOOL_CHOICE=1
      shift
      ;;
    --judge-disable-auto-tool-choice)
      JUDGE_ENABLE_AUTO_TOOL_CHOICE=0
      shift
      ;;
    --judge-tool-call-parser)
      need_value "$1" "$#"
      JUDGE_TOOL_CALL_PARSER="$2"
      shift 2
      ;;
    --judge-chat-template-kwargs-json)
      need_value "$1" "$#"
      JUDGE_CHAT_TEMPLATE_KWARGS_JSON="$2"
      shift 2
      ;;
    --judge-enforce-eager)
      JUDGE_ENFORCE_EAGER=1
      shift
      ;;
    --judge-disable-enforce-eager)
      JUDGE_ENFORCE_EAGER=0
      shift
      ;;
    --modal-enable-output)
      DFM_EVALS_MODAL_ENABLE_OUTPUT=1
      shift
      ;;
    --modal-disable-output)
      DFM_EVALS_MODAL_ENABLE_OUTPUT=0
      shift
      ;;
    --run-label)
      need_value "$1" "$#"
      RUN_LABEL="$2"
      shift 2
      ;;
    --extra-args)
      need_value "$1" "$#"
      EXTRA_ARGS="$2"
      shift 2
      ;;
    --eee-output-dir)
      need_value "$1" "$#"
      DFM_EVALS_EEE_OUTPUT_DIR="$2"
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

# If judge model is preconfigured via environment/.env, treat it as explicit
# unless the CLI later overrides it.
if [[ "$JUDGE_MODEL_SET" == "0" && -n "$JUDGE_MODEL" ]]; then
  JUDGE_MODEL_SET=1
fi

if [[ -z "$EVAL_MODEL" ]]; then
  EVAL_MODEL="vllm/$MODEL"
fi
if [[ -z "$TARGET_MODEL" ]]; then
  TARGET_MODEL="$EVAL_MODEL"
fi
if [[ -z "$JUDGE_MODEL" ]]; then
  if [[ "$TARGET_MODEL" == openai/* ]]; then
    JUDGE_MODEL="$TARGET_MODEL"
  elif [[ "$TARGET_MODEL" == vllm/* ]]; then
    JUDGE_MODEL="openai/${TARGET_MODEL#vllm/}"
  else
    JUDGE_MODEL="openai/$TARGET_MODEL"
  fi
fi
# Inspect model handles require <api>/<model>. Allow plain judge names
# from CLI (e.g., qwen-235b) and normalize to openai/<name>.
if [[ "$JUDGE_MODEL" != */* ]]; then
  JUDGE_MODEL="openai/$JUDGE_MODEL"
fi
if [[ -z "$OPENAI_BASE_URL_OVERRIDE" && -n "${OPENAI_BASE_URL:-}" ]]; then
  OPENAI_BASE_URL_OVERRIDE="$OPENAI_BASE_URL"
fi
if [[ "$JUDGE_CTX_SET" != "1" ]]; then
  JUDGE_CTX="$CTX"
fi
if [[ -z "$JUDGE_SERVER_MODEL" ]]; then
  JUDGE_SERVER_MODEL="$MODEL"
fi
if [[ "$TARGET_TOOL_CALL_PARSER" == "none" ]]; then
  TARGET_TOOL_CALL_PARSER=""
fi
if [[ "$JUDGE_TOOL_CALL_PARSER" == "none" ]]; then
  JUDGE_TOOL_CALL_PARSER=""
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
  die "target auto tool choice enabled but target tool call parser is empty; set --target-tool-call-parser <name> or disable auto tool choice"
fi
if [[ "$JUDGE_ENABLE_AUTO_TOOL_CHOICE" == "1" && -z "$JUDGE_TOOL_CALL_PARSER" ]]; then
  die "judge auto tool choice enabled but judge tool call parser is empty; set --judge-tool-call-parser <name> or disable auto tool choice"
fi

[[ -f "$SUBMIT_SCRIPT" ]] || die "submit script not found: $SUBMIT_SCRIPT"
[[ -d "$OVERLAY_DIR" ]] || die "overlay dir not found: $OVERLAY_DIR"
mkdir -p "$SLURM_LOG_DIR"

label_model_ref="$TARGET_MODEL"
if [[ -d "$MODEL" && -f "$MODEL/adapter_config.json" && -f "$MODEL/adapter_model.safetensors" ]]; then
  label_model_ref="$MODEL"
fi
default_model_label="$(model_label_from_ref "$label_model_ref")"
default_run_label_base="${SUITE}__${default_model_label}"

if [[ -n "$RUN_LABEL" ]]; then
  raw_slurm_log_label="$RUN_LABEL"
else
  raw_slurm_log_label="$default_run_label_base"
fi
slurm_log_label="${raw_slurm_log_label//[^[:alnum:]._-]/_}"
slurm_out_path="${SLURM_LOG_DIR}/${slurm_log_label}-%j.out"
slurm_err_path="${SLURM_LOG_DIR}/${slurm_log_label}-%j.err"

env_kv=(
  "DFM_EVALS_REPO_ROOT=$REPO_ROOT"
  "DFM_EVALS_RUN_ROOT=$DFM_EVALS_RUN_ROOT"
  "DFM_EVALS_LOG_ROOT=$DFM_EVALS_LOG_ROOT"
  "MODEL=$MODEL"
  "DFM_EVALS_MODEL=$EVAL_MODEL"
  "DFM_EVALS_TARGET_MODEL=$TARGET_MODEL"
  "DFM_EVALS_JUDGE_MODEL=$JUDGE_MODEL"
  "JUDGE_MODEL_AUTO=$([[ \"$JUDGE_MODEL_SET\" == \"0\" ]] && echo 1 || echo 0)"
  "DFM_EVALS_SUITE=$SUITE"
  "DFM_EVALS_LIMIT=$LIMIT"
  "DFM_EVALS_MODAL_ENABLE_OUTPUT=$DFM_EVALS_MODAL_ENABLE_OUTPUT"
  "TP=$TP"
  "PP=$PP"
  "DP=$DP"
  "CTX=$CTX"
  "GPU_MEM=$GPU_MEM"
  "TARGET_PORT=$TARGET_PORT"
  "MAX_CONNECTIONS=$MAX_CONNECTIONS"
  "TARGET_ENABLE_AUTO_TOOL_CHOICE=$TARGET_ENABLE_AUTO_TOOL_CHOICE"
  "TARGET_TOOL_CALL_PARSER=$TARGET_TOOL_CALL_PARSER"
  "TARGET_CHAT_TEMPLATE_KWARGS_JSON=$TARGET_CHAT_TEMPLATE_KWARGS_JSON"
  "TARGET_ENFORCE_EAGER=$TARGET_ENFORCE_EAGER"
  "JUDGE_SERVER_ENABLED=$JUDGE_SERVER_ENABLED"
  "JUDGE_SERVER_MODEL=$JUDGE_SERVER_MODEL"
  "JUDGE_PORT=$JUDGE_PORT"
  "JUDGE_TP=$JUDGE_TP"
  "JUDGE_PP=$JUDGE_PP"
  "JUDGE_DP=$JUDGE_DP"
  "JUDGE_CTX=$JUDGE_CTX"
  "JUDGE_GPU_MEM=$JUDGE_GPU_MEM"
  "JUDGE_ENABLE_AUTO_TOOL_CHOICE=$JUDGE_ENABLE_AUTO_TOOL_CHOICE"
  "JUDGE_TOOL_CALL_PARSER=$JUDGE_TOOL_CALL_PARSER"
  "JUDGE_CHAT_TEMPLATE_KWARGS_JSON=$JUDGE_CHAT_TEMPLATE_KWARGS_JSON"
  "JUDGE_ENFORCE_EAGER=$JUDGE_ENFORCE_EAGER"
)
if [[ -n "$RUN_LABEL" ]]; then
  env_kv+=("DFM_EVALS_RUN_LABEL=$RUN_LABEL")
fi
if [[ -n "$OPENAI_BASE_URL_OVERRIDE" ]]; then
  env_kv+=("DFM_EVALS_OPENAI_BASE_URL=$OPENAI_BASE_URL_OVERRIDE")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  env_kv+=("DFM_EVALS_EXTRA_ARGS=$EXTRA_ARGS")
fi
if [[ -n "$DFM_EVALS_EEE_OUTPUT_DIR" ]]; then
  env_kv+=("DFM_EVALS_EEE_OUTPUT_DIR=$DFM_EVALS_EEE_OUTPUT_DIR")
fi
if [[ -n "$TARGET_VISIBLE_DEVICES" ]]; then
  env_kv+=("TARGET_VISIBLE_DEVICES=$TARGET_VISIBLE_DEVICES")
fi
if [[ -n "$JUDGE_SERVER_SERVED_MODEL_NAME" ]]; then
  env_kv+=("JUDGE_SERVER_SERVED_MODEL_NAME=$JUDGE_SERVER_SERVED_MODEL_NAME")
fi
if [[ -n "$JUDGE_VISIBLE_DEVICES" ]]; then
  env_kv+=("JUDGE_VISIBLE_DEVICES=$JUDGE_VISIBLE_DEVICES")
fi

echo "Submit script: $SUBMIT_SCRIPT"
echo "Model: $MODEL"
echo "Eval model: $EVAL_MODEL"
echo "Target model: $TARGET_MODEL"
echo "Judge model: $JUDGE_MODEL"
if [[ "$JUDGE_MODEL_SET" == "0" ]]; then
  echo "Judge model source: auto (derived from target; finalized in run script after model normalization)"
else
  echo "Judge model source: explicit"
fi
echo "Suite: $SUITE"
if [[ -n "$LIMIT" ]]; then
  echo "Limit: $LIMIT"
else
  echo "Limit: <unset>"
fi
echo "TP/PP/DP: $TP/$PP/$DP"
echo "CTX: $CTX"
echo "GPU_MEM: $GPU_MEM"
echo "Target port: $TARGET_PORT"
echo "Target devices: ${TARGET_VISIBLE_DEVICES:-<all>}"
echo "Max connections: $MAX_CONNECTIONS"
echo "Target auto tool choice: $TARGET_ENABLE_AUTO_TOOL_CHOICE"
echo "Target tool call parser: ${TARGET_TOOL_CALL_PARSER:-<none>}"
echo "Target chat template kwargs: ${TARGET_CHAT_TEMPLATE_KWARGS_JSON:-<none>}"
echo "Target enforce eager: $TARGET_ENFORCE_EAGER"
echo "Judge server enabled: $JUDGE_SERVER_ENABLED"
echo "Judge server model: $JUDGE_SERVER_MODEL"
echo "Judge port: $JUDGE_PORT"
echo "Judge TP/PP/DP: $JUDGE_TP/$JUDGE_PP/$JUDGE_DP"
echo "Judge CTX: $JUDGE_CTX"
echo "Judge GPU_MEM: $JUDGE_GPU_MEM"
echo "Judge devices: ${JUDGE_VISIBLE_DEVICES:-<all>}"
echo "Judge auto tool choice: $JUDGE_ENABLE_AUTO_TOOL_CHOICE"
echo "Judge tool call parser: ${JUDGE_TOOL_CALL_PARSER:-<none>}"
echo "Judge chat template kwargs: ${JUDGE_CHAT_TEMPLATE_KWARGS_JSON:-<none>}"
echo "Judge enforce eager: $JUDGE_ENFORCE_EAGER"
echo "Modal enable output: $DFM_EVALS_MODAL_ENABLE_OUTPUT"
if [[ -n "$JUDGE_SERVER_SERVED_MODEL_NAME" ]]; then
  echo "Judge served name override: $JUDGE_SERVER_SERVED_MODEL_NAME"
fi
echo "Eval run root: $DFM_EVALS_RUN_ROOT"
echo "Eval log root: $DFM_EVALS_LOG_ROOT"
echo "Slurm stdout path pattern: $slurm_out_path"
echo "Slurm stderr path pattern: $slurm_err_path"
if [[ -n "$RUN_LABEL" ]]; then
  echo "Run label override: $RUN_LABEL"
else
  echo "Run label default base: $default_run_label_base"
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  echo "Extra args: $EXTRA_ARGS"
fi
if [[ -n "$DFM_EVALS_EEE_OUTPUT_DIR" ]]; then
  echo "EEE output dir override: $DFM_EVALS_EEE_OUTPUT_DIR"
fi
if [[ -n "$OPENAI_BASE_URL_OVERRIDE" ]]; then
  echo "OpenAI base URL override: $OPENAI_BASE_URL_OVERRIDE"
fi

cmd=(env "${env_kv[@]}" sbatch --output "$slurm_out_path" --error "$slurm_err_path" "$SUBMIT_SCRIPT")
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
  if [[ -n "$RUN_LABEL" ]]; then
    effective_label="$RUN_LABEL"
  else
    raw_label="${default_run_label_base}__job-${job_id}"
    effective_label="${raw_label//[^[:alnum:]._-]/_}"
  fi
  echo "Job id: $job_id"
  echo "Expected run label: $effective_label"
  echo "Expected host run dir: $DFM_EVALS_RUN_ROOT/$effective_label"
  echo "Expected host log dir: $DFM_EVALS_LOG_ROOT/$effective_label"
  echo "Expected host vLLM server logs: $DFM_EVALS_LOG_ROOT/$effective_label/_vllm_server"
  echo "Slurm stdout: ${slurm_out_path//%j/$job_id}"
  echo "Slurm stderr: ${slurm_err_path//%j/$job_id}"
  echo "View this run: ./lumi/view.sh start --job-id $job_id"
fi
