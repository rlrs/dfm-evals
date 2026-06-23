#!/bin/bash
# Run dfm-evals tournament phases with explicitly managed endpoint topology.

#SBATCH --job-name=dfm-evals-tournament
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
#SBATCH --output=dfm-evals-tournament-%j.out
#SBATCH --error=dfm-evals-tournament-%j.err

set -euo pipefail

REPO_ROOT="${DFM_EVALS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_ROOT/lumi"
source "$SCRIPT_DIR/laif_runtime.sh"
source "$SCRIPT_DIR/artifact_root.sh"
POST_ARTIFACT_ROOT="$(resolve_post_artifact_root "$REPO_ROOT")"
export POST_ARTIFACT_ROOT

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
SIF=${SIF:-$(dfm_lumi_default_sif)}
OVERLAY_DIR=${OVERLAY_DIR:-$(dfm_lumi_default_overlay_dir)}
SINGULARITY_GPU_MODE=${SINGULARITY_GPU_MODE:-$(dfm_lumi_default_gpu_mode "$SIF")}

PHASE=${DFM_TOURNAMENT_PHASE:-all}
CONFIG=${DFM_TOURNAMENT_CONFIG:-}
TARGET=${DFM_TOURNAMENT_TARGET:-}
LAUNCH_MAP=${DFM_TOURNAMENT_LAUNCH_MAP:-}
GENERATE_MODELS_CSV=${DFM_TOURNAMENT_GENERATE_MODELS:-}
ADDED_MODELS_CSV=${DFM_TOURNAMENT_ADDED_MODELS:-}
CONTESTANT_MODELS_CSV=${DFM_TOURNAMENT_CONTESTANT_MODELS:-}
JUDGE_MODEL_OVERRIDE=${DFM_TOURNAMENT_JUDGE_MODEL:-}
MAX_BATCHES=${DFM_TOURNAMENT_MAX_BATCHES:-}
MAX_TOTAL_MATCHES_OVERRIDE=${DFM_TOURNAMENT_MAX_TOTAL_MATCHES:-}
CONTESTANT_PORT=${DFM_TOURNAMENT_CONTESTANT_PORT:-8000}
JUDGE_PORT=${DFM_TOURNAMENT_JUDGE_PORT:-8001}
DFM_TOURNAMENT_EXPORT_EEE=${DFM_TOURNAMENT_EXPORT_EEE:-1}
DFM_EVALS_OPENAI_BASE_URL=${DFM_EVALS_OPENAI_BASE_URL:-${OPENAI_BASE_URL:-}}
DFM_EVALS_OPENAI_API_KEY=${DFM_EVALS_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}
DFM_EVALS_EEE_OUTPUT_DIR=${DFM_EVALS_EEE_OUTPUT_DIR:-$POST_ARTIFACT_ROOT/evals/eee/data}
DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME=${DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME:-dfm-evals}
DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP=${DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP:-third_party}

SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-900}
SERVER_START_STEP=${SERVER_START_STEP:-5}
HF_HOME=${HF_HOME:-/scratch/project_465002183/.cache/huggingface}
HF_HOME=${HF_HOME%/}
mkdir -p "$HF_HOME"
HF_HOME="$(readlink -f "$HF_HOME" 2>/dev/null || printf '%s' "$HF_HOME")"
LORA_MAX_RANK_OVERRIDE=${LORA_MAX_RANK_OVERRIDE:-}
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-29501}
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

SLURM_JOB_ID_SAFE=${SLURM_JOB_ID:-manual}
DFM_EVALS_RUN_ROOT=${DFM_EVALS_RUN_ROOT:-$POST_ARTIFACT_ROOT/evals/runs}
if [[ -z "$DFM_EVALS_EEE_OUTPUT_DIR" ]]; then
  DFM_EVALS_EEE_OUTPUT_DIR="${POST_ARTIFACT_ROOT}/evals/eee/data"
fi

DFM_EVALS_RUN_LABEL=${DFM_EVALS_RUN_LABEL:-}
DFM_EVALS_CACHE_DIR=""
DFM_EVALS_RUN_DIR=""
DFM_EVALS_SERVER_LOG_DIR=""
RUNTIME_CONFIG_PATH=""
ACTIVE_TOURNAMENT_SOURCE=""
ACTIVE_MATCH_ENDPOINT_URL=""
ACTIVE_MATCH_ENDPOINT_API_KEY=""

ACTIVE_SERVER_PID=""
ACTIVE_SERVER_URL=""
ACTIVE_SERVER_API_KEY=""
ACTIVE_SERVER_LOG=""

declare -a TARGET_CONTESTANT_MODELS=()
declare -a SING_BIND_ARGS=()
declare -a BOUND_PATHS=()
declare -a NODES=()
HEAD_NODE=""

die() {
  echo "FATAL: $*" >&2
  exit 1
}

sanitize_label() {
  local value="$1"
  value="${value//[^[:alnum:]._-]/_}"
  [[ -n "$value" ]] || value="tournament"
  printf '%s' "$value"
}

label_from_path() {
  local path="$1"
  local base

  if [[ -d "$path" ]]; then
    base="$(basename "$path")"
    if [[ "$base" == "state" || "$base" == "config" ]]; then
      base="$(basename "$(dirname "$path")")"
    fi
  else
    base="$(basename "$path")"
    base="${base%.json}"
    base="${base%.yaml}"
    base="${base%.yml}"
    base="${base%.db}"
    if [[ "$base" == "runtime" || "$base" == "tournament.runtime" || "$base" == "tournament" ]]; then
      base="$(basename "$(dirname "$path")")"
      if [[ "$base" == "state" || "$base" == "config" ]]; then
        base="$(basename "$(dirname "$(dirname "$path")")")"
      fi
    fi
  fi

  printf '%s' "$(sanitize_label "$base")"
}

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

join_by_comma() {
  local IFS=","
  printf '%s' "$*"
}

copy_source_config_into_run_dir() {
  local source_path="$1"
  local config_dir="$2"
  local suffix=".json"
  local destination=""

  [[ -f "$source_path" ]] || return 0
  case "$source_path" in
    *.yaml) suffix=".yaml" ;;
    *.yml) suffix=".yml" ;;
    *.json) suffix=".json" ;;
    *) suffix=".source" ;;
  esac

  destination="${config_dir}/source${suffix}"
  cp "$source_path" "$destination"
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
  printf '%s' "$label"
}

model_developer_from_ref() {
  local model_ref="$1"
  local ref="$model_ref"
  ref="${ref#vllm/}"
  ref="${ref#openai/}"
  if [[ "$ref" == */* ]]; then
    printf '%s' "${ref%%/*}"
  else
    printf '%s' ""
  fi
}

need_launch_map() {
  case "$PHASE" in
    all|generate|run|resume|add-model)
      [[ -n "$LAUNCH_MAP" ]] || die "DFM_TOURNAMENT_LAUNCH_MAP is required for phase $PHASE"
      ;;
  esac
}

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

container_env_prefix() {
  cat <<'EOC'
set -euo pipefail
source /overlay/venv/vllm-min/bin/activate
if [[ -f /overlay/overlay-runtime.env ]]; then
  # shellcheck disable=SC1091
  source /overlay/overlay-runtime.env
else
  OVERLAY_SITE=/overlay/venv/vllm-min/lib/python3.12/site-packages
  export PYTHONPATH=/overlay/src/transformers/src:/overlay/src/vllm:${OVERLAY_SITE}${PYTHONPATH:+:$PYTHONPATH}
fi
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi
unset ROCR_VISIBLE_DEVICES
export HF_HOME="__HF_HOME__"
if [[ -n "${DFM_EVALS_OPENAI_BASE_URL:-}" ]]; then
  export OPENAI_BASE_URL="${DFM_EVALS_OPENAI_BASE_URL}"
fi
if [[ -n "${DFM_EVALS_OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${DFM_EVALS_OPENAI_API_KEY}"
fi
EOC
}

render_env() {
  container_env_prefix | sed "s|__HF_HOME__|$HF_HOME|g"
}

append_bind_path() {
  local path="${1:-}"
  local resolved=""
  local candidate_real=""
  local bound
  local bound_real=""

  [[ -n "$path" ]] || return 0
  [[ -e "$path" ]] || return 0

  if resolved="$(readlink -f "$path" 2>/dev/null)"; then
    candidate_real="$resolved"
  fi

  for bound in "${BOUND_PATHS[@]}"; do
    bound_real="$bound"
    if resolved="$(readlink -f "$bound" 2>/dev/null)"; then
      bound_real="$resolved"
    fi

    if [[ "$path" == "$bound" || "$path" == "$bound/"* ]]; then
      return 0
    fi
    if [[ -n "$candidate_real" && ( "$candidate_real" == "$bound_real" || "$candidate_real" == "$bound_real/"* ) ]]; then
      return 0
    fi
  done

  if [[ "$path" != /* ]]; then
    if [[ -n "$candidate_real" ]]; then
      path="$candidate_real"
    else
      return 0
    fi
  fi

  SING_BIND_ARGS+=(-B "$path:$path")
  BOUND_PATHS+=("$path")
}

init_singularity_binds() {
  [[ -f "$SIF" ]] || die "SIF not found: $SIF"
  [[ -d "$OVERLAY_DIR/venv/vllm-min" ]] || die "overlay venv missing: $OVERLAY_DIR/venv/vllm-min"
  dfm_lumi_init_gpu_args

  SING_BIND_ARGS=(
    -B "$BASE_DIR:$LAIFS_APPL_DIR"
    -B "$OVERLAY_DIR:/overlay"
    -B "$REPO_ROOT:$REPO_ROOT"
  )
  if [[ -f "$OVERLAY_DIR/overlay-runtime.binds" ]]; then
    while IFS= read -r bind_spec; do
      [[ -n "$bind_spec" ]] || continue
      SING_BIND_ARGS+=(-B "$bind_spec")
    done < "$OVERLAY_DIR/overlay-runtime.binds"
  fi
  BOUND_PATHS=(
    "$BASE_DIR"
    "$OVERLAY_DIR"
    "$REPO_ROOT"
  )

  if [[ -e "$POST_ARTIFACT_ROOT" ]]; then
    ARTIFACT_ROOT_TARGET="$(readlink -f "$POST_ARTIFACT_ROOT" || true)"
    if [[ -n "$ARTIFACT_ROOT_TARGET" && -d "$ARTIFACT_ROOT_TARGET" ]]; then
      SING_BIND_ARGS+=(-B "$ARTIFACT_ROOT_TARGET:$ARTIFACT_ROOT_TARGET")
      BOUND_PATHS+=("$ARTIFACT_ROOT_TARGET")
    elif [[ -d "$POST_ARTIFACT_ROOT" ]]; then
      SING_BIND_ARGS+=(-B "$POST_ARTIFACT_ROOT:$POST_ARTIFACT_ROOT")
      BOUND_PATHS+=("$POST_ARTIFACT_ROOT")
    fi
  fi
  append_bind_path "$HF_HOME"
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

  append_bind_path "$CONFIG"
  append_bind_path "$TARGET"
  append_bind_path "$LAUNCH_MAP"
}

run_container_script() {
  local script="$1"
  singularity exec "${SINGULARITY_GPU_ARGS[@]}" "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$script"
}

run_overlay_python() {
  local -a cmd=(python "$REPO_ROOT/lumi/tournament_launch.py" "$@")
  local cmd_escaped
  local repo_root_q

  printf -v cmd_escaped '%q ' "${cmd[@]}"
  printf -v repo_root_q '%q' "$REPO_ROOT"

  run_container_script "$(render_env)
export PYTHONPATH=${repo_root_q}\${PYTHONPATH:+:\$PYTHONPATH}
cd ${repo_root_q}
${cmd_escaped}"
}

run_evals_command() {
  local endpoint_script="$1"
  shift

  local -a cmd=(evals "$@")
  local cmd_escaped
  local repo_root_q
  local run_dir_q
  local cache_dir_q

  printf -v cmd_escaped '%q ' "${cmd[@]}"
  printf -v repo_root_q '%q' "$REPO_ROOT"
  printf -v run_dir_q '%q' "$DFM_EVALS_RUN_DIR"
  printf -v cache_dir_q '%q' "$DFM_EVALS_CACHE_DIR"

  run_container_script "$(render_env)
export PYTHONPATH=${repo_root_q}\${PYTHONPATH:+:\$PYTHONPATH}
export XDG_CACHE_HOME=${cache_dir_q}
mkdir -p ${cache_dir_q}
mkdir -p ${run_dir_q}
${endpoint_script}
cd ${run_dir_q}
command -v evals >/dev/null 2>&1 || { echo 'FATAL: evals not installed in overlay venv'; exit 1; }
${cmd_escaped}"
}

eval_helper_shell() {
  local output="$1"
  eval "$output"
}

load_target_shell() {
  local output
  output="$(run_overlay_python emit-target-shell --source "$1")" || die "failed to resolve tournament target shell for $1"
  eval_helper_shell "$output"
}

load_target_shell_stateful() {
  local output
  output="$(run_overlay_python emit-target-shell --source "$1" --stateful)" || die "failed to resolve tournament target shell for $1"
  eval_helper_shell "$output"
}

load_status_shell() {
  local source="$1"
  shift || true
  local output
  output="$(run_overlay_python emit-status-shell --source "$source" "$@")" || die "failed to resolve tournament status for $source"
  eval_helper_shell "$output"
}

load_spec_shell() {
  local source="$1"
  local kind="$2"
  local model_name="${3:-}"
  local output
  local -a args=(
    emit-spec-shell
    --source "$source"
    --launch-map "$LAUNCH_MAP"
    --kind "$kind"
  )

  if [[ -n "$model_name" ]]; then
    args+=(--model "$model_name")
  fi

  output="$(run_overlay_python "${args[@]}")" || die "failed to resolve $kind launch spec"
  eval_helper_shell "$output"
}

stop_active_server() {
  set +e
  if [[ -n "$ACTIVE_SERVER_PID" ]] && kill -0 "$ACTIVE_SERVER_PID" >/dev/null 2>&1; then
    kill "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
    wait "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
  fi
  set -e

  ACTIVE_SERVER_PID=""
  ACTIVE_SERVER_URL=""
  ACTIVE_SERVER_API_KEY=""
  ACTIVE_SERVER_LOG=""
  ACTIVE_MATCH_ENDPOINT_URL=""
  ACTIVE_MATCH_ENDPOINT_API_KEY=""
}

cleanup() {
  stop_active_server
}
trap cleanup EXIT

start_vllm_server_from_spec() {
  local name="$1"
  local port="$2"
  local log_path="$3"

  local model_path="$SPEC_MODEL"
  local served_model_name="$SPEC_SERVED_MODEL_NAME"
  local api_key="$SPEC_API_KEY"
  local tp="${SPEC_TP:-1}"
  local pp="${SPEC_PP:-1}"
  local dp="${SPEC_DP:-1}"
  local nodes="${SPEC_NODES:-1}"
  local ctx="${SPEC_CTX:-4096}"
  local gpu_mem="${SPEC_GPU_MEM:-0.92}"
  local safetensors_load_strategy="${SPEC_SAFETENSORS_LOAD_STRATEGY:-}"
  local visible_devices="${SPEC_VISIBLE_DEVICES:-}"
  local default_chat_template_kwargs_json="${SPEC_DEFAULT_CHAT_TEMPLATE_KWARGS_JSON:-}"
  local trust_remote_code="${SPEC_TRUST_REMOTE_CODE:-0}"
  local enable_auto_tool_choice="${SPEC_ENABLE_AUTO_TOOL_CHOICE:-0}"
  local tool_call_parser="${SPEC_TOOL_CALL_PARSER:-}"
  local enforce_eager="${SPEC_ENFORCE_EAGER:-0}"
  local -a extra_env_kv=("${SPEC_EXTRA_ENV_KV[@]}")
  local -a extra_vllm_args=("${SPEC_EXTRA_VLLM_ARGS[@]}")

  local adapter_config="$model_path/adapter_config.json"
  local adapter_weights="$model_path/adapter_model.safetensors"
  local adapter_chat_template="$model_path/chat_template.jinja"
  local use_lora_adapter=0
  local model_tag="$model_path"
  local lora_base_model=""
  local lora_module_name=""
  local lora_module_spec=""
  local lora_adapter_path="$model_path"
  local lora_chat_template_path=""
  local lora_tokenizer_path=""
  local lora_max_rank="${LORA_MAX_RANK_OVERRIDE:-}"
  local adapter_meta=""
  local lora_base_developer=""
  local resolved_model_path=""
  local resolved_path=""
  local server_cmd=""
  local server_pid=""
  local ready=0
  local i=0
  local headless=0
  local rank_log_path=""
  local selected_node_list=""
  local master_addr="$MASTER_ADDR"
  local world_size=0
  local required_gpus_per_node=0
  local extra_env_exports=""
  local extra_vllm_flag_lines=""
  local extra_env_entry=""
  local extra_env_key=""
  local extra_env_value=""
  local extra_vllm_arg=""
  local -a selected_nodes=()
  local -a readiness_curl_args=()

  [[ -n "$model_path" ]] || die "$name local_vllm spec resolved an empty model path"

  if [[ -z "$api_key" ]]; then
    api_key="inspectai"
  fi
  if [[ -z "$tp" ]]; then
    tp="1"
  fi
  if [[ -z "$pp" ]]; then
    pp="1"
  fi
  if [[ -z "$dp" ]]; then
    dp="1"
  fi
  if [[ -z "$nodes" ]]; then
    nodes="1"
  fi
  if [[ -z "$ctx" ]]; then
    ctx="4096"
  fi
  if [[ -z "$gpu_mem" ]]; then
    gpu_mem="0.92"
  fi
  case "$safetensors_load_strategy" in
    "")
      safetensors_load_strategy="eager"
      ;;
    none)
      safetensors_load_strategy=""
      ;;
  esac
  if [[ "$tool_call_parser" == "none" ]]; then
    tool_call_parser=""
  fi
  case "$trust_remote_code" in
    ""|0|1) ;;
    *) die "invalid trust_remote_code for $name: $trust_remote_code" ;;
  esac
  case "$enable_auto_tool_choice" in
    ""|0|1) ;;
    *) die "invalid enable_auto_tool_choice for $name: $enable_auto_tool_choice" ;;
  esac
  case "$enforce_eager" in
    ""|0|1) ;;
    *) die "invalid enforce_eager for $name: $enforce_eager" ;;
  esac
  for extra_env_entry in "${extra_env_kv[@]}"; do
    extra_env_key="${extra_env_entry%%=*}"
    extra_env_value="${extra_env_entry#*=}"
    if [[ -z "$extra_env_key" || "$extra_env_key" == "$extra_env_entry" ]]; then
      die "invalid extra_env entry for $name: $extra_env_entry"
    fi
    if ! [[ "$extra_env_key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      die "invalid extra_env key for $name: $extra_env_key"
    fi
    case "$extra_env_key" in
      HF_HOME|HF_HUB_CACHE|HUGGINGFACE_HUB_CACHE|TRANSFORMERS_CACHE|HF_DATASETS_CACHE|VLLM_ASSETS_CACHE)
        if [[ -n "$extra_env_value" && "$extra_env_value" == /* ]]; then
          append_bind_path "$extra_env_value"
        fi
        ;;
    esac
    extra_env_exports+="export ${extra_env_key}=$(printf '%q' "$extra_env_value")"$'\n'
  done
  for extra_vllm_arg in "${extra_vllm_args[@]}"; do
    extra_vllm_flag_lines+="EXTRA_VLLM_FLAGS+=($(printf '%q' "$extra_vllm_arg"))"$'\n'
  done
  if [[ "$enable_auto_tool_choice" == "1" && -z "$tool_call_parser" ]]; then
    die "$name auto tool choice enabled but tool call parser is empty"
  fi
  if ! [[ "$nodes" =~ ^[0-9]+$ ]] || [[ "$nodes" -le 0 ]]; then
    die "invalid node count for $name: $nodes"
  fi
  if [[ "$nodes" -gt "${#NODES[@]}" ]]; then
    die "$name requires $nodes nodes, but allocation only has ${#NODES[@]}"
  fi
  world_size=$((tp * pp * dp))
  if [[ "$world_size" -lt "$nodes" ]]; then
    die "$name world size $world_size is smaller than requested nodes $nodes"
  fi
  if (( world_size % nodes != 0 )); then
    die "$name world size $world_size is not divisible across $nodes nodes"
  fi
  required_gpus_per_node=$((world_size / nodes))
  if [[ "$required_gpus_per_node" -gt 8 ]]; then
    die "$name requires $required_gpus_per_node GPUs per node, exceeding the 8-GPU LUMI node layout"
  fi
  selected_nodes=("${NODES[@]:0:$nodes}")
  selected_node_list="$(join_by_comma "${selected_nodes[@]}")"
  master_addr="${selected_nodes[0]}"
  if [[ "$nodes" -gt 1 ]]; then
    headless=1
  fi

  append_bind_path "$model_path"

  if [[ -d "$model_path" && -f "$adapter_config" && -f "$adapter_weights" ]]; then
    adapter_meta="$(parse_adapter_metadata "$adapter_config")" || die "failed to parse adapter metadata from $adapter_config"
    IFS=$'\t' read -r lora_base_model lora_max_rank <<<"$adapter_meta"
    [[ -n "$lora_base_model" ]] || die "empty base model in $adapter_config"
    [[ -n "$lora_max_rank" ]] || die "empty adapter rank in $adapter_config"

    use_lora_adapter=1
    model_tag="$lora_base_model"
    lora_module_name="$served_model_name"
    if [[ -z "$lora_module_name" ]]; then
      lora_module_name="${SPEC_NAME#vllm/}"
      lora_module_name="${lora_module_name#openai/}"
    fi
    if [[ -z "$lora_module_name" || "$lora_module_name" == "$SPEC_NAME" || "$lora_module_name" == "$model_path" ]]; then
      lora_module_name="$(model_label_from_ref "$model_path")"
      lora_base_developer="$(model_developer_from_ref "$lora_base_model")"
      if [[ -n "$lora_base_developer" ]]; then
        lora_module_name="${lora_base_developer}/${lora_module_name}"
      fi
    fi
    [[ -n "$lora_module_name" ]] || lora_module_name="adapter"
    served_model_name="$lora_module_name"

    if resolved_model_path="$(readlink -f "$model_path" 2>/dev/null)"; then
      lora_adapter_path="$resolved_model_path"
    fi
    append_bind_path "$lora_adapter_path"
    lora_module_spec="${lora_module_name}=${lora_adapter_path}"

    if [[ -f "$adapter_chat_template" ]]; then
      if resolved_path="$(readlink -f "$adapter_chat_template" 2>/dev/null)"; then
        lora_chat_template_path="$resolved_path"
      else
        lora_chat_template_path="$adapter_chat_template"
      fi
      append_bind_path "$lora_chat_template_path"
    fi

    if [[ -f "$model_path/tokenizer_config.json" || -f "$model_path/tokenizer.json" || -f "$model_path/tokenizer.model" || -f "$model_path/vocab.json" ]]; then
      if resolved_path="$(readlink -f "$model_path" 2>/dev/null)"; then
        lora_tokenizer_path="$resolved_path"
      else
        lora_tokenizer_path="$model_path"
      fi
      append_bind_path "$lora_tokenizer_path"
    fi
  fi

  server_cmd="$(render_env)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_PLUGINS=''
echo \"vllm executable: \$(command -v vllm)\"
/opt/venv/bin/python -c 'import importlib.metadata as md; print(\"runtime vllm=\" + md.version(\"vllm\")); print(\"runtime torch=\" + md.version(\"torch\")); print(\"runtime triton=\" + md.version(\"triton\"))'
${extra_env_exports}export NODE_RANK=\${SLURM_PROCID:-0}
if [[ -n \"\${VISIBLE_DEVICES:-}\" ]]; then
  export HIP_VISIBLE_DEVICES=\"\${VISIBLE_DEVICES}\"
  export CUDA_VISIBLE_DEVICES=\"\${VISIBLE_DEVICES}\"
fi
if [[ \"\${HEADLESS:-0}\" == \"1\" ]]; then
  export VLLM_DISABLE_PYNCCL=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export FI_PROVIDER=\"\${FI_PROVIDER_NAME}\"
  export FI_CXI_DISABLE_HMEM=\"\${FI_CXI_DISABLE_HMEM_FLAG}\"
  export GLOO_SOCKET_IFNAME=\"\${GLOO_IFNAME}\"
  export NCCL_SOCKET_IFNAME=\"\${NCCL_SOCKET_IFNAME}\"
  export NCCL_NET_GDR_LEVEL=\"\${NCCL_NET_GDR_LEVEL}\"
  export NCCL_DEBUG=\"\${NCCL_DEBUG_LEVEL}\"
  export NCCL_DEBUG_SUBSYS=\"\${NCCL_DEBUG_SUBSYS_LEVEL}\"
  if [[ -d \"\${AWS_OFI_RCCL_LIBDIR:-}\" ]]; then
    export LD_LIBRARY_PATH=\"\${AWS_OFI_RCCL_LIBDIR}:\${LD_LIBRARY_PATH:-}\"
  fi
  if [[ -n \"\${NCCL_NET_PLUGIN_NAME:-}\" ]]; then
    export NCCL_NET_PLUGIN=\"\${NCCL_NET_PLUGIN_NAME}\"
  else
    unset NCCL_NET_PLUGIN
  fi
  if [[ -n \"\${NCCL_NET_BACKEND:-}\" ]]; then
    export NCCL_NET=\"\${NCCL_NET_BACKEND}\"
  else
    unset NCCL_NET
  fi
fi
EXTRA_VLLM_FLAGS=()
if [[ -n \"\${DEFAULT_CHAT_TEMPLATE_KWARGS_JSON:-}\" ]]; then
  EXTRA_VLLM_FLAGS+=(--default-chat-template-kwargs \"\${DEFAULT_CHAT_TEMPLATE_KWARGS_JSON}\")
fi
if [[ \"\${TRUST_REMOTE_CODE:-0}\" == \"1\" ]]; then
  EXTRA_VLLM_FLAGS+=(--trust-remote-code)
fi
if [[ \"\${ENABLE_AUTO_TOOL_CHOICE:-0}\" == \"1\" ]]; then
  EXTRA_VLLM_FLAGS+=(--enable-auto-tool-choice)
fi
if [[ -n \"\${SAFETENSORS_LOAD_STRATEGY:-}\" ]]; then
  EXTRA_VLLM_FLAGS+=(--safetensors-load-strategy \"\${SAFETENSORS_LOAD_STRATEGY}\")
fi
if [[ -n \"\${TOOL_CALL_PARSER:-}\" ]]; then
  EXTRA_VLLM_FLAGS+=(--tool-call-parser \"\${TOOL_CALL_PARSER}\")
fi
if [[ \"\${ENFORCE_EAGER:-0}\" == \"1\" ]]; then
  EXTRA_VLLM_FLAGS+=(--enforce-eager)
fi
${extra_vllm_flag_lines}if [[ \"\${USE_LORA_ADAPTER:-0}\" == \"1\" ]]; then
  EXTRA_VLLM_FLAGS+=(--enable-lora --lora-modules \"\${LORA_MODULE_SPEC}\" --max-lora-rank \"\${LORA_MAX_RANK}\")
  if [[ -n \"\${LORA_CHAT_TEMPLATE_PATH:-}\" ]]; then
    EXTRA_VLLM_FLAGS+=(--chat-template \"\${LORA_CHAT_TEMPLATE_PATH}\")
  fi
  if [[ -n \"\${LORA_TOKENIZER_PATH:-}\" ]]; then
    EXTRA_VLLM_FLAGS+=(--tokenizer \"\${LORA_TOKENIZER_PATH}\")
  fi
fi
if [[ \"\${HEADLESS:-0}\" == \"1\" && \"\${NODE_RANK}\" != \"0\" ]]; then
  EXTRA_VLLM_FLAGS+=(--headless)
fi
if [[ \"\${NODES:-1}\" -gt 1 ]]; then
  EXTRA_VLLM_FLAGS+=(--distributed-executor-backend mp --data-parallel-backend mp --nnodes \"\${NODES}\" --node-rank \"\${NODE_RANK}\" --master-addr \"\${MASTER_ADDR}\" --master-port \"\${MASTER_PORT}\")
fi
vllm serve \"\${MODEL_TAG}\" \\
  --served-model-name \"\${SERVED_MODEL_NAME}\" \\
  --host 0.0.0.0 \\
  --port \"\${PORT}\" \\
  --api-key \"\${API_KEY}\" \\
  --dtype bfloat16 \\
  --tensor-parallel-size \"\${TP}\" \\
  --pipeline-parallel-size \"\${PP}\" \\
  --data-parallel-size \"\${DP}\" \\
  --gpu-memory-utilization \"\${GPU_MEM}\" \\
  --max-model-len \"\${CTX}\" \\
  \"\${EXTRA_VLLM_FLAGS[@]}\""

  mkdir -p "$(dirname "$log_path")"
  echo "Starting $name server on :$port using $nodes node(s) (log: $log_path)"
  if [[ "$nodes" -gt 1 ]]; then
    rank_log_path="${log_path%.log}-rank-%t.log"
    srun --overlap \
      --nodes="$nodes" \
      --ntasks="$nodes" \
      --ntasks-per-node=1 \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-56}" \
      --nodelist="$selected_node_list" \
      --output="$rank_log_path" \
      --error="$rank_log_path" \
      singularity exec "${SINGULARITY_GPU_ARGS[@]}" "${SING_BIND_ARGS[@]}" "$SIF" \
      env \
        MODEL_TAG="$model_tag" \
        SERVED_MODEL_NAME="$served_model_name" \
        PORT="$port" \
        API_KEY="$api_key" \
        TP="$tp" \
        PP="$pp" \
        DP="$dp" \
        NODES="$nodes" \
        CTX="$ctx" \
        GPU_MEM="$gpu_mem" \
        SAFETENSORS_LOAD_STRATEGY="$safetensors_load_strategy" \
        VISIBLE_DEVICES="$visible_devices" \
        DEFAULT_CHAT_TEMPLATE_KWARGS_JSON="$default_chat_template_kwargs_json" \
        TRUST_REMOTE_CODE="${trust_remote_code:-0}" \
        USE_LORA_ADAPTER="$use_lora_adapter" \
        LORA_MODULE_SPEC="$lora_module_spec" \
        LORA_MAX_RANK="$lora_max_rank" \
        LORA_CHAT_TEMPLATE_PATH="$lora_chat_template_path" \
        LORA_TOKENIZER_PATH="$lora_tokenizer_path" \
        ENABLE_AUTO_TOOL_CHOICE="${enable_auto_tool_choice:-0}" \
        TOOL_CALL_PARSER="$tool_call_parser" \
        ENFORCE_EAGER="${enforce_eager:-0}" \
        HEADLESS="$headless" \
        MASTER_ADDR="$master_addr" \
        MASTER_PORT="$MASTER_PORT" \
        FI_PROVIDER_NAME="$FI_PROVIDER_NAME" \
        FI_CXI_DISABLE_HMEM_FLAG="$FI_CXI_DISABLE_HMEM_FLAG" \
        GLOO_IFNAME="$GLOO_IFNAME" \
        NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
        NCCL_NET_GDR_LEVEL="$NCCL_NET_GDR_LEVEL" \
        NCCL_DEBUG_LEVEL="$NCCL_DEBUG_LEVEL" \
        NCCL_DEBUG_SUBSYS_LEVEL="$NCCL_DEBUG_SUBSYS_LEVEL" \
        NCCL_NET_BACKEND="$NCCL_NET_BACKEND" \
        NCCL_NET_PLUGIN_NAME="$NCCL_NET_PLUGIN_NAME" \
        AWS_OFI_RCCL_LIBDIR="$AWS_OFI_RCCL_LIBDIR" \
        bash -lc "$server_cmd" \
        >"$log_path" 2>&1 &
  else
    singularity exec "${SINGULARITY_GPU_ARGS[@]}" "${SING_BIND_ARGS[@]}" "$SIF" \
      env \
        MODEL_TAG="$model_tag" \
        SERVED_MODEL_NAME="$served_model_name" \
        PORT="$port" \
        API_KEY="$api_key" \
        TP="$tp" \
        PP="$pp" \
        DP="$dp" \
        NODES="$nodes" \
        CTX="$ctx" \
        GPU_MEM="$gpu_mem" \
        SAFETENSORS_LOAD_STRATEGY="$safetensors_load_strategy" \
        VISIBLE_DEVICES="$visible_devices" \
        DEFAULT_CHAT_TEMPLATE_KWARGS_JSON="$default_chat_template_kwargs_json" \
        TRUST_REMOTE_CODE="${trust_remote_code:-0}" \
        USE_LORA_ADAPTER="$use_lora_adapter" \
        LORA_MODULE_SPEC="$lora_module_spec" \
        LORA_MAX_RANK="$lora_max_rank" \
        LORA_CHAT_TEMPLATE_PATH="$lora_chat_template_path" \
        LORA_TOKENIZER_PATH="$lora_tokenizer_path" \
        ENABLE_AUTO_TOOL_CHOICE="${enable_auto_tool_choice:-0}" \
        TOOL_CALL_PARSER="$tool_call_parser" \
        ENFORCE_EAGER="${enforce_eager:-0}" \
        HEADLESS="$headless" \
        MASTER_ADDR="$master_addr" \
        MASTER_PORT="$MASTER_PORT" \
        FI_PROVIDER_NAME="$FI_PROVIDER_NAME" \
        FI_CXI_DISABLE_HMEM_FLAG="$FI_CXI_DISABLE_HMEM_FLAG" \
        GLOO_IFNAME="$GLOO_IFNAME" \
        NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
        NCCL_NET_GDR_LEVEL="$NCCL_NET_GDR_LEVEL" \
        NCCL_DEBUG_LEVEL="$NCCL_DEBUG_LEVEL" \
        NCCL_DEBUG_SUBSYS_LEVEL="$NCCL_DEBUG_SUBSYS_LEVEL" \
        NCCL_NET_BACKEND="$NCCL_NET_BACKEND" \
        NCCL_NET_PLUGIN_NAME="$NCCL_NET_PLUGIN_NAME" \
        AWS_OFI_RCCL_LIBDIR="$AWS_OFI_RCCL_LIBDIR" \
        bash -lc "$server_cmd" \
        >"$log_path" 2>&1 &
  fi

  server_pid=$!
  readiness_curl_args=(curl -sf)
  if [[ -n "$api_key" ]]; then
    readiness_curl_args+=(-H "Authorization: Bearer $api_key")
  fi

  for ((i=0; i<SERVER_START_TIMEOUT; i+=SERVER_START_STEP)); do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      tail -n 120 "$log_path" >&2 || true
      if [[ -n "$rank_log_path" ]]; then
        tail -n 120 "${rank_log_path//%t/0}" >&2 || true
      fi
      die "$name server exited during startup (pid=$server_pid)"
    fi
    if "${readiness_curl_args[@]}" "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep "$SERVER_START_STEP"
  done

  if [[ "$ready" != "1" ]]; then
    tail -n 120 "$log_path" >&2 || true
    if [[ -n "$rank_log_path" ]]; then
      tail -n 120 "${rank_log_path//%t/0}" >&2 || true
    fi
    die "$name server did not become ready within ${SERVER_START_TIMEOUT}s"
  fi

  ACTIVE_SERVER_PID="$server_pid"
  ACTIVE_SERVER_URL="http://127.0.0.1:$port/v1"
  ACTIVE_SERVER_API_KEY="$api_key"
  ACTIVE_SERVER_LOG="$log_path"
  echo "$name server ready at $ACTIVE_SERVER_URL"
}

resolve_external_endpoint() {
  local kind="$1"
  local resolved_base_url=""
  local resolved_api_key=""
  local env_api_key=""

  if [[ -n "$DFM_EVALS_OPENAI_BASE_URL" ]]; then
    resolved_base_url="$DFM_EVALS_OPENAI_BASE_URL"
  elif [[ -n "${SPEC_BASE_URL:-}" ]]; then
    resolved_base_url="$SPEC_BASE_URL"
  elif [[ -n "${SPEC_BASE_URL_ENV:-}" ]]; then
    resolved_base_url="${!SPEC_BASE_URL_ENV:-}"
  fi

  if [[ -n "${SPEC_API_KEY:-}" ]]; then
    resolved_api_key="$SPEC_API_KEY"
  elif [[ -n "${SPEC_API_KEY_ENV:-}" ]]; then
    env_api_key="${!SPEC_API_KEY_ENV:-}"
    if [[ -n "$env_api_key" ]]; then
      resolved_api_key="$env_api_key"
    fi
  fi

  if [[ -z "$resolved_api_key" && -n "${DFM_EVALS_OPENAI_API_KEY:-}" ]]; then
    resolved_api_key="$DFM_EVALS_OPENAI_API_KEY"
  elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
    resolved_api_key="$OPENAI_API_KEY"
  elif [[ -n "${VLLM_API_KEY:-}" ]]; then
    resolved_api_key="$VLLM_API_KEY"
  fi

  if [[ -z "$resolved_api_key" ]]; then
    resolved_api_key="inspectai"
  fi

  [[ -n "$resolved_base_url" ]] || die "$kind external_openai spec is missing a resolved base_url"

  ACTIVE_MATCH_ENDPOINT_URL="$resolved_base_url"
  ACTIVE_MATCH_ENDPOINT_API_KEY="$resolved_api_key"
}

build_endpoint_exports() {
  local base_url="$1"
  local api_key="$2"
  local base_url_q
  local api_key_q

  printf -v base_url_q '%q' "$base_url"
  printf -v api_key_q '%q' "$api_key"

  cat <<EOF
unset OPENAI_BASE_URL OPENAI_API_KEY VLLM_BASE_URL VLLM_API_KEY
export OPENAI_BASE_URL=$base_url_q
export OPENAI_API_KEY=$api_key_q
export VLLM_BASE_URL=$base_url_q
export VLLM_API_KEY=$api_key_q
EOF
}

has_contestant_model() {
  local target_model="$1"
  local model_name
  for model_name in "${TARGET_CONTESTANT_MODELS[@]}"; do
    if [[ "$model_name" == "$target_model" ]]; then
      return 0
    fi
  done
  return 1
}

build_generate_model_list() {
  local -n out_models_ref=$1
  local raw=""
  local value=""
  local -a raw_models=()
  declare -A seen=()

  out_models_ref=()
  if [[ -z "$GENERATE_MODELS_CSV" ]]; then
    out_models_ref=("${TARGET_CONTESTANT_MODELS[@]}")
    return 0
  fi

  IFS=, read -r -a raw_models <<<"$GENERATE_MODELS_CSV"
  for raw in "${raw_models[@]}"; do
    value="$(trim_whitespace "$raw")"
    [[ -n "$value" ]] || continue
    if [[ -n "${seen[$value]:-}" ]]; then
      continue
    fi
    has_contestant_model "$value" || die "generate model '$value' is not present in the tournament config"
    seen["$value"]=1
    out_models_ref+=("$value")
  done

  [[ "${#out_models_ref[@]}" -gt 0 ]] || die "no generate models were selected"
}

build_runtime_override_args() {
  local -n out_args_ref=$1
  local raw=""
  local value=""
  local -a raw_models=()
  declare -A seen=()

  out_args_ref=()

  if [[ -n "$CONTESTANT_MODELS_CSV" ]]; then
    IFS=, read -r -a raw_models <<<"$CONTESTANT_MODELS_CSV"
    for raw in "${raw_models[@]}"; do
      value="$(trim_whitespace "$raw")"
      [[ -n "$value" ]] || continue
      if [[ -n "${seen[$value]:-}" ]]; then
        continue
      fi
      seen["$value"]=1
      out_args_ref+=(--contestant-model "$value")
    done
  fi

  if [[ -n "$JUDGE_MODEL_OVERRIDE" ]]; then
    out_args_ref+=(--judge-model "$JUDGE_MODEL_OVERRIDE")
  fi
}

build_add_model_list() {
  local -n out_models_ref=$1
  local raw=""
  local value=""
  local -a raw_models=()
  declare -A seen=()

  out_models_ref=()
  [[ -n "$ADDED_MODELS_CSV" ]] || die "DFM_TOURNAMENT_ADDED_MODELS is required for phase add-model"

  IFS=, read -r -a raw_models <<<"$ADDED_MODELS_CSV"
  for raw in "${raw_models[@]}"; do
    value="$(trim_whitespace "$raw")"
    [[ -n "$value" ]] || continue
    if [[ -n "${seen[$value]:-}" ]]; then
      continue
    fi
    seen["$value"]=1
    out_models_ref+=("$value")
  done

  [[ "${#out_models_ref[@]}" -gt 0 ]] || die "no add-model contestants were selected"
}

ensure_target_paths_bound() {
  append_bind_path "$RUNTIME_CONFIG_PATH"
  append_bind_path "$DFM_EVALS_RUN_DIR"
  append_bind_path "$TARGET_RUN_DIR"
  append_bind_path "$TARGET_CONFIG_DIR"
  append_bind_path "$TARGET_INSPECT_DIR"
  append_bind_path "$TARGET_GENERATION_LOG_DIR"
  append_bind_path "$TARGET_JUDGE_LOG_DIR"
  append_bind_path "$TARGET_STATE_DIR"
  append_bind_path "$TARGET_EXPORTS_DIR"
  append_bind_path "$TARGET_TRACES_DIR"
  append_bind_path "$TARGET_SERVICES_DIR"
  append_bind_path "$TARGET_VLLM_LOG_DIR"
}

prepare_new_target() {
  local default_label_base=""
  local raw_run_label=""
  local -a runtime_override_args=()

  [[ -n "$CONFIG" ]] || die "DFM_TOURNAMENT_CONFIG is required for phase $PHASE"
  default_label_base="tournament__$(label_from_path "$CONFIG")"
  if [[ -n "$DFM_EVALS_RUN_LABEL" ]]; then
    raw_run_label="$DFM_EVALS_RUN_LABEL"
  else
    raw_run_label="${default_label_base}__job-${SLURM_JOB_ID_SAFE}"
  fi

  DFM_EVALS_RUN_LABEL="$(sanitize_label "$raw_run_label")"
  DFM_EVALS_CACHE_DIR="/overlay/cache/dfm-evals-${DFM_EVALS_RUN_LABEL}"
  DFM_EVALS_RUN_DIR="${DFM_EVALS_RUN_ROOT}/${DFM_EVALS_RUN_LABEL}"
  DFM_EVALS_SERVER_LOG_DIR="${DFM_EVALS_RUN_DIR}/services/vllm"
  RUNTIME_CONFIG_PATH="${DFM_EVALS_RUN_DIR}/config/runtime.json"

  mkdir -p "$DFM_EVALS_RUN_DIR/config" "$DFM_EVALS_SERVER_LOG_DIR"

  build_runtime_override_args runtime_override_args

  run_overlay_python \
    write-runtime-config \
    --source "$CONFIG" \
    --output "$RUNTIME_CONFIG_PATH" \
    --run-dir "$DFM_EVALS_RUN_DIR" \
    "${runtime_override_args[@]}" \
    >/dev/null

  load_target_shell "$RUNTIME_CONFIG_PATH"
  mkdir -p \
    "$TARGET_CONFIG_DIR" \
    "$TARGET_GENERATION_LOG_DIR" \
    "$TARGET_JUDGE_LOG_DIR" \
    "$TARGET_STATE_DIR" \
    "$TARGET_EXPORTS_DIR" \
    "$TARGET_TRACES_DIR" \
    "$TARGET_VLLM_LOG_DIR"
  DFM_EVALS_SERVER_LOG_DIR="$TARGET_VLLM_LOG_DIR"
  copy_source_config_into_run_dir "$CONFIG" "$TARGET_CONFIG_DIR"
  ensure_target_paths_bound
  ACTIVE_TOURNAMENT_SOURCE="$RUNTIME_CONFIG_PATH"

  echo "Run label: $DFM_EVALS_RUN_LABEL"
  echo "Runtime config: $RUNTIME_CONFIG_PATH"
  echo "Host run dir: $DFM_EVALS_RUN_DIR"
  echo "Inspect generation dir: $TARGET_GENERATION_LOG_DIR"
  echo "Inspect judge dir: $TARGET_JUDGE_LOG_DIR"
  echo "Tournament state dir: $TARGET_STATE_DIR"
  echo "Tournament exports dir: $TARGET_EXPORTS_DIR"
}

prepare_existing_target() {
  [[ -n "$TARGET" ]] || die "DFM_TOURNAMENT_TARGET is required for phase $PHASE"

  load_target_shell_stateful "$TARGET"

  DFM_EVALS_RUN_LABEL="$TARGET_RUN_LABEL"
  DFM_EVALS_CACHE_DIR="/overlay/cache/dfm-evals-${DFM_EVALS_RUN_LABEL}"
  DFM_EVALS_RUN_DIR="$TARGET_RUN_DIR"
  DFM_EVALS_SERVER_LOG_DIR="$TARGET_VLLM_LOG_DIR"
  RUNTIME_CONFIG_PATH="$TARGET_RUNTIME_CONFIG_PATH"

  mkdir -p "$DFM_EVALS_RUN_DIR" "$TARGET_CONFIG_DIR" "$DFM_EVALS_SERVER_LOG_DIR"
  append_bind_path "$DFM_EVALS_RUN_DIR"
  append_bind_path "$DFM_EVALS_SERVER_LOG_DIR"

  run_overlay_python \
    write-runtime-config \
    --source "$TARGET" \
    --stateful \
    --output "$RUNTIME_CONFIG_PATH" \
    >/dev/null

  load_target_shell "$RUNTIME_CONFIG_PATH"
  mkdir -p \
    "$TARGET_CONFIG_DIR" \
    "$TARGET_GENERATION_LOG_DIR" \
    "$TARGET_JUDGE_LOG_DIR" \
    "$TARGET_STATE_DIR" \
    "$TARGET_EXPORTS_DIR" \
    "$TARGET_TRACES_DIR" \
    "$TARGET_VLLM_LOG_DIR"
  ensure_target_paths_bound
  ACTIVE_TOURNAMENT_SOURCE="$RUNTIME_CONFIG_PATH"

  echo "Run label: $DFM_EVALS_RUN_LABEL"
  echo "Target: $TARGET"
  echo "Runtime config: $RUNTIME_CONFIG_PATH"
  echo "Host run dir: $DFM_EVALS_RUN_DIR"
  echo "Inspect generation dir: $TARGET_GENERATION_LOG_DIR"
  echo "Inspect judge dir: $TARGET_JUDGE_LOG_DIR"
  echo "Tournament state dir: $TARGET_STATE_DIR"
  echo "Tournament exports dir: $TARGET_EXPORTS_DIR"
}

refresh_existing_runtime_config() {
  run_overlay_python \
    write-runtime-config \
    --source "$TARGET" \
    --stateful \
    --output "$RUNTIME_CONFIG_PATH" \
    >/dev/null

  load_target_shell "$RUNTIME_CONFIG_PATH"
  mkdir -p \
    "$TARGET_CONFIG_DIR" \
    "$TARGET_GENERATION_LOG_DIR" \
    "$TARGET_JUDGE_LOG_DIR" \
    "$TARGET_STATE_DIR" \
    "$TARGET_EXPORTS_DIR" \
    "$TARGET_TRACES_DIR" \
    "$TARGET_VLLM_LOG_DIR"
  ensure_target_paths_bound
  ACTIVE_TOURNAMENT_SOURCE="$RUNTIME_CONFIG_PATH"
}

apply_max_total_matches_override() {
  [[ -n "$MAX_TOTAL_MATCHES_OVERRIDE" ]] || return 0
  echo "Updating max_total_matches to $MAX_TOTAL_MATCHES_OVERRIDE"
  run_evals_command \
    "" \
    tournament update-config \
    "$ACTIVE_TOURNAMENT_SOURCE" \
    --max-total-matches "$MAX_TOTAL_MATCHES_OVERRIDE"
  refresh_existing_runtime_config
}

initialize_target_state() {
  echo "Initializing tournament state"
  run_evals_command "" tournament init --config "$ACTIVE_TOURNAMENT_SOURCE"
}

generate_for_model() {
  local model_name="$1"
  local endpoint_script=""
  local server_log=""
  local log_label=""

  load_spec_shell "$ACTIVE_TOURNAMENT_SOURCE" contestant "$model_name"
  echo "Generating contestant responses for $model_name (mode: $SPEC_MODE)"

  case "$SPEC_MODE" in
    local_vllm)
      log_label="$(model_label_from_ref "$model_name")"
      server_log="${DFM_EVALS_SERVER_LOG_DIR}/contestant-${log_label}-vllm-${SLURM_JOB_ID_SAFE}.log"
      start_vllm_server_from_spec "contestant/$model_name" "$CONTESTANT_PORT" "$server_log"
      ACTIVE_MATCH_ENDPOINT_URL="$ACTIVE_SERVER_URL"
      ACTIVE_MATCH_ENDPOINT_API_KEY="$ACTIVE_SERVER_API_KEY"
      ;;
    external_openai)
      resolve_external_endpoint "contestant/$model_name"
      ;;
    *)
      die "unsupported contestant mode: $SPEC_MODE"
      ;;
  esac

  endpoint_script="$(build_endpoint_exports "$ACTIVE_MATCH_ENDPOINT_URL" "$ACTIVE_MATCH_ENDPOINT_API_KEY")"
  run_evals_command \
    "$endpoint_script" \
    tournament generate \
    --config "$ACTIVE_TOURNAMENT_SOURCE" \
    --models "$model_name"

  stop_active_server
}

run_generation_phase() {
  local models=()
  local model_name=""

  build_generate_model_list models
  echo "Generation models: ${models[*]}"
  for model_name in "${models[@]}"; do
    generate_for_model "$model_name"
  done
}

register_add_model_phase() {
  local models=()
  local model_name=""
  local -a register_cmd=(
    tournament
    register-model
    "$ACTIVE_TOURNAMENT_SOURCE"
  )

  build_add_model_list models
  echo "Registering contestant models: ${models[*]}"
  for model_name in "${models[@]}"; do
    register_cmd+=(--model "$model_name")
  done

  run_evals_command "" "${register_cmd[@]}"
  refresh_existing_runtime_config
}

check_generation_complete() {
  local previous_generate_models_csv=""
  if [[ -n "$GENERATE_MODELS_CSV" && ( "$PHASE" == "run" || "$PHASE" == "resume" ) ]]; then
    echo "Regenerating explicitly requested contestant responses before phase $PHASE: ${GENERATE_MODELS_CSV}"
    run_generation_phase
  fi
  load_status_shell "$ACTIVE_TOURNAMENT_SOURCE" --index-generation
  if [[ "${STATUS_MISSING_RESPONSES:-0}" != "0" && ( "$PHASE" == "run" || "$PHASE" == "resume" || "$PHASE" == "add-model" ) ]]; then
    [[ -n "${STATUS_MISSING_MODELS_CSV:-}" ]] || die "tournament state is missing responses but did not report missing contestant models"
    echo "Recovering incomplete generation for missing contestants: ${STATUS_MISSING_MODELS_CSV}"
    previous_generate_models_csv="$GENERATE_MODELS_CSV"
    GENERATE_MODELS_CSV="$STATUS_MISSING_MODELS_CSV"
    run_generation_phase
    GENERATE_MODELS_CSV="$previous_generate_models_csv"
    load_status_shell "$ACTIVE_TOURNAMENT_SOURCE" --index-generation
  fi
  if [[ "${STATUS_MISSING_RESPONSES:-0}" != "0" ]]; then
    die "tournament state still has ${STATUS_MISSING_RESPONSES} missing responses; run generate for the remaining contestants before phase $PHASE"
  fi
}

run_judge_phase() {
  local endpoint_script=""
  local judge_cmd=""

  load_spec_shell "$ACTIVE_TOURNAMENT_SOURCE" judge
  echo "Running judge phase (phase: $PHASE, mode: $SPEC_MODE, judge model: $TARGET_JUDGE_MODEL)"

  case "$SPEC_MODE" in
    local_vllm)
      start_vllm_server_from_spec "judge" "$JUDGE_PORT" "${DFM_EVALS_SERVER_LOG_DIR}/judge-vllm-${SLURM_JOB_ID_SAFE}.log"
      ACTIVE_MATCH_ENDPOINT_URL="$ACTIVE_SERVER_URL"
      ACTIVE_MATCH_ENDPOINT_API_KEY="$ACTIVE_SERVER_API_KEY"
      ;;
    external_openai)
      resolve_external_endpoint "judge"
      ;;
    *)
      die "unsupported judge mode: $SPEC_MODE"
      ;;
  esac

  endpoint_script="$(build_endpoint_exports "$ACTIVE_MATCH_ENDPOINT_URL" "$ACTIVE_MATCH_ENDPOINT_API_KEY")"
  if [[ "$PHASE" == "run" || "$PHASE" == "all" ]]; then
    if [[ -n "$MAX_BATCHES" ]]; then
      run_evals_command \
        "$endpoint_script" \
        tournament run \
        --config "$ACTIVE_TOURNAMENT_SOURCE" \
        --max-batches "$MAX_BATCHES"
    else
      run_evals_command \
        "$endpoint_script" \
        tournament run \
        --config "$ACTIVE_TOURNAMENT_SOURCE"
    fi
  else
    if [[ -n "$MAX_BATCHES" ]]; then
      run_evals_command \
        "$endpoint_script" \
        tournament resume \
        "$ACTIVE_TOURNAMENT_SOURCE" \
        --max-batches "$MAX_BATCHES"
    else
      run_evals_command \
        "$endpoint_script" \
        tournament resume \
        "$ACTIVE_TOURNAMENT_SOURCE"
    fi
  fi

  stop_active_server
}

run_export_phase() {
  echo "Exporting tournament rankings"
  run_evals_command "" tournament export "$ACTIVE_TOURNAMENT_SOURCE"

  if [[ "$DFM_TOURNAMENT_EXPORT_EEE" == "1" ]]; then
    mkdir -p "$DFM_EVALS_EEE_OUTPUT_DIR"
    append_bind_path "$DFM_EVALS_EEE_OUTPUT_DIR"
    echo "Exporting tournament Every Eval Ever records to $DFM_EVALS_EEE_OUTPUT_DIR"
    run_evals_command \
      "" \
      eee tournament \
      --target "$ACTIVE_TOURNAMENT_SOURCE" \
      --output-dir "$DFM_EVALS_EEE_OUTPUT_DIR" \
      --source-organization-name "$DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME" \
      --evaluator-relationship "$DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP"
  fi
}

case "$PHASE" in
  all|generate|run|resume|add-model|export)
    ;;
  *)
    die "invalid DFM_TOURNAMENT_PHASE: $PHASE"
    ;;
esac

need_launch_map
init_singularity_binds
[[ "${SLURM_NNODES:-0}" -ge 1 ]] || die "this script requires at least 1 node"
readarray -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#NODES[@]}" -ge 1 ]] || die "expected at least 1 node, got ${#NODES[@]}"
HEAD_NODE="${NODES[0]}"
if [[ -z "$MASTER_ADDR" ]]; then
  MASTER_ADDR="$HEAD_NODE"
fi

echo "SIF: $SIF"
echo "Overlay: $OVERLAY_DIR"
echo "Singularity GPU mode: $SINGULARITY_GPU_MODE"
echo "Phase: $PHASE"
echo "Allocated nodes: ${NODES[*]}"
echo "Head node: $HEAD_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "HF_HOME: $HF_HOME"
if [[ -n "$CONFIG" ]]; then
  echo "Config: $CONFIG"
fi
if [[ -n "$TARGET" ]]; then
  echo "Target: $TARGET"
fi
if [[ -n "$LAUNCH_MAP" ]]; then
  echo "Launch map: $LAUNCH_MAP"
fi
echo "Contestant port: $CONTESTANT_PORT"
echo "Judge port: $JUDGE_PORT"
echo "Export EEE: $DFM_TOURNAMENT_EXPORT_EEE"
if [[ -n "$MAX_BATCHES" ]]; then
  echo "Max batches: $MAX_BATCHES"
fi

case "$PHASE" in
  all|generate)
    prepare_new_target
    initialize_target_state
    ;;
  run|resume|add-model|export)
    prepare_existing_target
    ;;
esac

if [[ "$PHASE" == "run" || "$PHASE" == "resume" ]]; then
  apply_max_total_matches_override
fi

case "$PHASE" in
  all)
    run_generation_phase
    check_generation_complete
    run_judge_phase
    run_export_phase
    ;;
  generate)
    run_generation_phase
    ;;
  run|resume)
    check_generation_complete
    run_judge_phase
    ;;
  add-model)
    register_add_model_phase
    apply_max_total_matches_override
    check_generation_complete
    run_judge_phase
    ;;
  export)
    run_export_phase
    ;;
esac

echo "Completed tournament phase: $PHASE"
