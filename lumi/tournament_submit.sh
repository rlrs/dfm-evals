#!/bin/bash
# Friendly wrapper for submitting tournament jobs on LUMI.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/artifact_root.sh"
source "$SCRIPT_DIR/laif_runtime.sh"
POST_ARTIFACT_ROOT="$(resolve_post_artifact_root "$REPO_ROOT")"
export POST_ARTIFACT_ROOT
DEFAULT_SUBMIT_SCRIPT="$SCRIPT_DIR/run_tournament.sbatch"
ENV_FILE=${ENV_FILE:-$REPO_ROOT/.env}

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

SUBMIT_SCRIPT=${SUBMIT_SCRIPT:-$DEFAULT_SUBMIT_SCRIPT}
BASE_DIR=/pfs/lustref1/appl/local/laifs
OVERLAY_DIR=${OVERLAY_DIR:-$(dfm_lumi_default_overlay_dir)}

DFM_EVALS_RUN_ROOT=${DFM_EVALS_RUN_ROOT:-$POST_ARTIFACT_ROOT/evals/runs}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-$POST_ARTIFACT_ROOT/evals/slurm}
TOURNAMENT_DEFINITIONS_DIR=${TOURNAMENT_DEFINITIONS_DIR:-$REPO_ROOT/configs/tournaments}

PHASE=${PHASE:-all}
DEFINITION=${DEFINITION:-}
CONFIG=${CONFIG:-}
TARGET=${TARGET:-}
LAUNCH_MAP=${LAUNCH_MAP:-}
RUN_LABEL=${RUN_LABEL:-}
MAX_BATCHES=${MAX_BATCHES:-}
MAX_TOTAL_MATCHES_OVERRIDE=${MAX_TOTAL_MATCHES_OVERRIDE:-}
CONTESTANT_PORT=${CONTESTANT_PORT:-8000}
JUDGE_PORT=${JUDGE_PORT:-8001}
DFM_TOURNAMENT_EXPORT_EEE=${DFM_TOURNAMENT_EXPORT_EEE:-1}
DFM_EVALS_EEE_OUTPUT_DIR=${DFM_EVALS_EEE_OUTPUT_DIR:-$POST_ARTIFACT_ROOT/evals/eee/data}
DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME=${DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME:-dfm-evals}
DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP=${DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP:-third_party}
NODES_OVERRIDE=${NODES_OVERRIDE:-}
OPENAI_BASE_URL_OVERRIDE=${OPENAI_BASE_URL_OVERRIDE:-}
OPENAI_API_KEY_OVERRIDE=${OPENAI_API_KEY_OVERRIDE:-${OPENAI_API_KEY:-}}
JUDGE_MODEL_OVERRIDE=${JUDGE_MODEL_OVERRIDE:-}

GENERATE_MODELS=()
CONTESTANT_MODEL_OVERRIDES=()
DRY_RUN=0
LIST_DEFINITIONS=0

usage() {
  cat <<'EOF'
Usage:
  ./lumi/tournament_submit.sh [options]

Options:
  --phase <name>            Phase to run: all|generate|run|resume|add-model|export (default: all)
  --definition <name|path>  Tournament definition dir or committed definition name
  --config <path>           Core tournament config file/dir (required for all/generate)
  --target <path>           Existing run dir, runtime config, state dir, or DB target (required for run/resume/add-model/export)
  --launch-map <path>       LUMI launch-map YAML/JSON file/dir (required for all/generate/run/resume/add-model)
  --contestant-model <name> Override the committed contestant roster (repeatable; all/generate only)
  --judge-model <name>      Override the committed judge model (all/generate only)
  --model <name>            Contestant model name to add or generate/regenerate (repeatable)
  --max-batches <n>         Forwarded to tournament run/resume
  --max-total-matches <n>   Persist a new max_total_matches for an existing run before resume/add-model
  --contestant-port <n>     Port for per-contestant vLLM server (default: 8000)
  --judge-port <n>          Port for judge vLLM server (default: 8001)
  --nodes <n>               Slurm node count override (default: auto from launch-map for non-export phases)
  --run-label <label>       Override run label for new all/generate runs
  --openai-base-url <url>   Export DFM_EVALS_OPENAI_BASE_URL into the job env
  --eee-output-dir <path>   EEE root data dir override (default: $POST_ARTIFACT_ROOT/evals/eee/data)
  --export-eee             Export tournament EEE in export/all phase (default)
  --no-export-eee          Skip tournament EEE export
  --slurm-log-dir <path>    Slurm stdout/err directory (default: $POST_ARTIFACT_ROOT/evals/slurm)
  --list-definitions        List committed tournament definitions and exit
  --script <path>           sbatch script to submit
  --dry-run                 Print sbatch command/env and exit
  --help                    Show help

Examples:
  ./lumi/tournament_submit.sh --phase all --definition creative-writing-da-smoke
  ./lumi/tournament_submit.sh --phase all --definition creative-writing-da-smoke --contestant-model vllm/google/gemma-3-4b-it --contestant-model vllm/google-gemma-3-4b-pt-hermes-final --judge-model openai/qwen-235b
  ./lumi/tournament_submit.sh --phase run --target "$POST_ARTIFACT_ROOT/evals/runs/tournament__demo__job-123" --launch-map ./configs/tournaments/creative-writing-da-smoke
  ./lumi/tournament_submit.sh --phase add-model --target "$POST_ARTIFACT_ROOT/evals/runs/tournament__demo__job-123" --launch-map ./configs/tournaments/creative-writing-da-smoke --model vllm/google/gemma-3-12b-it
  ./lumi/tournament_submit.sh --phase export --target "$POST_ARTIFACT_ROOT/evals/runs/tournament__demo__job-123"
EOF
}

die() {
  echo "FATAL: $*" >&2
  exit 1
}

select_definition_file() {
  local directory="$1"
  local kind="$2"
  local filename=""
  local -a candidates=()

  case "$kind" in
    config)
      candidates=(tournament.yaml tournament.yml tournament.json)
      ;;
    launch_map)
      candidates=(launch-map.yaml launch-map.yml launch-map.json lumi.yaml lumi.yml lumi.json)
      ;;
    *)
      die "unsupported definition kind: $kind"
      ;;
  esac

  for filename in "${candidates[@]}"; do
    if [[ -f "$directory/$filename" ]]; then
      printf '%s' "$directory/$filename"
      return 0
    fi
  done

  return 1
}

eval_shell_assignments() {
  local output="$1"
  eval "$output"
}

compute_required_nodes() {
  local source="$1"
  local stateful="$2"
  local output=""
  local -a cmd=(
    uv
    run
    --no-sync
    python
    "$REPO_ROOT/lumi/tournament_launch.py"
    emit-resource-shell
    --source "$source"
    --launch-map "$LAUNCH_MAP"
    --phase "$PHASE"
  )
  local model_name=""

  if [[ "$stateful" == "1" ]]; then
    cmd+=(--stateful)
  fi
  for model_name in "${GENERATE_MODELS[@]}"; do
    cmd+=(--model "$model_name")
  done

  output="$(cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}")" || die "failed to resolve required nodes"
  eval_shell_assignments "$output"
  printf '%s' "${RESOURCE_MAX_NODES:-1}"
}

resolve_definition_input() {
  local value="$1"
  local kind="$2"
  local candidate=""

  if [[ -e "$value" ]]; then
    candidate="$value"
  elif [[ -e "$REPO_ROOT/$value" ]]; then
    candidate="$REPO_ROOT/$value"
  elif [[ -e "$TOURNAMENT_DEFINITIONS_DIR/$value" ]]; then
    candidate="$TOURNAMENT_DEFINITIONS_DIR/$value"
  else
    candidate="$TOURNAMENT_DEFINITIONS_DIR/$value"
  fi

  if [[ -d "$candidate" ]]; then
    candidate="$(select_definition_file "$candidate" "$kind")" || die "no $kind file found under definition directory: $candidate"
  fi

  [[ -f "$candidate" ]] || die "tournament $kind not found: $value"
  printf '%s' "$candidate"
}

list_definitions() {
  local path=""

  if [[ ! -d "$TOURNAMENT_DEFINITIONS_DIR" ]]; then
    return 0
  fi

  while IFS= read -r path; do
    printf '%s\n' "$(basename "$path")"
  done < <(find "$TOURNAMENT_DEFINITIONS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
}

need_value() {
  local opt="$1"
  local remaining="$2"
  if [[ "$remaining" -lt 2 ]]; then
    die "missing value for $opt"
  fi
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      need_value "$1" "$#"
      PHASE="$2"
      shift 2
      ;;
    --definition)
      need_value "$1" "$#"
      DEFINITION="$2"
      shift 2
      ;;
    --config)
      need_value "$1" "$#"
      CONFIG="$2"
      shift 2
      ;;
    --target)
      need_value "$1" "$#"
      TARGET="$2"
      shift 2
      ;;
    --launch-map)
      need_value "$1" "$#"
      LAUNCH_MAP="$2"
      shift 2
      ;;
    --contestant-model)
      need_value "$1" "$#"
      CONTESTANT_MODEL_OVERRIDES+=("$2")
      shift 2
      ;;
    --judge-model)
      need_value "$1" "$#"
      JUDGE_MODEL_OVERRIDE="$2"
      shift 2
      ;;
    --model)
      need_value "$1" "$#"
      GENERATE_MODELS+=("$2")
      shift 2
      ;;
    --max-batches)
      need_value "$1" "$#"
      MAX_BATCHES="$2"
      shift 2
      ;;
    --max-total-matches)
      need_value "$1" "$#"
      MAX_TOTAL_MATCHES_OVERRIDE="$2"
      shift 2
      ;;
    --contestant-port)
      need_value "$1" "$#"
      CONTESTANT_PORT="$2"
      shift 2
      ;;
    --judge-port)
      need_value "$1" "$#"
      JUDGE_PORT="$2"
      shift 2
      ;;
    --nodes)
      need_value "$1" "$#"
      NODES_OVERRIDE="$2"
      shift 2
      ;;
    --run-label)
      need_value "$1" "$#"
      RUN_LABEL="$2"
      shift 2
      ;;
    --openai-base-url)
      need_value "$1" "$#"
      OPENAI_BASE_URL_OVERRIDE="$2"
      shift 2
      ;;
    --eee-output-dir)
      need_value "$1" "$#"
      DFM_EVALS_EEE_OUTPUT_DIR="$2"
      shift 2
      ;;
    --export-eee)
      DFM_TOURNAMENT_EXPORT_EEE=1
      shift
      ;;
    --no-export-eee)
      DFM_TOURNAMENT_EXPORT_EEE=0
      shift
      ;;
    --slurm-log-dir)
      need_value "$1" "$#"
      SLURM_LOG_DIR="$2"
      shift 2
      ;;
    --list-definitions)
      LIST_DEFINITIONS=1
      shift
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

if [[ "$LIST_DEFINITIONS" == "1" ]]; then
  list_definitions
  exit 0
fi

case "$PHASE" in
  all|generate|run|resume|add-model|export)
    ;;
  *)
    die "invalid --phase: $PHASE (expected all|generate|run|resume|add-model|export)"
    ;;
esac

if [[ -n "$DEFINITION" && "$PHASE" != "all" && "$PHASE" != "generate" ]]; then
  die "--definition is only supported for new all/generate runs"
fi

if [[ -n "$DEFINITION" ]]; then
  if [[ -z "$CONFIG" ]]; then
    CONFIG="$DEFINITION"
  fi
  if [[ -z "$LAUNCH_MAP" ]]; then
    LAUNCH_MAP="$DEFINITION"
  fi
fi

if [[ -z "$OPENAI_BASE_URL_OVERRIDE" && -n "${OPENAI_BASE_URL:-}" ]]; then
  OPENAI_BASE_URL_OVERRIDE="$OPENAI_BASE_URL"
fi

if [[ "$PHASE" == "all" || "$PHASE" == "generate" ]]; then
  [[ -n "$CONFIG" ]] || die "--config is required for phase $PHASE"
fi

if [[ "$PHASE" == "run" || "$PHASE" == "resume" || "$PHASE" == "add-model" || "$PHASE" == "export" ]]; then
  [[ -n "$TARGET" ]] || die "--target is required for phase $PHASE"
fi

if [[ "$PHASE" != "export" ]]; then
  [[ -n "$LAUNCH_MAP" ]] || die "--launch-map is required for phase $PHASE"
fi

if [[ "$PHASE" == "export" && "${#GENERATE_MODELS[@]}" -gt 0 ]]; then
  die "--model is not supported for --phase export"
fi

if [[ "$PHASE" == "add-model" && "${#GENERATE_MODELS[@]}" -eq 0 ]]; then
  die "--model is required for --phase add-model"
fi

if [[ -n "$MAX_TOTAL_MATCHES_OVERRIDE" && "$PHASE" != "run" && "$PHASE" != "resume" && "$PHASE" != "add-model" ]]; then
  die "--max-total-matches is only supported for run/resume/add-model"
fi

if [[ -n "$NODES_OVERRIDE" ]]; then
  if ! [[ "$NODES_OVERRIDE" =~ ^[0-9]+$ ]] || [[ "$NODES_OVERRIDE" -le 0 ]]; then
    die "invalid --nodes: $NODES_OVERRIDE"
  fi
fi

if [[ ("$PHASE" != "all" && "$PHASE" != "generate") && "${#CONTESTANT_MODEL_OVERRIDES[@]}" -gt 0 ]]; then
  die "--contestant-model is only supported for new all/generate runs"
fi

if [[ -n "$JUDGE_MODEL_OVERRIDE" && "$PHASE" != "all" && "$PHASE" != "generate" ]]; then
  die "--judge-model is only supported for new all/generate runs"
fi

if [[ -n "$RUN_LABEL" && "$PHASE" != "all" && "$PHASE" != "generate" ]]; then
  die "--run-label is only supported for new all/generate runs"
fi

[[ -f "$SUBMIT_SCRIPT" ]] || die "submit script not found: $SUBMIT_SCRIPT"
[[ -d "$OVERLAY_DIR" ]] || die "overlay dir not found: $OVERLAY_DIR"
mkdir -p "$SLURM_LOG_DIR"

if [[ -n "$CONFIG" ]]; then
  CONFIG="$(resolve_definition_input "$CONFIG" config)"
fi
if [[ -n "$LAUNCH_MAP" ]]; then
  LAUNCH_MAP="$(resolve_definition_input "$LAUNCH_MAP" launch_map)"
fi

default_label_base=""
if [[ "$PHASE" == "all" || "$PHASE" == "generate" ]]; then
  default_label_base="tournament__$(label_from_path "$CONFIG")"
else
  default_label_base="$(label_from_path "$TARGET")"
fi
[[ -n "$default_label_base" ]] || default_label_base="tournament"

if [[ -n "$RUN_LABEL" ]]; then
  raw_slurm_log_label="$RUN_LABEL"
else
  raw_slurm_log_label="$default_label_base"
fi

slurm_log_label="$(sanitize_label "$raw_slurm_log_label")"
sanitized_run_label=""
if [[ -n "$RUN_LABEL" ]]; then
  sanitized_run_label="$(sanitize_label "$RUN_LABEL")"
fi
slurm_out_path="${SLURM_LOG_DIR}/${slurm_log_label}-%j.out"
slurm_err_path="${SLURM_LOG_DIR}/${slurm_log_label}-%j.err"
required_nodes="1"

if [[ "$PHASE" != "export" ]]; then
  if [[ -n "$NODES_OVERRIDE" ]]; then
    required_nodes="$NODES_OVERRIDE"
  elif [[ "$PHASE" == "all" || "$PHASE" == "generate" ]]; then
    required_nodes="$(compute_required_nodes "$CONFIG" 0)"
  else
    required_nodes="$(compute_required_nodes "$TARGET" 1)"
  fi
fi

env_kv=(
  "DFM_EVALS_REPO_ROOT=$REPO_ROOT"
  "DFM_EVALS_RUN_ROOT=$DFM_EVALS_RUN_ROOT"
  "DFM_TOURNAMENT_PHASE=$PHASE"
  "DFM_TOURNAMENT_CONTESTANT_PORT=$CONTESTANT_PORT"
  "DFM_TOURNAMENT_JUDGE_PORT=$JUDGE_PORT"
  "DFM_TOURNAMENT_EXPORT_EEE=$DFM_TOURNAMENT_EXPORT_EEE"
  "DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME=$DFM_EVALS_EEE_SOURCE_ORGANIZATION_NAME"
  "DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP=$DFM_EVALS_EEE_EVALUATOR_RELATIONSHIP"
)
if [[ -n "$CONFIG" ]]; then
  env_kv+=("DFM_TOURNAMENT_CONFIG=$CONFIG")
fi
if [[ -n "$TARGET" ]]; then
  env_kv+=("DFM_TOURNAMENT_TARGET=$TARGET")
fi
if [[ -n "$LAUNCH_MAP" ]]; then
  env_kv+=("DFM_TOURNAMENT_LAUNCH_MAP=$LAUNCH_MAP")
fi
if [[ -n "$RUN_LABEL" ]]; then
  env_kv+=("DFM_EVALS_RUN_LABEL=$RUN_LABEL")
fi
if [[ -n "$MAX_BATCHES" ]]; then
  env_kv+=("DFM_TOURNAMENT_MAX_BATCHES=$MAX_BATCHES")
fi
if [[ -n "$MAX_TOTAL_MATCHES_OVERRIDE" ]]; then
  env_kv+=("DFM_TOURNAMENT_MAX_TOTAL_MATCHES=$MAX_TOTAL_MATCHES_OVERRIDE")
fi
if [[ "${#CONTESTANT_MODEL_OVERRIDES[@]}" -gt 0 ]]; then
  contestant_models_csv="$(IFS=,; echo "${CONTESTANT_MODEL_OVERRIDES[*]}")"
  env_kv+=("DFM_TOURNAMENT_CONTESTANT_MODELS=$contestant_models_csv")
fi
if [[ -n "$JUDGE_MODEL_OVERRIDE" ]]; then
  env_kv+=("DFM_TOURNAMENT_JUDGE_MODEL=$JUDGE_MODEL_OVERRIDE")
fi
if [[ -n "$OPENAI_BASE_URL_OVERRIDE" ]]; then
  env_kv+=("DFM_EVALS_OPENAI_BASE_URL=$OPENAI_BASE_URL_OVERRIDE")
fi
if [[ -n "$OPENAI_API_KEY_OVERRIDE" ]]; then
  env_kv+=("DFM_EVALS_OPENAI_API_KEY=$OPENAI_API_KEY_OVERRIDE")
fi
if [[ -n "$DFM_EVALS_EEE_OUTPUT_DIR" ]]; then
  env_kv+=("DFM_EVALS_EEE_OUTPUT_DIR=$DFM_EVALS_EEE_OUTPUT_DIR")
fi
if [[ "${#GENERATE_MODELS[@]}" -gt 0 ]]; then
  generate_models_csv="$(IFS=,; echo "${GENERATE_MODELS[*]}")"
  if [[ "$PHASE" == "add-model" ]]; then
    env_kv+=("DFM_TOURNAMENT_ADDED_MODELS=$generate_models_csv")
  else
    env_kv+=("DFM_TOURNAMENT_GENERATE_MODELS=$generate_models_csv")
  fi
fi

echo "Submit script: $SUBMIT_SCRIPT"
echo "Phase: $PHASE"
if [[ -n "$CONFIG" ]]; then
  echo "Config: $CONFIG"
fi
if [[ -n "$TARGET" ]]; then
  echo "Target: $TARGET"
fi
if [[ -n "$LAUNCH_MAP" ]]; then
  echo "Launch map: $LAUNCH_MAP"
fi
if [[ "${#CONTESTANT_MODEL_OVERRIDES[@]}" -gt 0 ]]; then
  echo "Contestant model overrides: ${CONTESTANT_MODEL_OVERRIDES[*]}"
fi
if [[ -n "$JUDGE_MODEL_OVERRIDE" ]]; then
  echo "Judge model override: $JUDGE_MODEL_OVERRIDE"
fi
echo "Contestant port: $CONTESTANT_PORT"
echo "Judge port: $JUDGE_PORT"
echo "Slurm nodes: $required_nodes"
echo "Export EEE: $DFM_TOURNAMENT_EXPORT_EEE"
echo "Eval run root: $DFM_EVALS_RUN_ROOT"
echo "Slurm stdout path pattern: $slurm_out_path"
echo "Slurm stderr path pattern: $slurm_err_path"
if [[ -n "$RUN_LABEL" ]]; then
  echo "Run label override: $RUN_LABEL"
  if [[ "$sanitized_run_label" != "$RUN_LABEL" ]]; then
    echo "Sanitized run label: $sanitized_run_label"
  fi
else
  echo "Run label default base: $default_label_base"
fi
if [[ -n "$MAX_BATCHES" ]]; then
  echo "Max batches: $MAX_BATCHES"
fi
if [[ -n "$MAX_TOTAL_MATCHES_OVERRIDE" ]]; then
  echo "Max total matches override: $MAX_TOTAL_MATCHES_OVERRIDE"
fi
if [[ -n "$NODES_OVERRIDE" ]]; then
  echo "Nodes override: $NODES_OVERRIDE"
fi
if [[ "${#GENERATE_MODELS[@]}" -gt 0 ]]; then
  if [[ "$PHASE" == "add-model" ]]; then
    echo "Add models: ${GENERATE_MODELS[*]}"
  else
    echo "Generate models: ${GENERATE_MODELS[*]}"
  fi
fi
if [[ -n "$DFM_EVALS_EEE_OUTPUT_DIR" ]]; then
  echo "EEE output dir override: $DFM_EVALS_EEE_OUTPUT_DIR"
fi
if [[ -n "$OPENAI_BASE_URL_OVERRIDE" ]]; then
  echo "OpenAI base URL override: $OPENAI_BASE_URL_OVERRIDE"
fi
if [[ -n "$OPENAI_API_KEY_OVERRIDE" ]]; then
  echo "OpenAI API key override: [set]"
fi

redact_sensitive_env() {
  local value="$1"
  case "$value" in
    DFM_EVALS_OPENAI_API_KEY=*)
      printf '%s' 'DFM_EVALS_OPENAI_API_KEY=[redacted]'
      ;;
    *)
      printf '%s' "$value"
      ;;
  esac
}

cmd=(env "${env_kv[@]}" sbatch --nodes "$required_nodes" --output "$slurm_out_path" --error "$slurm_err_path" "$SUBMIT_SCRIPT")
if [[ "$DRY_RUN" == "1" ]]; then
  local_cmd=()
  for item in "${cmd[@]}"; do
    local_cmd+=("$(redact_sensitive_env "$item")")
  done
  printf 'Dry run command: '
  printf '(cd %q && ' "$REPO_ROOT"
  printf '%q ' "${local_cmd[@]}"
  printf ')'
  echo
  exit 0
fi

submit_out="$(cd "$REPO_ROOT" && "${cmd[@]}")"
echo "$submit_out"

job_id="$(awk '/Submitted batch job/{print $4}' <<<"$submit_out")"
if [[ -n "$job_id" ]]; then
  if [[ -n "$RUN_LABEL" ]]; then
    effective_label="$sanitized_run_label"
  elif [[ "$PHASE" == "all" || "$PHASE" == "generate" ]]; then
    raw_label="${default_label_base}__job-${job_id}"
    effective_label="$(sanitize_label "$raw_label")"
  else
    effective_label="$default_label_base"
  fi
  run_dir="$DFM_EVALS_RUN_ROOT/$effective_label"
  launcher_dir="$run_dir/launcher"
  resolved_slurm_out="${slurm_out_path//%j/$job_id}"
  resolved_slurm_err="${slurm_err_path//%j/$job_id}"
  mkdir -p "$launcher_dir"
  ln -sfn "$resolved_slurm_out" "$launcher_dir/slurm.out"
  ln -sfn "$resolved_slurm_err" "$launcher_dir/slurm.err"
  echo "Job id: $job_id"
  echo "Expected run label: $effective_label"
  echo "Expected host run dir: $run_dir"
  echo "Expected runtime config: $run_dir/config/runtime.json"
  echo "Expected inspect logs: $run_dir/inspect"
  echo "Expected vLLM server logs: $run_dir/services/vllm"
  echo "Expected tournament state dir: $run_dir/state"
  echo "Launcher symlinks: $launcher_dir/slurm.out, $launcher_dir/slurm.err"
  echo "Slurm stdout: $resolved_slurm_out"
  echo "Slurm stderr: $resolved_slurm_err"
  echo "View this run: ./lumi/view.sh start --job-id $job_id"
fi
