#!/bin/bash
# Launch inspect view inside the overlay container for dfm-evals logs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
SIF=${SIF:-$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif}
OVERLAY_DIR=${OVERLAY_DIR:-$REPO_ROOT/overlay_vllm_minimal}
if [[ ! -d "$OVERLAY_DIR" && -d "$REPO_ROOT/../overlay_vllm_minimal" ]]; then
  OVERLAY_DIR="$REPO_ROOT/../overlay_vllm_minimal"
fi

EVAL_LOG_ROOT_HOST=${EVAL_LOG_ROOT_HOST:-${OVERLAY_LOG_ROOT_HOST:-$REPO_ROOT/logs/evals-logs}}
EVAL_LOG_ROOT_CONTAINER=${EVAL_LOG_ROOT_CONTAINER:-${OVERLAY_LOG_ROOT_CONTAINER:-$EVAL_LOG_ROOT_HOST}}
SLURM_LOG_DIR_HOST=${SLURM_LOG_DIR_HOST:-${SLURM_LOG_DIR:-$REPO_ROOT/logs/slurm}}

VIEW_MODE=start
if [[ $# -gt 0 ]]; then
  case "$1" in
    start|bundle|list)
      VIEW_MODE="$1"
      shift
      ;;
  esac
fi

VIEW_SELECTOR=${VIEW_SELECTOR:-all}
VIEW_SELECTOR_VALUE=${VIEW_SELECTOR_VALUE:-}
VIEW_LOG_DIR=${VIEW_LOG_DIR:-}
VIEW_HOST=${VIEW_HOST:-127.0.0.1}
VIEW_PORT=${VIEW_PORT:-7575}
VIEW_RECURSIVE=${VIEW_RECURSIVE:-1}
VIEW_OUTPUT_DIR=${VIEW_OUTPUT_DIR:-$REPO_ROOT/view-bundle}
VIEW_OVERWRITE=${VIEW_OVERWRITE:-1}
VIEW_RUN_LABEL=${VIEW_RUN_LABEL:-}
VIEW_XDG_CACHE_HOME=${VIEW_XDG_CACHE_HOME:-}
VIEW_XDG_DATA_HOME=${VIEW_XDG_DATA_HOME:-}
HF_HOME=${HF_HOME:-/flash/project_465002183/.cache/huggingface/}
PASSTHROUGH_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./lumi/view.sh [start] [selector options] [-- inspect view start args...]
  ./lumi/view.sh bundle [selector options] [-- inspect view bundle args...]
  ./lumi/view.sh list

Selector options:
  --all                Use all runs (default, logs/evals-logs)
  --latest             Use latest run dir under eval logs
  --job-id <id>        Use run dir matching *__job-<id>, <id>, or
                       slurm label file *-<id>.out/.err
  --label <run_label>  Use <eval-log-root>/<run_label>
  --log-dir <path>     Use explicit log dir path:
                       - host path under eval log root (auto-mapped)
                       - /overlay/... path
                       - host path under OVERLAY_DIR (auto-mapped to /overlay)
                       - absolute host path (auto bind-mounted)
                       - bare name treated as run label

General options:
  --host <ip>          Viewer bind host (default: 127.0.0.1)
  --port <port>        Viewer port (default: 7575)
  --recursive          Include logs recursively (default on)
  --no-recursive       Disable recursive scan
  --output-dir <path>  Bundle output dir (bundle mode)
  --overwrite          Overwrite bundle dir (default on)
  --no-overwrite       Disable overwrite

Environment overrides:
  VIEW_SELECTOR, VIEW_SELECTOR_VALUE, VIEW_LOG_DIR
  VIEW_HOST, VIEW_PORT, VIEW_RECURSIVE
  VIEW_OUTPUT_DIR, VIEW_OVERWRITE
  VIEW_RUN_LABEL, VIEW_XDG_CACHE_HOME, VIEW_XDG_DATA_HOME
  EVAL_LOG_ROOT_HOST, EVAL_LOG_ROOT_CONTAINER
  SLURM_LOG_DIR_HOST (used by --job-id fallback resolution)
  OVERLAY_LOG_ROOT_HOST, OVERLAY_LOG_ROOT_CONTAINER (legacy aliases)
  SIF             (override singularity image path)
  OVERLAY_DIR     (override overlay dir)

Examples:
  ./lumi/view.sh list
  ./lumi/view.sh start --latest
  ./lumi/view.sh start --job-id 16316016
  ./lumi/view.sh start --label fundamentals__vllm_google_gemma-3-4b-it__job-16316016
  ./lumi/view.sh start --job-id 16636687  # auto-loads XDG data/cache for live traces
  ./lumi/view.sh bundle --latest --output-dir ./view-bundle-latest
EOF
}

die() {
  echo "FATAL: $*" >&2
  exit 1
}

need_option_value() {
  local opt="$1"
  local remaining="$2"
  if [[ "$remaining" -lt 2 ]]; then
    die "missing value for $opt"
  fi
}

list_runs() {
  [[ -d "$EVAL_LOG_ROOT_HOST" ]] || die "eval logs root not found: $EVAL_LOG_ROOT_HOST"
  mapfile -t entries < <(
    find "$EVAL_LOG_ROOT_HOST" -mindepth 1 -maxdepth 1 -type d -printf '%T@|%f\n' \
      | sort -t'|' -k1,1nr
  )
  echo "Available runs under $EVAL_LOG_ROOT_HOST (newest first):"
  if [[ "${#entries[@]}" -eq 0 ]]; then
    echo "  (none)"
    return
  fi
  for entry in "${entries[@]}"; do
    local ts_epoch="${entry%%|*}"
    local run_name="${entry#*|}"
    local ts_text
    ts_text="$(date -d "@${ts_epoch%.*}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || printf '%s' "$ts_epoch")"
    printf '  %s  %s\n' "$ts_text" "$run_name"
  done
}

run_dir_for_latest() {
  [[ -d "$EVAL_LOG_ROOT_HOST" ]] || die "eval logs root not found: $EVAL_LOG_ROOT_HOST"
  local newest
  newest="$(
    find "$EVAL_LOG_ROOT_HOST" -mindepth 1 -maxdepth 1 -type d -printf '%T@|%f\n' \
      | sort -t'|' -k1,1nr \
      | head -n 1 \
      | cut -d'|' -f2
  )"
  [[ -n "$newest" ]] || die "no run directories found under $EVAL_LOG_ROOT_HOST"
  printf '%s' "$newest"
}

run_dir_for_job_id() {
  local job_id="$1"
  [[ -d "$EVAL_LOG_ROOT_HOST" ]] || die "eval logs root not found: $EVAL_LOG_ROOT_HOST"
  local match
  match="$(
    find "$EVAL_LOG_ROOT_HOST" -mindepth 1 -maxdepth 1 -type d -name "*__job-${job_id}" -printf '%T@|%f\n' \
      | sort -t'|' -k1,1nr \
      | head -n 1 \
      | cut -d'|' -f2
  )"
  if [[ -z "$match" ]]; then
    match="$(
      find "$EVAL_LOG_ROOT_HOST" -mindepth 1 -maxdepth 1 -type d -name "${job_id}" -printf '%T@|%f\n' \
        | sort -t'|' -k1,1nr \
        | head -n 1 \
        | cut -d'|' -f2
    )"
  fi
  if [[ -z "$match" && -d "$SLURM_LOG_DIR_HOST" ]]; then
    local slurm_match
    slurm_match="$(
      find "$SLURM_LOG_DIR_HOST" -mindepth 1 -maxdepth 1 -type f \( -name "*-${job_id}.out" -o -name "*-${job_id}.err" \) -printf '%T@|%f\n' \
        | sort -t'|' -k1,1nr \
        | head -n 1 \
        | cut -d'|' -f2
    )"
    if [[ -n "$slurm_match" ]]; then
      local run_label="$slurm_match"
      run_label="${run_label%-${job_id}.out}"
      run_label="${run_label%-${job_id}.err}"
      if [[ -d "$EVAL_LOG_ROOT_HOST/$run_label" ]]; then
        match="$run_label"
      fi
    fi
  fi
  [[ -n "$match" ]] || die "no run directory found for job id $job_id under $EVAL_LOG_ROOT_HOST"
  printf '%s' "$match"
}

slurm_stdout_for_run_label() {
  local run_label="$1"
  [[ -d "$SLURM_LOG_DIR_HOST" ]] || return 0
  find "$SLURM_LOG_DIR_HOST" -mindepth 1 -maxdepth 1 -type f -name "${run_label}-*.out" -printf '%T@|%p\n' \
    | sort -t'|' -k1,1nr \
    | head -n 1 \
    | cut -d'|' -f2-
}

extract_logged_value() {
  local log_file="$1"
  local prefix="$2"
  [[ -f "$log_file" ]] || return 0
  grep -F "$prefix" "$log_file" | tail -n 1 | sed "s|^.*${prefix}||"
}

path_to_container_log_dir() {
  local input_path="$1"
  if [[ "$input_path" == "$EVAL_LOG_ROOT_HOST" ]]; then
    printf '%s' "$EVAL_LOG_ROOT_CONTAINER"
    return
  fi
  if [[ "$input_path" == "$EVAL_LOG_ROOT_HOST/"* ]]; then
    printf '%s/%s' "$EVAL_LOG_ROOT_CONTAINER" "${input_path#$EVAL_LOG_ROOT_HOST/}"
    return
  fi
  if [[ "$input_path" == /overlay/* ]]; then
    printf '%s' "$input_path"
    return
  fi
  if [[ "$input_path" == "$OVERLAY_DIR" ]]; then
    printf '/overlay'
    return
  fi
  if [[ "$input_path" == "$OVERLAY_DIR/"* ]]; then
    printf '/overlay/%s' "${input_path#$OVERLAY_DIR/}"
    return
  fi
  if [[ "$input_path" == "$REPO_ROOT" || "$input_path" == "$REPO_ROOT/"* ]]; then
    printf '%s' "$input_path"
    return
  fi
  if [[ "$input_path" == /* ]]; then
    [[ -d "$input_path" ]] || die "host log dir not found: $input_path"
    EXTRA_BINDS+=("-B" "$input_path:$input_path")
    printf '%s' "$input_path"
    return
  fi
  # Bare name: interpret as a run label under eval log root.
  local candidate="$EVAL_LOG_ROOT_HOST/$input_path"
  [[ -d "$candidate" ]] || die "run label not found under eval logs: $input_path"
  printf '%s/%s' "$EVAL_LOG_ROOT_CONTAINER" "$input_path"
}

resolve_view_log_dir() {
  if [[ -n "$VIEW_LOG_DIR" ]]; then
    VIEW_LOG_DIR="$(path_to_container_log_dir "$VIEW_LOG_DIR")"
    return
  fi
  case "$VIEW_SELECTOR" in
    all)
      VIEW_LOG_DIR="$EVAL_LOG_ROOT_CONTAINER"
      ;;
    latest)
      VIEW_LOG_DIR="$EVAL_LOG_ROOT_CONTAINER/$(run_dir_for_latest)"
      ;;
    job-id)
      [[ -n "$VIEW_SELECTOR_VALUE" ]] || die "--job-id requires a value"
      VIEW_LOG_DIR="$EVAL_LOG_ROOT_CONTAINER/$(run_dir_for_job_id "$VIEW_SELECTOR_VALUE")"
      ;;
    label)
      [[ -n "$VIEW_SELECTOR_VALUE" ]] || die "--label requires a value"
      [[ -d "$EVAL_LOG_ROOT_HOST/$VIEW_SELECTOR_VALUE" ]] || die "run label not found: $VIEW_SELECTOR_VALUE"
      VIEW_LOG_DIR="$EVAL_LOG_ROOT_CONTAINER/$VIEW_SELECTOR_VALUE"
      ;;
    log-dir)
      [[ -n "$VIEW_SELECTOR_VALUE" ]] || die "--log-dir requires a value"
      VIEW_LOG_DIR="$(path_to_container_log_dir "$VIEW_SELECTOR_VALUE")"
      ;;
    *)
      die "unknown view selector: $VIEW_SELECTOR"
      ;;
  esac
}

resolve_view_run_label() {
  if [[ -n "$VIEW_RUN_LABEL" ]]; then
    return
  fi
  if [[ -n "$VIEW_LOG_DIR" && "$VIEW_LOG_DIR" != "$EVAL_LOG_ROOT_CONTAINER" ]]; then
    VIEW_RUN_LABEL="$(basename "$VIEW_LOG_DIR")"
    return
  fi
  case "$VIEW_SELECTOR" in
    latest)
      VIEW_RUN_LABEL="$(run_dir_for_latest)"
      ;;
    job-id)
      [[ -n "$VIEW_SELECTOR_VALUE" ]] || return
      VIEW_RUN_LABEL="$(run_dir_for_job_id "$VIEW_SELECTOR_VALUE")"
      ;;
    label)
      VIEW_RUN_LABEL="$VIEW_SELECTOR_VALUE"
      ;;
    log-dir)
      if [[ -n "$VIEW_SELECTOR_VALUE" && "$VIEW_SELECTOR_VALUE" != /* ]]; then
        VIEW_RUN_LABEL="$VIEW_SELECTOR_VALUE"
      elif [[ -n "$VIEW_LOG_DIR" && "$VIEW_LOG_DIR" != "$EVAL_LOG_ROOT_CONTAINER" ]]; then
        VIEW_RUN_LABEL="$(basename "$VIEW_LOG_DIR")"
      fi
      ;;
  esac
}

resolve_view_xdg_homes() {
  [[ "$VIEW_MODE" == "start" ]] || return
  resolve_view_run_label
  [[ -n "$VIEW_RUN_LABEL" ]] || return

  if [[ -n "$VIEW_XDG_CACHE_HOME" && -n "$VIEW_XDG_DATA_HOME" ]]; then
    return
  fi

  local slurm_stdout=""
  slurm_stdout="$(slurm_stdout_for_run_label "$VIEW_RUN_LABEL" || true)"
  if [[ -n "$slurm_stdout" ]]; then
    if [[ -z "$VIEW_XDG_CACHE_HOME" ]]; then
      VIEW_XDG_CACHE_HOME="$(extract_logged_value "$slurm_stdout" "XDG cache home: " || true)"
    fi
    if [[ -z "$VIEW_XDG_DATA_HOME" ]]; then
      VIEW_XDG_DATA_HOME="$(extract_logged_value "$slurm_stdout" "XDG data home: " || true)"
    fi
  fi

  if [[ -z "$VIEW_XDG_CACHE_HOME" ]]; then
    VIEW_XDG_CACHE_HOME="/overlay/cache/dfm-evals-${VIEW_RUN_LABEL}/xdg-cache"
  fi
  if [[ -z "$VIEW_XDG_DATA_HOME" ]]; then
    VIEW_XDG_DATA_HOME="/overlay/cache/dfm-evals-${VIEW_RUN_LABEL}/xdg-data"
  fi
}

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --latest)
        VIEW_SELECTOR=latest
        VIEW_SELECTOR_VALUE=
        shift
        ;;
      --job-id)
        need_option_value "--job-id" "$#"
        VIEW_SELECTOR=job-id
        VIEW_SELECTOR_VALUE="$2"
        shift 2
        ;;
      --label)
        need_option_value "--label" "$#"
        VIEW_SELECTOR=label
        VIEW_SELECTOR_VALUE="$2"
        shift 2
        ;;
      --log-dir)
        need_option_value "--log-dir" "$#"
        VIEW_SELECTOR=log-dir
        VIEW_SELECTOR_VALUE="$2"
        shift 2
        ;;
      --all)
        VIEW_SELECTOR=all
        VIEW_SELECTOR_VALUE=
        shift
        ;;
      --host)
        need_option_value "--host" "$#"
        VIEW_HOST="$2"
        shift 2
        ;;
      --port)
        need_option_value "--port" "$#"
        VIEW_PORT="$2"
        shift 2
        ;;
      --output-dir)
        need_option_value "--output-dir" "$#"
        VIEW_OUTPUT_DIR="$2"
        shift 2
        ;;
      --recursive)
        VIEW_RECURSIVE=1
        shift
        ;;
      --no-recursive)
        VIEW_RECURSIVE=0
        shift
        ;;
      --overwrite)
        VIEW_OVERWRITE=1
        shift
        ;;
      --no-overwrite)
        VIEW_OVERWRITE=0
        shift
        ;;
      --list)
        VIEW_MODE=list
        shift
        ;;
      --help|-h|help)
        usage
        exit 0
        ;;
      --)
        shift
        PASSTHROUGH_ARGS+=("$@")
        break
        ;;
      *)
        PASSTHROUGH_ARGS+=("$1")
        shift
        ;;
    esac
  done
fi

[[ -f "$SIF" ]] || die "SIF not found: $SIF"
[[ -d "$OVERLAY_DIR/venv/vllm-min" ]] || die "overlay venv missing: $OVERLAY_DIR/venv/vllm-min"

if [[ "$VIEW_MODE" == "list" ]]; then
  list_runs
  exit 0
fi

SING_BIND_ARGS=(
  -B "$BASE_DIR:$LAIFS_APPL_DIR"
  -B "$OVERLAY_DIR:/overlay"
  -B "$REPO_ROOT:$REPO_ROOT"
)
EXTRA_BINDS=()
if [[ -d /flash ]]; then
  SING_BIND_ARGS+=(-B /flash:/flash)
fi
if [[ -L /flash/project_465002183 ]]; then
  FLASH_TARGET="$(readlink -f /flash/project_465002183 || true)"
  if [[ -n "$FLASH_TARGET" && -d "$FLASH_TARGET" ]]; then
    SING_BIND_ARGS+=(-B "$FLASH_TARGET:$FLASH_TARGET")
  fi
fi

container_env_prefix() {
  cat <<'EOC'
set -euo pipefail
source /overlay/venv/vllm-min/bin/activate
OVERLAY_SITE=/overlay/venv/vllm-min/lib/python3.12/site-packages
export PYTHONPATH=/overlay/src/transformers/src:/overlay/src/vllm:${OVERLAY_SITE}${PYTHONPATH:+:$PYTHONPATH}
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export HF_HOME="__HF_HOME__"
export HF_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
EOC
}

render_env() {
  container_env_prefix | sed "s|__HF_HOME__|$HF_HOME|g"
}

resolve_view_log_dir
resolve_view_xdg_homes

# If bundle output is outside known bind mounts, bind it explicitly.
if [[ "$VIEW_MODE" == "bundle" ]]; then
  if [[ "$VIEW_OUTPUT_DIR" != /* ]]; then
    VIEW_OUTPUT_DIR="$REPO_ROOT/$VIEW_OUTPUT_DIR"
  fi
  mkdir -p "$VIEW_OUTPUT_DIR"
  if [[ "$VIEW_OUTPUT_DIR" != /overlay/* && "$VIEW_OUTPUT_DIR" != "$REPO_ROOT" && "$VIEW_OUTPUT_DIR" != "$REPO_ROOT/"* ]]; then
    EXTRA_BINDS+=("-B" "$VIEW_OUTPUT_DIR:$VIEW_OUTPUT_DIR")
  fi
fi

if [[ "${#EXTRA_BINDS[@]}" -gt 0 ]]; then
  SING_BIND_ARGS+=("${EXTRA_BINDS[@]}")
fi

inspect_args=()
case "$VIEW_MODE" in
  start)
    inspect_args=(view start --log-dir "$VIEW_LOG_DIR" --host "$VIEW_HOST" --port "$VIEW_PORT")
    if [[ "$VIEW_RECURSIVE" == "1" ]]; then
      inspect_args+=(--recursive)
    fi
    inspect_args+=("${PASSTHROUGH_ARGS[@]}")
    ;;
  bundle)
    mkdir -p "$VIEW_OUTPUT_DIR"
    inspect_args=(view bundle --log-dir "$VIEW_LOG_DIR" --output-dir "$VIEW_OUTPUT_DIR")
    if [[ "$VIEW_OVERWRITE" == "1" ]]; then
      inspect_args+=(--overwrite)
    fi
    inspect_args+=("${PASSTHROUGH_ARGS[@]}")
    ;;
  *)
    usage
    die "unknown mode: $VIEW_MODE"
    ;;
esac

printf -v INSPECT_CMD '%q ' inspect "${inspect_args[@]}"

echo "SIF: $SIF"
echo "Overlay: $OVERLAY_DIR"
echo "Mode: $VIEW_MODE"
echo "Log dir: $VIEW_LOG_DIR"
if [[ -n "$VIEW_RUN_LABEL" ]]; then
  echo "Run label: $VIEW_RUN_LABEL"
fi
if [[ "$VIEW_MODE" == "start" && -n "$VIEW_XDG_DATA_HOME" ]]; then
  echo "XDG cache home: $VIEW_XDG_CACHE_HOME"
  echo "XDG data home: $VIEW_XDG_DATA_HOME"
fi
if [[ "$VIEW_MODE" == "start" ]]; then
  echo "View URL: http://${VIEW_HOST}:${VIEW_PORT}"
fi
if [[ "$VIEW_MODE" == "bundle" ]]; then
  echo "Bundle output dir: $VIEW_OUTPUT_DIR"
fi

RUN_CMD="$(render_env)
if [[ -n \"${VIEW_XDG_CACHE_HOME}\" ]]; then
  export XDG_CACHE_HOME=\"${VIEW_XDG_CACHE_HOME}\"
fi
if [[ -n \"${VIEW_XDG_DATA_HOME}\" ]]; then
  export XDG_DATA_HOME=\"${VIEW_XDG_DATA_HOME}\"
fi
${INSPECT_CMD}"

singularity exec --rocm "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$RUN_CMD"
