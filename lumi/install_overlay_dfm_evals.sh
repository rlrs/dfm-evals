#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
source "$SCRIPT_DIR/laif_runtime.sh"

: "${SIF:=$(dfm_lumi_default_sif)}"
: "${OVERLAY_DIR:=$(dfm_lumi_default_overlay_dir)}"
: "${SINGULARITY_GPU_MODE:=$(dfm_lumi_default_gpu_mode "$SIF")}"

EXTRAS=""
NO_DEPS_SET=0

usage() {
  cat <<'EOF'
Usage:
  ./lumi/install_overlay_dfm_evals.sh [options]

Options:
  --extras <csv>   Install optional extras, e.g. harbor,sandboxes
  --no-deps        Reinstall only dfm-evals itself without dependency changes
  --help           Show this help

Notes:
  - The install is performed inside the same Singularity container and with the
    same repo-path bind that `lumi/run_suite.sbatch` uses.
  - Editable installs therefore point at the host repo path, not at /workspace.
  - Default behavior is:
      * no extras: `pip install -e .`
      * with extras: `pip install -e ".[extras]"`
      * pass `--no-deps` for a fast code-only reinstall into an overlay that
        already has the project dependencies
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
    --extras)
      need_value "$1" "$#"
      EXTRAS="$2"
      shift 2
      ;;
    --no-deps)
      NO_DEPS_SET=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -f "$SIF" ]] || die "SIF not found: $SIF"
[[ -d "$OVERLAY_DIR/venv/vllm-min" ]] || die "overlay venv missing: $OVERLAY_DIR/venv/vllm-min"
dfm_lumi_init_gpu_args

INSTALL_TARGET="."
if [[ -n "$EXTRAS" ]]; then
  INSTALL_TARGET=".[${EXTRAS}]"
fi

REQ_EXPORT_FILE_HOST="$OVERLAY_DIR/runtime/dfm-evals-install.requirements.txt"
REQ_EXPORT_FILE_CONTAINER="/overlay/runtime/dfm-evals-install.requirements.txt"
rm -f "$REQ_EXPORT_FILE_HOST"

if [[ "$NO_DEPS_SET" != "1" ]]; then
  command -v uv >/dev/null 2>&1 || die "uv is required for locked dependency export"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/project_465002183/.cache/uv}"
  mkdir -p "$UV_CACHE_DIR" "$(dirname "$REQ_EXPORT_FILE_HOST")"

  UV_EXPORT_ARGS=(
    export
    --locked
    --no-dev
    --no-hashes
    --no-header
    --no-emit-project
    --format requirements-txt
    --output-file "$REQ_EXPORT_FILE_HOST"
  )

  if [[ -n "$EXTRAS" ]]; then
    IFS=',' read -r -a extras_list <<<"$EXTRAS"
    for extra in "${extras_list[@]}"; do
      [[ -n "$extra" ]] || continue
      UV_EXPORT_ARGS+=(--extra "$extra")
    done
  fi

  (
    cd "$REPO_ROOT"
    uv "${UV_EXPORT_ARGS[@]}"
  )

  python3 - "$REQ_EXPORT_FILE_HOST" <<'PY'
from pathlib import Path
import sys

req_path = Path(sys.argv[1])
preserve = {
    "hf-xet",
    "huggingface-hub",
    "tokenizers",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "triton",
    "triton-rocm",
    "vllm",
}

lines = []
for raw in req_path.read_text(encoding="utf-8").splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        continue
    name = stripped.split(";", 1)[0].split("@", 1)[0].split("==", 1)[0].strip().lower()
    if name in preserve:
        continue
    lines.append(raw)

req_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
fi

printf -v INSTALL_TARGET_Q '%q' "$INSTALL_TARGET"
printf -v REPO_ROOT_Q '%q' "$REPO_ROOT"
printf -v REQ_EXPORT_FILE_Q '%q' "$REQ_EXPORT_FILE_CONTAINER"

INSTALL_CMD="set -euo pipefail
source /overlay/venv/vllm-min/bin/activate
if [[ -f /overlay/overlay-runtime.env ]]; then
  # shellcheck disable=SC1091
  source /overlay/overlay-runtime.env
fi
export PIP_USER=0
unset PYTHONUSERBASE
export XDG_CACHE_HOME=/overlay/cache
export PIP_CACHE_DIR=/overlay/cache/pip
export UV_CACHE_DIR=/overlay/cache/uv
export TMPDIR=/overlay/cache/tmp
export HOME=/overlay/cache/home
mkdir -p \"\$XDG_CACHE_HOME\" \"\$PIP_CACHE_DIR\" \"\$UV_CACHE_DIR\" \"\$TMPDIR\" \"\$HOME\"
CONSTRAINTS_FILE=\"\${TMPDIR%/}/dfm-evals-overlay-constraints.txt\"
python - <<'PY' > \"\$CONSTRAINTS_FILE\"
import importlib.metadata as md

for dist_name in (\"transformers\", \"huggingface-hub\", \"tokenizers\"):
    try:
        print(f\"{dist_name}=={md.version(dist_name)}\")
    except md.PackageNotFoundError:
        pass
PY
cd ${REPO_ROOT_Q}
"

if [[ "$NO_DEPS_SET" == "1" ]]; then
  INSTALL_CMD+="python -m pip install --no-user -U -c \"\$CONSTRAINTS_FILE\" -e ${INSTALL_TARGET_Q} --no-deps"
else
  INSTALL_CMD+="python -m pip install --no-user -U -c \"\$CONSTRAINTS_FILE\" -r ${REQ_EXPORT_FILE_Q}
python -m pip install --no-user -U -c \"\$CONSTRAINTS_FILE\" -e ${INSTALL_TARGET_Q} --no-deps"
fi

echo "+ SIF: $SIF"
echo "+ Overlay: $OVERLAY_DIR"
echo "+ Singularity GPU mode: $SINGULARITY_GPU_MODE"
echo "+ Repo: $REPO_ROOT"
echo "+ Install target: $INSTALL_TARGET"
if [[ "$NO_DEPS_SET" == "1" ]]; then
  echo "+ Dependency mode: no-deps"
else
  echo "+ Dependency mode: locked deps via uv export"
  echo "+ Exported requirements: $REQ_EXPORT_FILE_HOST"
fi

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

singularity exec "${SINGULARITY_GPU_ARGS[@]}" \
  "${SING_BIND_ARGS[@]}" \
  "$SIF" bash -lc "$INSTALL_CMD"

echo "+ Overlay install complete."
