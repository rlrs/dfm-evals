#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs

: "${SIF:=$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif}"
: "${OVERLAY_DIR:=$REPO_ROOT/overlay_vllm_minimal}"
if [[ ! -d "$OVERLAY_DIR" && -d "$REPO_ROOT/../overlay_vllm_minimal" ]]; then
  OVERLAY_DIR="$REPO_ROOT/../overlay_vllm_minimal"
fi

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
      * no extras: `pip install -e . --no-deps`
      * with extras: `pip install -e ".[extras]"`
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

INSTALL_TARGET="."
if [[ -n "$EXTRAS" ]]; then
  INSTALL_TARGET=".[${EXTRAS}]"
fi

NO_DEPS_FLAG=""
if [[ "$NO_DEPS_SET" == "1" || -z "$EXTRAS" ]]; then
  NO_DEPS_FLAG="--no-deps"
fi

printf -v INSTALL_TARGET_Q '%q' "$INSTALL_TARGET"
printf -v REPO_ROOT_Q '%q' "$REPO_ROOT"

INSTALL_CMD="set -euo pipefail
source /overlay/venv/vllm-min/bin/activate
cd ${REPO_ROOT_Q}
python -m pip install --no-user -U -e ${INSTALL_TARGET_Q}"

if [[ -n "$NO_DEPS_FLAG" ]]; then
  INSTALL_CMD+=" ${NO_DEPS_FLAG}"
fi

echo "+ SIF: $SIF"
echo "+ Overlay: $OVERLAY_DIR"
echo "+ Repo: $REPO_ROOT"
echo "+ Install target: $INSTALL_TARGET"
if [[ -n "$NO_DEPS_FLAG" ]]; then
  echo "+ Dependency mode: no-deps"
else
  echo "+ Dependency mode: resolve deps"
fi

singularity exec --rocm \
  -B "$BASE_DIR:$LAIFS_APPL_DIR" \
  -B "$OVERLAY_DIR:/overlay" \
  -B "$REPO_ROOT:$REPO_ROOT" \
  "$SIF" bash -lc "$INSTALL_CMD"

echo "+ Overlay install complete."
