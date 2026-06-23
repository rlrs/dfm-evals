#!/bin/bash
# Rescore existing WMT24++ EEE JSONL exports with COMET. This does not rerun vLLM.

#SBATCH --job-name=dfm-wmt24-comet
#SBATCH --account=project_465002183
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=dfm-wmt24-comet-%j.out
#SBATCH --error=dfm-wmt24-comet-%j.err

set -euo pipefail

REPO_ROOT="${DFM_EVALS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_ROOT/lumi"
source "$SCRIPT_DIR/artifact_root.sh"
POST_ARTIFACT_ROOT="$(resolve_post_artifact_root "$REPO_ROOT")"
export POST_ARTIFACT_ROOT

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
SIF=${SIF:-$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif}
OVERLAY_DIR=${OVERLAY_DIR:-$REPO_ROOT/overlay_vllm_minimal}
if [[ ! -d "$OVERLAY_DIR" && -d "$REPO_ROOT/../overlay_vllm_minimal" ]]; then
  OVERLAY_DIR="$REPO_ROOT/../overlay_vllm_minimal"
fi

INPUT_ROOT=${INPUT_ROOT:-$POST_ARTIFACT_ROOT/evals/eee/data/WMT24___en-da}
OUTPUT_DIR=${OUTPUT_DIR:-$POST_ARTIFACT_ROOT/evals/maintenance/wmt24-comet-rescore-${SLURM_JOB_ID:-manual}}
COMET_MODEL=${COMET_MODEL:-Unbabel/wmt22-comet-da}
COMET_DEVICE=${COMET_DEVICE:-auto}
BATCH_SIZE=${BATCH_SIZE:-16}
LIMIT_FILES=${LIMIT_FILES:-}
LIMIT_SAMPLES=${LIMIT_SAMPLES:-}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}

HF_HOME_HOST=${HF_HOME:-/scratch/project_465002183/.cache/huggingface}
HF_HOME_HOST=${HF_HOME_HOST%/}
HF_HOME_CONTAINER=/hf-home
mkdir -p "$HF_HOME_HOST" "$OUTPUT_DIR"
HF_HOME_HOST="$(readlink -f "$HF_HOME_HOST" 2>/dev/null || printf '%s' "$HF_HOME_HOST")"

[[ -f "$SIF" ]] || { echo "FATAL: SIF not found: $SIF" >&2; exit 1; }
[[ -d "$OVERLAY_DIR/venv/vllm-min" ]] || { echo "FATAL: overlay venv missing: $OVERLAY_DIR/venv/vllm-min" >&2; exit 1; }
[[ -d "$INPUT_ROOT" ]] || { echo "FATAL: input root not found: $INPUT_ROOT" >&2; exit 1; }

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

printf -v REPO_ROOT_Q '%q' "$REPO_ROOT"
printf -v INPUT_ROOT_Q '%q' "$INPUT_ROOT"
printf -v OUTPUT_DIR_Q '%q' "$OUTPUT_DIR"
printf -v COMET_MODEL_Q '%q' "$COMET_MODEL"
printf -v COMET_DEVICE_Q '%q' "$COMET_DEVICE"
printf -v BATCH_SIZE_Q '%q' "$BATCH_SIZE"
printf -v HF_HOME_CONTAINER_Q '%q' "$HF_HOME_CONTAINER"

EXTRA_ARGS=""
if [[ -n "$LIMIT_FILES" ]]; then
  printf -v LIMIT_FILES_Q '%q' "$LIMIT_FILES"
  EXTRA_ARGS+=" --limit-files ${LIMIT_FILES_Q}"
fi
if [[ -n "$LIMIT_SAMPLES" ]]; then
  printf -v LIMIT_SAMPLES_Q '%q' "$LIMIT_SAMPLES"
  EXTRA_ARGS+=" --limit-samples ${LIMIT_SAMPLES_Q}"
fi
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  EXTRA_ARGS+=" --local-files-only"
fi

JOB_ID_SAFE=${SLURM_JOB_ID:-manual}
RUN_CMD="
set -euo pipefail
source /overlay/venv/vllm-min/bin/activate
export HF_HOME=${HF_HOME_CONTAINER_Q}
export XDG_CACHE_HOME=/overlay/cache/dfm-wmt24-comet-${JOB_ID_SAFE}/xdg-cache
export TMPDIR=/overlay/cache/dfm-wmt24-comet-${JOB_ID_SAFE}/tmp
mkdir -p \"\$XDG_CACHE_HOME\" \"\$TMPDIR\" ${OUTPUT_DIR_Q}
export PYTHONPATH=${REPO_ROOT_Q}:/overlay/src/transformers/src:/overlay/src/vllm:/overlay/venv/vllm-min/lib/python3.12/site-packages:\${PYTHONPATH:-}
python -u ${REPO_ROOT_Q}/lumi/rescore_wmt24_comet.py \
  --input-root ${INPUT_ROOT_Q} \
  --output-dir ${OUTPUT_DIR_Q} \
  --model ${COMET_MODEL_Q} \
  --device ${COMET_DEVICE_Q} \
  --batch-size ${BATCH_SIZE_Q}${EXTRA_ARGS}
"

echo "Rescoring WMT24++ EEE exports with COMET"
echo "Input root: $INPUT_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "COMET model: $COMET_MODEL"
echo "COMET device: $COMET_DEVICE"
echo "HF_HOME host: $HF_HOME_HOST"
echo "HF_HOME container: $HF_HOME_CONTAINER"

singularity exec --rocm "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$RUN_CMD"
