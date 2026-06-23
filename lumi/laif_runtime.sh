#!/usr/bin/env bash

dfm_lumi_default_sif() {
  local laif70_sif
  for laif70_sif in \
    "$BASE_DIR/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif" \
    "$BASE_DIR/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif"
  do
    if [[ -f "$laif70_sif" ]]; then
      printf '%s\n' "$laif70_sif"
      return
    fi
  done
  printf '%s\n' "$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif"
}

dfm_lumi_default_overlay_dir() {
  local candidate
  for candidate in \
    "$REPO_ROOT/../../overlay_lumi_laif70" \
    "$REPO_ROOT/overlay_lumi_laif70" \
    "$REPO_ROOT/../overlay_lumi_laif70" \
    "$REPO_ROOT/overlay_vllm_minimal" \
    "$REPO_ROOT/../overlay_vllm_minimal"
  do
    if [[ -d "$candidate/venv/vllm-min" ]]; then
      (cd "$candidate" && pwd)
      return
    fi
  done
  printf '%s\n' "$REPO_ROOT/../../overlay_lumi_laif70"
}

dfm_lumi_default_gpu_mode() {
  local sif="${1:-}"
  case "$sif" in
    *u24r70f21m50t210-20260513_121430*|\
    *u24r70f21m50t210-20260415_130625*)
      printf '%s\n' none
      ;;
    *)
      printf '%s\n' rocm
      ;;
  esac
}

dfm_lumi_init_gpu_args() {
  SINGULARITY_GPU_ARGS=()
  case "$SINGULARITY_GPU_MODE" in
    rocm)
      SINGULARITY_GPU_ARGS=(--rocm)
      ;;
    none|"")
      ;;
    *)
      die "unsupported SINGULARITY_GPU_MODE=$SINGULARITY_GPU_MODE (expected rocm or none)"
      ;;
  esac
}
