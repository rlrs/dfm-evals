# LUMI Integration Plan for `dfm-evals`

## Objective

Move the LUMI-specific operational tooling from the wrapper repo layout into this repo under `lumi/`, so `dfm-evals` becomes the primary entrypoint and the dependency direction is inverted.

## Status Snapshot (February 25, 2026)

Completed:
- Created branch `lumi/integration`.
- Added `lumi/` folder and migrated operational scripts:
  - `lumi/submit.sh`
  - `lumi/run_suite.sbatch`
  - `lumi/view.sh`
  - `lumi/build_overlay_minimal.sh`
  - `lumi/euroeval_submit.sh`
  - `lumi/run_euroeval.sbatch`
- Added `lumi/README.md` and top-level README pointer.
- Normalized script entrypoints and references to `./lumi/...`.
- Enforced strict external-model config checks in runtime scripts:
  - no fallback behavior for `OPENAI_BASE_URL`
  - fail fast when required OpenAI env is missing

In progress:
- Track and apply inspect local-server logging patch from repo (currently runtime patch exists, but patch asset flow is not yet in `lumi/`).
- End-to-end validation matrix and documented smoke outputs from the new `lumi/` scripts.

Not started:
- Add tracked patch/apply/check helper scripts under `lumi/patches` and `lumi/scripts`.
- Final rollout/deprecation steps for the old wrapper-layer scripts.

## Scope

In scope:
- Add a first-class `lumi/` folder in this repository.
- Migrate LUMI submit/run/view/build scripts and docs.
- Preserve current behavior for fundamentals suite execution.
- Preserve run/log naming and host-visible log locations.
- Preserve and document inspect view workflow.
- Preserve full inspect-spawned vLLM server logs on disk.
- Enforce strict OpenAI config for `openai/*` models (no fallback behavior).

Out of scope:
- Reworking eval task definitions unrelated to LUMI orchestration.
- Changing scheduler target (non-LUMI clusters).
- Replacing inspect/vLLM provider internals.

## Desired End State

- `dfm-evals` contains a `lumi/` toolkit that is self-documented.
- Typical usage is from this repo directly:
  - build overlay
  - submit eval
  - view logs
- All relevant logs (suite logs + raw vLLM process logs) are persisted under overlay paths.
- OpenAI-compatible external judge usage is explicit and correctly configured.

## Proposed Repository Layout

```text
lumi/
  README.md
  submit.sh
  run_suite.sbatch
  euroeval_submit.sh
  run_euroeval.sbatch
  view.sh
  build_overlay_minimal.sh
  patches/
    inspect_local_server_logging.patch
  scripts/
    apply_inspect_patch.sh
```

Notes:
- Keep filenames short and role-focused.
- `patches/` contains tracked diffs for third-party runtime tweaks.
- `scripts/` contains idempotent helpers (patch apply/check, sanity checks).

## Migration Phases and Status

### Phase 1: Scaffold and Copy (Completed)

Deliverables:
- Create `lumi/` tree and copy current operational scripts into it.
- Keep behavior equivalent first; rename only where beneficial.
- Add header comments to clarify purpose and expected execution context.

Acceptance:
- Scripts run from `dfm-evals/lumi` without relying on parent wrapper layout.

Current status:
- Completed on branch `lumi/integration`.

### Phase 2: Path and Environment Normalization (Partially Completed)

Deliverables:
- Convert scripts to resolve paths relative to `dfm-evals` root.
- Centralize defaults:
  - SIF path
  - overlay dir
  - cache dirs
  - suite defaults
- Ensure `.env` loading is explicit in submit path.
- Ensure strict config checks:
  - `OPENAI_API_KEY` required for `openai/*`.
  - `OPENAI_BASE_URL` required for `openai/*`.

Acceptance:
- Dry-run shows complete evaluated config and required env.
- Missing required config fails fast with clear error.

Current status:
- Path normalization for script location and command references is in place.
- Strict config checks are in place.
- Remaining: finalize environment defaults for both current wrapper workspace and future in-repo overlay location, and verify with smoke jobs using only `lumi/` entrypoints.

### Phase 3: Logging and Observability Hardening (In Progress)

Deliverables:
- Ensure inspect-spawned vLLM subprocess logs are persisted per process:
  - `<run_log_dir>/_vllm_server/vllm-server-*.log`
- Store exact launch command in each server log file.
- Surface server log path in submit output and run output.

Implementation approach:
- Track a patch for `inspect_ai/_util/local_server.py`.
- Add patch-apply helper for reproducibility in overlay lifecycle.

Acceptance:
- Failed server startup includes full log location and tail context.
- Host-visible server logs exist for each spawn.

Current status:
- Runtime behavior exists (full vLLM server logs are captured), but patch mechanism is not yet committed as reusable assets in `lumi/`.

### Phase 4: Documentation and UX (In Progress)

Deliverables:
- `lumi/README.md` with end-to-end instructions:
  - prerequisites
  - overlay build/update
  - submit examples
  - inspect view usage
  - log locations
  - common failure modes
- Document judge/target matrix examples:
  - local vLLM target + external OpenAI-compatible judge
  - local vLLM target + local vLLM judge

Acceptance:
- New user can run a smoke job without needing ad-hoc commands.

Current status:
- Initial docs created.
- Remaining: expand troubleshooting and add validated command/output examples from the new layout.

### Phase 5: Validation Matrix (Pending)

Run and verify:
- fundamentals smoke (`--limit 1`) with:
  - target `vllm/google/gemma-3-4b-it`
  - judge `openai/<model>`
- fundamentals larger sample (`--limit 100`) for runtime stability.
- inspect view start against:
  - `--latest`
  - `--job-id`
  - explicit run label
- Verify logs:
  - `.eval` files in run log dir
  - `_vllm_server/*.log` files for spawned servers

Acceptance:
- Validation commands and expected outputs documented in README.

## Script Mapping (Current -> New)

- `dfm_evals_submit.sh` -> `lumi/submit.sh`
- `dfm_evals_spawn_vllm_smoke.sbatch` -> `lumi/run_suite.sbatch`
- `dfm_evals_view.sh` -> `lumi/view.sh`
- `build_vllm_overlay_minimal.sh` -> `lumi/build_overlay_minimal.sh`

## Key Design Decisions

1. Keep LUMI-specific logic under `lumi/` only.
2. Keep sbatch scripts scheduler-focused; no dependency install in job scripts.
3. Keep config strict and explicit; avoid hidden fallback behavior.
4. Treat third-party runtime patches as tracked assets, not manual edits.
5. Preserve current run label and log directory semantics.

## Risks and Mitigations

- Risk: Path breakage after moving scripts.
  - Mitigation: derive all paths from script location + repo root checks.
- Risk: Drift between overlay-installed code and repo code.
  - Mitigation: explicit overlay install/update step in README.
- Risk: Inspect/vLLM patch divergence across versions.
  - Mitigation: pin patch target file and add patch verification helper.
- Risk: Hidden env dependencies.
  - Mitigation: startup config print + explicit required var checks.

## Rollout Plan

1. Land `lumi/` scripts + docs in feature branch.
2. Smoke-test on LUMI with `--limit 1`.
3. Compare behavior with current wrapper workflow.
4. Switch team usage to `dfm-evals/lumi/*`.
5. Decommission wrapper-layer scripts after confirmation.

## Immediate Next Tasks (Updated)

1. Add `lumi/patches/inspect_local_server_logging.patch` from the current runtime modifications.
2. Add `lumi/scripts/apply_inspect_patch.sh` (and optional `verify_inspect_patch.sh`) and document when to run it.
3. Run smoke validation from `dfm-evals` root using only:
   - `./lumi/submit.sh`
   - `./lumi/view.sh`
4. Confirm `_vllm_server/*.log` artifacts and include concrete paths/examples in `lumi/README.md`.
5. Decide and document final overlay default strategy:
   - in-repo `dfm-evals/overlay_vllm_minimal`
   - or explicit `OVERLAY_DIR` required for existing external overlay paths.
