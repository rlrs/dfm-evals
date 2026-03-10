# EuroEval Concurrency Control Design

## 1. Problem Statement
EuroEval's LiteLLM backend currently starts with a fixed concurrency of 20 requests, with only one dynamic behavior: it halves on `429` rate-limit errors. This is too rigid for local high-throughput endpoints (for example vLLM on LUMI), and too naive for remote providers where timeout behavior does not reliably indicate overload.

On generic OpenAI-compatible endpoints, queue/scheduler state is usually not observable. On vLLM, it is observable via `/metrics` (for example `vllm:num_requests_waiting`, `vllm:num_requests_running`, `vllm:request_queue_time_seconds`).

We want a design that:
1. Keeps backward-compatible fixed behavior.
2. Allows explicit user control.
3. Adds a safe adaptive mode when the backend exposes useful metrics.

## 2. Goals
1. Make concurrency configurable from CLI and environment.
2. Preserve stable default behavior for existing users.
3. Add an optional adaptive controller that uses vLLM metrics when available.
4. Avoid using request timeout as a primary overload signal.
5. Make all decisions observable in logs and benchmark metadata.

## 3. Non-Goals
1. No cross-process global controller across multiple EuroEval runs.
2. No provider-specific heuristics beyond simple endpoint classes (generic API vs vLLM metrics).
3. No dependency on private vLLM internals; use public HTTP metrics only.

## 4. Proposed User-Facing API

### 4.1 New CLI options
1. `--max-concurrent-calls <int>`
2. `--min-concurrent-calls <int>`
3. `--concurrency-mode <fixed|adaptive|auto>`
4. `--adaptive-increase-step <int>` (default: 1)
5. `--adaptive-decrease-factor <float>` (default: 0.5)
6. `--adaptive-cooldown-seconds <int>` (default: 20)
7. `--metrics-url <url>` (optional override; default inferred from `api_base`)
8. `--metrics-poll-seconds <int>` (default: 5)
9. `--adaptive-max-queue-waiting <int>` (default: 0)
10. `--adaptive-max-queue-p95-seconds <float>` (default: 1.5)

### 4.2 Environment variable equivalents
1. `EUROEVAL_MAX_CONCURRENT_CALLS`
2. `EUROEVAL_MIN_CONCURRENT_CALLS`
3. `EUROEVAL_CONCURRENCY_MODE`
4. `EUROEVAL_ADAPTIVE_INCREASE_STEP`
5. `EUROEVAL_ADAPTIVE_DECREASE_FACTOR`
6. `EUROEVAL_ADAPTIVE_COOLDOWN_SECONDS`
7. `EUROEVAL_METRICS_URL`
8. `EUROEVAL_METRICS_POLL_SECONDS`
9. `EUROEVAL_ADAPTIVE_MAX_QUEUE_WAITING`
10. `EUROEVAL_ADAPTIVE_MAX_QUEUE_P95_SECONDS`

### 4.3 Defaults
1. Default mode: `fixed` (backward compatible).
2. Default fixed concurrency: 20.
3. `auto` mode behavior:
- If vLLM metrics endpoint is reachable and contains expected keys, use `adaptive`.
- Else use `fixed`.

## 5. Controller Architecture

### 5.1 Interface
Create a small controller abstraction used by LiteLLM generation loop.

`ConcurrencyController` methods:
1. `current_limit() -> int`
2. `on_batch_success(stats)`
3. `on_batch_partial_failure(errors, stats)`
4. `on_hard_failure(error, stats)`
5. `snapshot() -> dict` (for metadata/logging)

### 5.2 Implementations
1. `FixedConcurrencyController`
- Always returns configured limit.
- Optional legacy behavior: halve on `429` if enabled.

2. `AdaptiveConcurrencyController`
- Starts from `initial=max_concurrent_calls`.
- Clamped between `[min_concurrent_calls, max_concurrent_calls]`.
- Uses both request outcomes and optional metrics probe.

3. `MetricsProbe`
- Polls `/metrics` periodically.
- Extracts latest:
- `vllm:num_requests_waiting`
- `vllm:num_requests_running`
- `vllm:request_queue_time_seconds` histogram (derive p95 estimate)
- Returns `None` if unavailable/parse fails.

## 6. Adaptive Algorithm

Use conservative AIMD-like control, but driven mainly by queue pressure and explicit overload errors.

### 6.1 Increase rule
Increase by `adaptive_increase_step` only if all are true:
1. No severe errors in last window (`429`, connection reset, service unavailable).
2. `num_requests_waiting <= adaptive_max_queue_waiting`.
3. `queue_p95 <= adaptive_max_queue_p95_seconds`.
4. Not in cooldown.

### 6.2 Decrease rule
Apply multiplicative decrease immediately when either occurs:
1. Severe errors (`429`, provider overload signals, repeated transport failures).
2. Queue pressure exceeds threshold for `N` consecutive polls.

Update:
`new_limit = max(min_limit, floor(current_limit * adaptive_decrease_factor))`
Then enter cooldown (`adaptive_cooldown_seconds`).

### 6.3 Timeout handling
Do **not** treat single request timeout as an automatic overload signal.
Timeout may reflect long generations. Require either:
1. repeated timeout ratio in a window, or
2. timeout + queue-pressure evidence.

### 6.4 Safeguards
1. Hysteresis: require repeated pressure samples before decreasing on metrics-only signal.
2. Cooldown after decrease.
3. Hard floor and cap.
4. If metrics disappear, degrade to fixed at current limit.

## 7. Integration Points in EuroEval
1. Replace hardcoded `self.buffer["max_concurrent_calls"] = 20` with controller setup.
2. In each generation attempt:
- ask controller for `limit`.
- run `_generate_async(..., max_concurrent_calls=limit)`.
- feed outcomes + metrics to controller.
3. Keep existing retry/error handling, but let controller consume error categories.
4. Preserve current logs; add concise concurrency decision logs.

## 8. Observability

### 8.1 Logs
Emit structured debug logs when limit changes:
1. previous limit
2. new limit
3. reason (`rate_limit`, `queue_pressure`, `stable_success`)
4. key signals (`waiting`, `running`, `queue_p95`)

### 8.2 Metadata
Include in benchmark metadata:
1. mode
2. initial/min/max limits
3. final limit
4. number of increases/decreases
5. metrics availability ratio

## 9. Testing Plan
1. Unit tests for controller transitions:
- stable ramp-up
- 429 decrease
- cooldown behavior
- clamp min/max
2. Unit tests for metrics parser from sample Prometheus text.
3. Integration test with mocked endpoint and synthetic metrics.
4. Backward-compatibility test: default config behaves as current fixed mode.

## 10. Rollout Plan
1. Phase 1: ship configurability (`fixed`) + logging.
2. Phase 2: ship `adaptive` behind explicit flag.
3. Phase 3: enable `auto` mode (metrics-aware) once validated on vLLM.

## 11. Recommended Initial Values (LUMI vLLM)
1. `concurrency_mode=adaptive`
2. `max_concurrent_calls=64`
3. `min_concurrent_calls=8`
4. `adaptive_increase_step=1`
5. `adaptive_decrease_factor=0.5`
6. `adaptive_cooldown_seconds=20`
7. `adaptive_max_queue_waiting=0`
8. `adaptive_max_queue_p95_seconds=1.5`

These should be tuned empirically per model size and prompt/output budget.

---

## Appendix A: Current Local EuroEval Dirty Changes (Overlay)

This appendix is a snapshot of the local, non-upstream EuroEval modifications
currently present in our LUMI overlay environment. It is intentionally separate
from the concurrency-control design above.

### A.1 Scope
1. Location:
`<OVERLAY_DIR>/venv/vllm-min/lib/python3.12/site-packages/euroeval`
2. Installed package baseline:
`euroeval==16.15.0` wheel
3. Verification method:
compare installed files against `euroeval-16.15.0.dist-info/RECORD` hashes.

### A.2 Dirty files and purpose
1. `euroeval/__init__.py`
- Local patch allows `flash_attn` when `EUROEVAL_ALLOW_FLASH_ATTN=1` (default
in our local patch path), instead of hard-failing import.
- Purpose: keep EuroEval runnable in the same runtime where `flash_attn` is
present for vLLM compatibility.

2. `euroeval/benchmark_modules/litellm.py`
- Local patch uses cleaned `model_id` consistently in LiteLLM router calls
(`model_name=model_id`, `model=model_id`).
- Purpose: avoid model-ID mismatch behavior on custom OpenAI-compatible API
base usage.

### A.3 Re-check command
Use this to list modified EuroEval files from RECORD hashes:

```bash
python - <<'PY'
import base64
import csv
import hashlib
import os
from pathlib import Path

overlay_dir = Path(os.environ["OVERLAY_DIR"])
sp = overlay_dir / "venv/vllm-min/lib/python3.12/site-packages"
record = sp / "euroeval-16.15.0.dist-info/RECORD"

for path, hashfield, _ in csv.reader(record.read_text().splitlines()):
    if not hashfield:
        continue
    file_path = sp / path
    if not file_path.exists():
        continue
    algo, expected = hashfield.split("=", 1)
    if algo != "sha256":
        continue
    got = base64.urlsafe_b64encode(
        hashlib.sha256(file_path.read_bytes()).digest()
    ).rstrip(b"=").decode()
    if got != expected:
        print(path)
PY
```

### A.4 Tracking note
The site-packages EuroEval installation is not a git checkout. For long-term
maintenance and upstream PR work, these patches should be replicated in a
proper EuroEval source fork/branch and kept as regular commits.
