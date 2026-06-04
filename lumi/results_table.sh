#!/bin/bash
# Print aggregate task/scorer/metric results from Every Eval Ever JSON artifacts.
# Supports model-comparison pivot table across multiple exports.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/artifact_root.sh"
POST_ARTIFACT_ROOT="$(resolve_post_artifact_root "$REPO_ROOT")"
export POST_ARTIFACT_ROOT

EEE_DATA_ROOT_HOST=${EEE_DATA_ROOT_HOST:-$POST_ARTIFACT_ROOT/evals/eee/data}
LATEST_WINDOW_SECONDS=${LATEST_WINDOW_SECONDS:-120}
RESULTS_TABLE_WIDTH=${RESULTS_TABLE_WIDTH:-168}
RESULTS_TABLE_HTML_DIR=${RESULTS_TABLE_HTML_DIR:-$POST_ARTIFACT_ROOT/evals/results_table}
RESULTS_TABLE_AUTO_OPEN=${RESULTS_TABLE_AUTO_OPEN:-1}
RESULTS_TABLE_SERVE=${RESULTS_TABLE_SERVE:-1}
RESULTS_TABLE_KEEP_OPEN=${RESULTS_TABLE_KEEP_OPEN:-1}
RESULTS_TABLE_HOST=${RESULTS_TABLE_HOST:-127.0.0.1}
RESULTS_TABLE_BIND_HOST=${RESULTS_TABLE_BIND_HOST:-127.0.0.1}
RESULTS_TABLE_PORT=${RESULTS_TABLE_PORT:-7586}

SELECTOR="latest"
SELECTOR_SET=0
RUN_LABEL=""
LOG_DIR=""

COMPARE_MODELS=0
PRIMARY_ONLY=1
FORMAT=${FORMAT:-html}
COMPARE_ORIENTATION=${COMPARE_ORIENTATION:-model-rows}

usage() {
  cat <<'USAGE'
Usage:
  ./lumi/results_table.sh [selector options] [view options]

Selector options:
  --latest             Use newest EEE export batch under data root (default)
  --all-runs           Use all EEE records under data root
  --run-label <label>  Use legacy evals/runs/<label>/every_eval_ever if present,
                       otherwise EEE subdir <data_root>/<label> if present
  --log-dir <path>     Use explicit EEE directory path

View options:
  --compare-models     Pivot table by model (columns are models, rows are tasks)
  --primary-only       For compare mode: use one primary metric per task (default)
  --all-metrics        For compare mode: include every scorer+metric row per task
  --model-rows         For compare mode: rows=models, columns=tasks (default)
  --task-rows          For compare mode: rows=tasks, columns=models
  --format <fmt>       Output format: html|table|csv|json (default: html)
  --help               Show help

Environment overrides:
  EEE_DATA_ROOT_HOST     Host EEE data root (default: $POST_ARTIFACT_ROOT/evals/eee/data)
  LATEST_WINDOW_SECONDS  Window for --latest selection by file mtime (default: 120)
  RESULTS_TABLE_WIDTH    Preferred rich table width for --format table (default: 168)
  RESULTS_TABLE_HTML_DIR Output dir for generated HTML reports
  RESULTS_TABLE_SERVE    Serve HTML reports on a local HTTP port (default: 1)
  RESULTS_TABLE_KEEP_OPEN Keep the script running and stop the HTML server on Ctrl+C (default: 1)
  RESULTS_TABLE_HOST     Hostname to use in printed HTML report URLs (default: 127.0.0.1)
  RESULTS_TABLE_BIND_HOST Bind host for the local HTML server (default: 127.0.0.1)
  RESULTS_TABLE_PORT     Preferred local HTML server port (default: 7586)
  RESULTS_TABLE_AUTO_OPEN Attempt to open generated HTML reports when a GUI session is available (default: 1)

Examples:
  ./lumi/results_table.sh --latest
  ./lumi/results_table.sh --all-runs
  ./lumi/results_table.sh --compare-models --all-runs
  ./lumi/results_table.sh --compare-models --all-runs --format table
  ./lumi/results_table.sh --compare-models --all-runs --all-metrics --format csv
USAGE
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

resolve_run_label_dir() {
  local label="$1"
  local legacy="$POST_ARTIFACT_ROOT/evals/runs/$label/every_eval_ever"
  local rooted="$EEE_DATA_ROOT_HOST/$label"

  if [[ -d "$legacy" ]]; then
    printf '%s' "$legacy"
    return 0
  fi
  if [[ -d "$rooted" ]]; then
    printf '%s' "$rooted"
    return 0
  fi

  die "could not resolve run-label '$label' in legacy or EEE data root paths"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latest)
      SELECTOR="latest"
      SELECTOR_SET=1
      RUN_LABEL=""
      LOG_DIR=""
      shift
      ;;
    --all-runs)
      SELECTOR="all-runs"
      SELECTOR_SET=1
      RUN_LABEL=""
      LOG_DIR=""
      shift
      ;;
    --run-label)
      need_value "$1" "$#"
      SELECTOR="run-label"
      SELECTOR_SET=1
      RUN_LABEL="$2"
      LOG_DIR=""
      shift 2
      ;;
    --log-dir)
      need_value "$1" "$#"
      SELECTOR="log-dir"
      SELECTOR_SET=1
      LOG_DIR="$2"
      RUN_LABEL=""
      shift 2
      ;;
    --compare-models)
      COMPARE_MODELS=1
      shift
      ;;
    --primary-only)
      PRIMARY_ONLY=1
      shift
      ;;
    --all-metrics)
      PRIMARY_ONLY=0
      shift
      ;;
    --model-rows)
      COMPARE_ORIENTATION="model-rows"
      shift
      ;;
    --task-rows)
      COMPARE_ORIENTATION="task-rows"
      shift
      ;;
    --format)
      need_value "$1" "$#"
      FORMAT="$2"
      shift 2
      ;;
    --help|-h|help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

case "$FORMAT" in
  html|table|csv|json)
    ;;
  *)
    die "invalid --format: $FORMAT (expected html|table|csv|json)"
    ;;
esac

case "$COMPARE_ORIENTATION" in
  model-rows|task-rows)
    ;;
  *)
    die "invalid compare orientation: $COMPARE_ORIENTATION (expected model-rows|task-rows)"
    ;;
esac

if [[ "$COMPARE_MODELS" == "1" && "$SELECTOR_SET" == "0" ]]; then
  SELECTOR="all-runs"
fi

SOURCE_DIRS=()
case "$SELECTOR" in
  latest|all-runs)
    [[ -d "$EEE_DATA_ROOT_HOST" ]] || die "EEE data root not found: $EEE_DATA_ROOT_HOST"
    SOURCE_DIRS+=("$EEE_DATA_ROOT_HOST")
    ;;
  run-label)
    [[ -n "$RUN_LABEL" ]] || die "--run-label requires a value"
    SOURCE_DIRS+=("$(resolve_run_label_dir "$RUN_LABEL")")
    ;;
  log-dir)
    [[ -n "$LOG_DIR" ]] || die "--log-dir requires a value"
    if [[ "$LOG_DIR" == /* ]]; then
      SOURCE_DIRS+=("$LOG_DIR")
    elif [[ -d "$LOG_DIR" ]]; then
      SOURCE_DIRS+=("$LOG_DIR")
    else
      SOURCE_DIRS+=("$EEE_DATA_ROOT_HOST/$LOG_DIR")
    fi
    ;;
  *)
    die "unknown selector: $SELECTOR"
    ;;
esac

[[ "${#SOURCE_DIRS[@]}" -gt 0 ]] || die "no source dirs resolved"
for d in "${SOURCE_DIRS[@]}"; do
  [[ -d "$d" ]] || die "source dir not found: $d"
done

{
  echo "Selector: $SELECTOR"
  echo "Compare models: $COMPARE_MODELS"
  if [[ "$COMPARE_MODELS" == "1" ]]; then
    echo "Primary only: $PRIMARY_ONLY"
    echo "Orientation: $COMPARE_ORIENTATION"
  fi
  echo "Format: $FORMAT"
  echo "EEE data root: $EEE_DATA_ROOT_HOST"
  echo "Latest window seconds: $LATEST_WINDOW_SECONDS"
  echo "Results table width: $RESULTS_TABLE_WIDTH"
  echo "Results table HTML dir: $RESULTS_TABLE_HTML_DIR"
  echo "Results table serve: $RESULTS_TABLE_SERVE"
  echo "Results table keep open: $RESULTS_TABLE_KEEP_OPEN"
  echo "Results table host: $RESULTS_TABLE_HOST"
  echo "Results table bind host: $RESULTS_TABLE_BIND_HOST"
  echo "Results table port: $RESULTS_TABLE_PORT"
  echo "Results table auto open: $RESULTS_TABLE_AUTO_OPEN"
  echo "Source dirs:"
  for d in "${SOURCE_DIRS[@]}"; do
    echo "  - $d"
  done
} >&2

SOURCE_DIRS_NL="$(printf '%s\n' "${SOURCE_DIRS[@]}")"

SOURCE_DIRS_NL="$SOURCE_DIRS_NL" \
FORMAT="$FORMAT" \
COMPARE_MODELS="$COMPARE_MODELS" \
PRIMARY_ONLY="$PRIMARY_ONLY" \
COMPARE_ORIENTATION="$COMPARE_ORIENTATION" \
SELECTOR="$SELECTOR" \
LATEST_WINDOW_SECONDS="$LATEST_WINDOW_SECONDS" \
RESULTS_TABLE_WIDTH="$RESULTS_TABLE_WIDTH" \
RESULTS_TABLE_HTML_DIR="$RESULTS_TABLE_HTML_DIR" \
RESULTS_TABLE_SERVE="$RESULTS_TABLE_SERVE" \
RESULTS_TABLE_KEEP_OPEN="$RESULTS_TABLE_KEEP_OPEN" \
RESULTS_TABLE_HOST="$RESULTS_TABLE_HOST" \
RESULTS_TABLE_BIND_HOST="$RESULTS_TABLE_BIND_HOST" \
RESULTS_TABLE_PORT="$RESULTS_TABLE_PORT" \
RESULTS_TABLE_AUTO_OPEN="$RESULTS_TABLE_AUTO_OPEN" \
python3 - <<'PY'
import csv
import glob
import html
import json
import os
import re
import shlex
import socket
import statistics
import subprocess
import signal
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    HAVE_RICH = True
    try:
        rich_width = int(os.environ.get("RESULTS_TABLE_WIDTH", "140"))
    except Exception:
        rich_width = 140
    if rich_width < 80:
        rich_width = 80
    RICH_CONSOLE = Console(highlight=False, width=rich_width)
except Exception:
    HAVE_RICH = False
    RICH_CONSOLE = None

source_dirs = [d for d in os.environ["SOURCE_DIRS_NL"].splitlines() if d]
fmt = os.environ["FORMAT"]
compare_models = os.environ["COMPARE_MODELS"] == "1"
primary_only = os.environ["PRIMARY_ONLY"] == "1"
orientation = os.environ["COMPARE_ORIENTATION"]
selector = os.environ["SELECTOR"]
latest_window_seconds = float(os.environ["LATEST_WINDOW_SECONDS"])
html_output_dir = Path(os.environ["RESULTS_TABLE_HTML_DIR"])
serve_html = os.environ.get("RESULTS_TABLE_SERVE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
keep_open_html = os.environ.get("RESULTS_TABLE_KEEP_OPEN", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
html_url_host = os.environ.get("RESULTS_TABLE_HOST", "127.0.0.1").strip() or "127.0.0.1"
html_bind_host = os.environ.get("RESULTS_TABLE_BIND_HOST", "127.0.0.1").strip() or "127.0.0.1"
try:
    html_preferred_port = int(os.environ.get("RESULTS_TABLE_PORT", "7586"))
except Exception:
    html_preferred_port = 7586
auto_open_html = os.environ.get("RESULTS_TABLE_AUTO_OPEN", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

preferred_metrics = [
    "accuracy",
    "final_acc",
    "mean",
    "correct",
    "f_score",
    "prompt_strict_acc",
    "inst_strict_acc",
]
preferred_rank = {name: idx for idx, name in enumerate(preferred_metrics)}

preferred_scorers_by_task = {
    "gec_dala": ["exact"],
    "MultiWikiQA-da": ["f1"],
    "multi_wiki_qa": ["f1"],
}

required_primary_scorers_by_task = {
    "gec_dala": ["exact"],
    "MultiWikiQA-da": ["f1"],
    "multi_wiki_qa": ["f1"],
}


def emit_string_table(columns, rows, *, title=None):
    # columns: list of (key, header, cap, align)
    if HAVE_RICH:
        table = Table(
            title=title,
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=True,
            header_style="bold bright_white",
            row_styles=["none", "dim"],
            pad_edge=True,
            expand=False,
        )
        for _key, header, cap, align in columns:
            justify = "right" if align == "right" else "left"
            min_width = None
            max_width = None
            if isinstance(cap, int) and cap > 0:
                max_width = cap
            elif isinstance(cap, (tuple, list)) and len(cap) == 2:
                cap_min, cap_max = cap
                min_width = cap_min if isinstance(cap_min, int) and cap_min > 0 else None
                max_width = cap_max if isinstance(cap_max, int) and cap_max > 0 else None
            table.add_column(
                str(header),
                justify=justify,
                min_width=min_width,
                max_width=max_width,
                overflow="ellipsis",
            )

        for row in rows:
            table.add_row(*(str(row.get(key, "-")) for key, _h, _c, _a in columns))

        RICH_CONSOLE.print(table)
        return

    widths = {}
    for key, title_text, cap, _align in columns:
        max_len = len(str(title_text))
        for row in rows:
            max_len = max(max_len, len(str(row.get(key, "-"))))
        cap_min = None
        cap_max = None
        if isinstance(cap, int):
            cap_max = cap
        elif isinstance(cap, (tuple, list)) and len(cap) == 2:
            cap_min, cap_max = cap

        width = max_len
        if isinstance(cap_max, int) and cap_max > 0:
            width = min(width, cap_max)
        if isinstance(cap_min, int) and cap_min > 0:
            width = max(width, cap_min)
        widths[key] = width

    header = " ".join(trim(str(title_text), widths[key]).ljust(widths[key]) for key, title_text, _cap, _align in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        parts = []
        for key, _title, _cap, align in columns:
            cell = trim(str(row.get(key, "-")), widths[key])
            if align == "right":
                parts.append(cell.rjust(widths[key]))
            else:
                parts.append(cell.ljust(widths[key]))
        print(" ".join(parts))


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "").strip()).strip("-._")
    return slug or "results"


def results_table_output_path(*, selector: str, compare_models: bool, orientation: str, fmt: str) -> Path:
    html_output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    mode = "compare-models" if compare_models else "results"
    orientation_slug = orientation if compare_models else "flat"
    filename = f"{slugify(selector)}__{mode}__{slugify(orientation_slug)}__{stamp}.{fmt}"
    return html_output_dir / filename


def maybe_open_html(path: Path) -> None:
    if not auto_open_html:
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return
    try:
        target = path if isinstance(path, str) else path.resolve().as_uri()
        webbrowser.open_new_tab(str(target))
    except Exception:
        return


def server_state_path() -> Path:
    return html_output_dir / ".results-table-server.json"


def server_log_path() -> Path:
    return html_output_dir / ".results-table-server.log"


def read_server_state() -> Optional[dict]:
    path = server_state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_server_state(state: dict) -> None:
    server_state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def port_is_listening(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_free_port(host: str, preferred_port: int, attempts: int = 50) -> int:
    port = max(1, preferred_port)
    for _ in range(attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                port += 1
                continue
            return port
    raise RuntimeError(f"could not find a free port starting at {preferred_port}")


def ensure_html_server() -> int:
    html_output_dir.mkdir(parents=True, exist_ok=True)
    state = read_server_state()
    if state:
        pid = int(state.get("pid") or 0)
        port = int(state.get("port") or 0)
        bind_host = str(state.get("bind_host") or html_bind_host)
        directory = str(state.get("directory") or "")
        if directory == str(html_output_dir) and pid and port and is_pid_alive(pid) and port_is_listening(bind_host, port):
            return port

    port = find_free_port(html_bind_host, html_preferred_port)
    log_path = server_log_path()
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                os.sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                html_bind_host,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
            cwd=str(html_output_dir),
        )
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"results table server exited early with code {proc.returncode}")
        if port_is_listening(html_bind_host, port):
            write_server_state(
                {
                    "pid": proc.pid,
                    "port": port,
                    "bind_host": html_bind_host,
                    "url_host": html_url_host,
                    "directory": str(html_output_dir),
                    "log_path": str(log_path),
                    "started_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }
            )
            return port
        time.sleep(0.1)
    raise RuntimeError(f"results table server did not become ready on {html_bind_host}:{port}")


def start_html_server() -> Tuple[subprocess.Popen, int]:
    html_output_dir.mkdir(parents=True, exist_ok=True)
    port = find_free_port(html_bind_host, html_preferred_port)
    log_path = server_log_path()
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                os.sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                html_bind_host,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
            cwd=str(html_output_dir),
        )
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"results table server exited early with code {proc.returncode}")
        if port_is_listening(html_bind_host, port):
            return proc, port
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError(f"results table server did not become ready on {html_bind_host}:{port}")


def wait_for_server(proc: subprocess.Popen, url: str) -> None:
    print(f"Serving results table at {url} (Ctrl+C to stop)", file=os.sys.stderr)
    stopping = False

    def _stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True
        if proc.poll() is None:
            proc.terminate()

    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                if stopping and rc in (-signal.SIGTERM, 128 + signal.SIGTERM, 143):
                    return
                if rc != 0:
                    raise RuntimeError(f"results table server exited with code {rc}")
                return
            time.sleep(0.25)
    except KeyboardInterrupt:
        _stop()
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        if proc.poll() is None:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)


def html_report_target(output_path: Path) -> Tuple[str, Optional[subprocess.Popen]]:
    if not serve_html:
        return str(output_path), None
    if keep_open_html:
        proc, port = start_html_server()
        return f"http://{html_url_host}:{port}/{quote(output_path.name)}", proc
    port = ensure_html_server()
    return f"http://{html_url_host}:{port}/{quote(output_path.name)}", None


def cell_heatmap_colors(value: Optional[float], low: Optional[float], high: Optional[float]) -> Optional[Tuple[str, str]]:
    if value is None or low is None or high is None:
        return None
    if high <= low:
        ratio = 0.5
    else:
        ratio = max(0.0, min(1.0, (value - low) / (high - low)))
    hue = 8.0 + (ratio * 112.0)
    sat = 68.0 + (ratio * 14.0)
    light = 88.0 - (ratio * 10.0)
    alpha = 0.82
    return (
        f"hsl({hue:.1f} {sat:.1f}% {light:.1f}% / {alpha:.2f})",
        f"hsl({hue:.1f} 44% {max(light - 16.0, 38.0):.1f}% / 0.82)",
    )


def relative_heatmap_colors(value: Optional[float], scale: Optional[float]) -> Optional[Tuple[str, str]]:
    if value is None or scale is None or scale <= 0:
        return None
    ratio = max(-1.0, min(1.0, value / scale))
    magnitude = abs(ratio)
    if magnitude < 1e-9:
        return None
    # Keep very small absolute deltas visually neutral, then ramp contrast more
    # aggressively once the difference is meaningfully above rounding noise.
    deadband = min(0.015, max(scale * 0.25, 0.004))
    abs_value = abs(float(value))
    if abs_value <= deadband:
        return None
    effective_scale = max(scale - deadband, 1e-9)
    boosted = min(1.0, (abs_value - deadband) / effective_scale) ** 0.8
    hue = 120.0 if ratio > 0 else 8.0
    sat = 58.0 + (boosted * 30.0)
    light = 95.0 - (boosted * 24.0)
    alpha = 0.16 + (boosted * 0.56)
    return (
        f"hsl({hue:.1f} {sat:.1f}% {light:.1f}% / {alpha:.2f})",
        f"hsl({hue:.1f} {max(sat - 4.0, 48.0):.1f}% {max(light - 26.0, 28.0):.1f}% / 0.92)",
    )


def lookup_mode_field(row: dict, prefix: str, mode_id: str, key: str):
    mode_field = f"{prefix}_{mode_id}_{key}"
    if mode_field in row:
        return row.get(mode_field)
    return row.get(f"{prefix}_{key}")


def emit_html_report(
    columns,
    rows,
    *,
    numeric_keys: Optional[Set[str]] = None,
    relative_numeric_keys: Optional[Set[str]] = None,
    heatmap_mode: Optional[str] = None,
    sticky_columns: int = 1,
    column_groups: Optional[List[Tuple[str, int]]] = None,
    row_group_key: Optional[str] = None,
    row_group_modes: Optional[List[Tuple[str, str, str]]] = None,
    color_group_modes: Optional[List[str]] = None,
):
    numeric_keys = numeric_keys or set()
    relative_numeric_keys = relative_numeric_keys or set()
    output_path = results_table_output_path(
        selector=selector,
        compare_models=compare_models,
        orientation=orientation,
        fmt="html",
    )

    row_ranges: List[Dict[str, Tuple[Optional[float], Optional[float]]]] = []
    col_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    absolute_group_ranges_by_mode: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
    relative_row_scales: List[Optional[float]] = []
    relative_col_scales: Dict[str, Optional[float]] = {}
    relative_group_scales_by_mode: Dict[str, Dict[str, Optional[float]]] = {}

    for mode_id in color_group_modes or []:
        absolute_group_values: Dict[str, List[float]] = {}
        relative_group_values: Dict[str, List[float]] = {}
        for row in rows:
            for key, *_rest in columns:
                if key in numeric_keys:
                    abs_group = lookup_mode_field(row, "__absolute_group", mode_id, key)
                    abs_value = row.get(key)
                    if abs_group is not None and isinstance(abs_value, (int, float)):
                        absolute_group_values.setdefault(str(abs_group), []).append(float(abs_value))
                if key in relative_numeric_keys:
                    rel_group = lookup_mode_field(row, "__relative_group", mode_id, key)
                    rel_value = lookup_mode_field(row, "__relative", mode_id, key)
                    if rel_group is not None and isinstance(rel_value, (int, float)):
                        relative_group_values.setdefault(str(rel_group), []).append(float(rel_value))
        absolute_group_ranges_by_mode[mode_id] = {
            group: ((min(values), max(values)) if values else (None, None))
            for group, values in absolute_group_values.items()
        }
        relative_group_scales_by_mode[mode_id] = {
            group: (max(abs(min(values)), abs(max(values))) if values else None)
            for group, values in relative_group_values.items()
        }

    if heatmap_mode == "row":
        for row in rows:
            values = [
                float(row.get(key))
                for key, *_rest in columns
                if key in numeric_keys and isinstance(row.get(key), (int, float))
            ]
            if values:
                row_ranges.append({"__row__": (min(values), max(values))})
            else:
                row_ranges.append({"__row__": (None, None)})
            relative_values = [
                float(row.get(f"__relative_{key}"))
                for key, *_rest in columns
                if key in relative_numeric_keys and isinstance(row.get(f"__relative_{key}"), (int, float))
            ]
            if relative_values:
                relative_row_scales.append(max(abs(min(relative_values)), abs(max(relative_values))))
            else:
                relative_row_scales.append(None)
    elif heatmap_mode == "column":
        for key, *_rest in columns:
            if key not in numeric_keys:
                continue
            values = [float(row.get(key)) for row in rows if isinstance(row.get(key), (int, float))]
            if values:
                col_ranges[key] = (min(values), max(values))
            else:
                col_ranges[key] = (None, None)
            if key in relative_numeric_keys:
                relative_values = [
                    float(row.get(f"__relative_{key}"))
                    for row in rows
                    if isinstance(row.get(f"__relative_{key}"), (int, float))
                ]
                if relative_values:
                    relative_col_scales[key] = max(abs(min(relative_values)), abs(max(relative_values)))
                else:
                    relative_col_scales[key] = None

    if row_group_modes is None:
        row_group_modes = []
    if row_group_key and not row_group_modes:
        row_group_modes = [("default", "Group", row_group_key)]
    default_row_group_mode = row_group_modes[0][0] if row_group_modes else None
    default_row_group_field = row_group_modes[0][2] if row_group_modes else None
    if row_group_modes:
        mode_to_field = {str(mode_id): field_key for mode_id, _label, field_key in row_group_modes}
        if "sizefamily" in mode_to_field:
            default_row_group_mode = "sizefamily"
            default_row_group_field = mode_to_field["sizefamily"]
    if color_group_modes is None:
        color_group_modes = [default_row_group_mode or "default"]
    default_color_group_mode = default_row_group_mode or color_group_modes[0]
    has_relative_mode = any(
        isinstance(lookup_mode_field(row, "__relative", mode_id, key), (int, float))
        for row in rows
        for key in relative_numeric_keys
        for mode_id in color_group_modes
    )
    default_display_mode = "relative" if has_relative_mode else "absolute"

    def render_header_cells(*, sortable: bool) -> str:
        rendered = []
        for idx, column in enumerate(columns):
            key, header, cap, _align = column[:4]
            header_title = column[4] if len(column) >= 5 else header
            classes = []
            if sortable:
                classes.append("sortable")
            if idx < sticky_columns:
                classes.append(f"sticky-col-{idx}")
                classes.append("sticky-header")
            if key in numeric_keys:
                classes.append("numeric")
            cap_style = ""
            if isinstance(cap, int) and cap > 0:
                cap_style = f"max-width:{cap}ch;"
            elif isinstance(cap, (tuple, list)) and len(cap) == 2:
                cap_min, cap_max = cap
                if isinstance(cap_min, int) and cap_min > 0:
                    cap_style += f"min-width:{cap_min}ch;"
                if isinstance(cap_max, int) and cap_max > 0:
                    cap_style += f"max-width:{cap_max}ch;"
            rendered.append(
                f'<th class="{" ".join(classes)}" data-key="{html.escape(str(key))}" '
                f'data-type="{"number" if key in numeric_keys else "text"}" '
                f'title="{html.escape(str(header_title))}" style="{cap_style}">{html.escape(str(header))}</th>'
            )
        return "".join(rendered)

    header_cells = render_header_cells(sortable=True)
    repeated_header_cells = render_header_cells(sortable=False)

    def group_header_rows(group: str) -> str:
        if not default_row_group_field:
            return ""
        return (
            f'<tr class="group-row"><td colspan="{len(columns)}">{html.escape(group)}</td></tr>'
            f'<tr class="repeated-header-row">{repeated_header_cells}</tr>'
        )

    grouped_header_row = ""
    if column_groups:
        group_cells = []
        if sticky_columns > 0:
            group_cells.append(
                f'<th class="group-gap" colspan="{sticky_columns}"></th>'
            )
        for label, span in column_groups:
            if span <= 0:
                continue
            group_cells.append(
                f'<th class="group-band" colspan="{span}">{html.escape(str(label))}</th>'
            )
        grouped_header_row = "<tr>" + "".join(group_cells) + "</tr>"

    body_rows = []
    previous_group = None
    for row_index, row in enumerate(rows):
        if default_row_group_field:
            current_group = str(row.get(default_row_group_field) or "other")
            if current_group != previous_group:
                body_rows.append(group_header_rows(current_group))
                previous_group = current_group
        cells = []
        for col_index, column in enumerate(columns):
            key, _header, cap, _align = column[:4]
            raw_value = row.get(key)
            text_value = value_text(raw_value) if key in numeric_keys else str(raw_value if raw_value is not None else "-")
            title_value = str(row.get(f"__title_{key}", text_value))
            sort_value = ""
            if key in numeric_keys and isinstance(raw_value, (int, float)):
                sort_value = f"{float(raw_value):.12g}"
            else:
                sort_value = text_value.lower()

            classes = []
            if col_index < sticky_columns:
                classes.append(f"sticky-col-{col_index}")
            if key in numeric_keys:
                classes.append("numeric")
                classes.append("mode-aware")
            if key == "model":
                classes.append("model-name")
            if key == "size":
                classes.append("size-chip")
            copy_command = row.get(f"__copy_{key}")
            if copy_command:
                classes.append("copyable")

            base_style = ""
            if isinstance(cap, int) and cap > 0:
                base_style += f"max-width:{cap}ch;"
            elif isinstance(cap, (tuple, list)) and len(cap) == 2:
                cap_min, cap_max = cap
                if isinstance(cap_min, int) and cap_min > 0:
                    base_style += f"min-width:{cap_min}ch;"
                if isinstance(cap_max, int) and cap_max > 0:
                    base_style += f"max-width:{cap_max}ch;"
            abs_colors_by_mode: Dict[str, Optional[Tuple[str, str]]] = {}
            if key in numeric_keys and isinstance(raw_value, (int, float)):
                for mode_id in color_group_modes:
                    abs_group = row.get(f"__absolute_group_{mode_id}_{key}")
                    if abs_group is None:
                        abs_group = row.get(f"__absolute_group_{key}")
                    if abs_group is not None:
                        low, high = absolute_group_ranges_by_mode.get(mode_id, {}).get(str(abs_group), (None, None))
                        colors = cell_heatmap_colors(float(raw_value), low, high)
                    elif heatmap_mode == "row":
                        low, high = row_ranges[row_index]["__row__"]
                        colors = cell_heatmap_colors(float(raw_value), low, high)
                    elif heatmap_mode == "column":
                        low, high = col_ranges.get(key, (None, None))
                        colors = cell_heatmap_colors(float(raw_value), low, high)
                    else:
                        colors = None
                    abs_colors_by_mode[mode_id] = colors

            default_abs_colors = abs_colors_by_mode.get(default_color_group_mode)
            abs_bg = default_abs_colors[0] if default_abs_colors else ""
            abs_border = default_abs_colors[1] if default_abs_colors else ""

            rel_display_by_mode: Dict[str, str] = {}
            rel_title_by_mode: Dict[str, str] = {}
            rel_sort_by_mode: Dict[str, str] = {}
            rel_group_by_mode: Dict[str, str] = {}

            rel_colors_by_mode: Dict[str, Optional[Tuple[str, str]]] = {}
            if key in relative_numeric_keys:
                for mode_id in color_group_modes:
                    mode_rel_raw_value = lookup_mode_field(row, "__relative", mode_id, key)
                    mode_rel_display_value = normalize_relative_value(mode_rel_raw_value)
                    mode_rel_title_value = str(
                        lookup_mode_field(row, "__relative_title", mode_id, key) or title_value
                    )
                    mode_rel_group_value = lookup_mode_field(row, "__relative_group", mode_id, key)
                    rel_display_by_mode[mode_id] = delta_text(mode_rel_display_value)
                    rel_title_by_mode[mode_id] = mode_rel_title_value
                    rel_group_by_mode[mode_id] = str(mode_rel_group_value or "")
                    if isinstance(mode_rel_display_value, (int, float)):
                        rel_sort_by_mode[mode_id] = f"{float(mode_rel_display_value):.12g}"
                    else:
                        rel_sort_by_mode[mode_id] = ""
                    if isinstance(mode_rel_display_value, (int, float)):
                        if mode_rel_group_value is not None:
                            colors = relative_heatmap_colors(
                                float(mode_rel_display_value),
                                relative_group_scales_by_mode.get(mode_id, {}).get(str(mode_rel_group_value)),
                            )
                        elif heatmap_mode == "row":
                            colors = relative_heatmap_colors(float(mode_rel_display_value), relative_row_scales[row_index])
                        elif heatmap_mode == "column":
                            colors = relative_heatmap_colors(float(mode_rel_display_value), relative_col_scales.get(key))
                        else:
                            colors = None
                    else:
                        colors = None
                    rel_colors_by_mode[mode_id] = colors
            else:
                for mode_id in color_group_modes:
                    rel_display_by_mode[mode_id] = text_value
                    rel_title_by_mode[mode_id] = title_value
                    rel_sort_by_mode[mode_id] = sort_value
                    rel_group_by_mode[mode_id] = ""
            default_rel_colors = rel_colors_by_mode.get(default_color_group_mode)
            rel_bg = default_rel_colors[0] if default_rel_colors else ""
            rel_border = default_rel_colors[1] if default_rel_colors else ""
            rel_text_value = rel_display_by_mode.get(default_color_group_mode, text_value)
            rel_title_value = rel_title_by_mode.get(default_color_group_mode, title_value)
            rel_sort_value = rel_sort_by_mode.get(default_color_group_mode, sort_value)
            rel_group = rel_group_by_mode.get(default_color_group_mode, "")

            style_value = base_style
            if abs_bg:
                style_value += f"--abs-bg:{abs_bg};"
            if abs_border:
                style_value += f"--abs-border:{abs_border};"
            if rel_bg:
                style_value += f"--rel-bg:{rel_bg};"
            if rel_border:
                style_value += f"--rel-border:{rel_border};"
            for mode_id in color_group_modes:
                colors = abs_colors_by_mode.get(mode_id)
                if colors:
                    style_value += f"--abs-bg-{mode_id}:{colors[0]};--abs-border-{mode_id}:{colors[1]};"
                rel_colors = rel_colors_by_mode.get(mode_id)
                if rel_colors:
                    style_value += f"--rel-bg-{mode_id}:{rel_colors[0]};--rel-border-{mode_id}:{rel_colors[1]};"

            display_html = html.escape(text_value)
            if key in relative_numeric_keys:
                display_html = (
                    f'<span class="value value-abs">{html.escape(text_value)}</span>'
                    f'<span class="value value-rel">{html.escape(rel_text_value)}</span>'
                )
            if copy_command:
                title_value = f"{title_value}\nClick to copy: {copy_command}"
                rel_title_value = f"{rel_title_value}\nClick to copy: {copy_command}"
                for mode_id in list(rel_title_by_mode.keys()):
                    rel_title_by_mode[mode_id] = (
                        f"{rel_title_by_mode[mode_id]}\nClick to copy: {copy_command}"
                    )

            cells.append(
                f'<td class="{" ".join(classes)}" data-sort="{html.escape(sort_value)}" '
                f'data-copy-command="{html.escape(str(copy_command or ""))}" '
                f'data-abs-display="{html.escape(text_value)}" '
                f'data-abs-sort="{html.escape(sort_value)}" '
                f'data-abs-title="{html.escape(title_value)}" '
                f'data-abs-style="{html.escape(base_style)}" '
                f'data-rel-display="{html.escape(rel_text_value)}" '
                f'data-rel-sort="{html.escape(rel_sort_value)}" '
                f'data-rel-title="{html.escape(rel_title_value)}" '
                f'data-rel-group="{html.escape(str(rel_group or ""))}" '
                f'data-rel-style="{html.escape(base_style)}" '
                + "".join(
                    [
                        f'data-rel-display-{html.escape(str(mode_id))}="{html.escape(rel_display_by_mode.get(mode_id, rel_text_value))}" '
                        f'data-rel-sort-{html.escape(str(mode_id))}="{html.escape(rel_sort_by_mode.get(mode_id, rel_sort_value))}" '
                        f'data-rel-title-{html.escape(str(mode_id))}="{html.escape(rel_title_by_mode.get(mode_id, rel_title_value))}" '
                        f'data-rel-group-{html.escape(str(mode_id))}="{html.escape(rel_group_by_mode.get(mode_id, rel_group))}" '
                        for mode_id in color_group_modes
                    ]
                )
                + f'title="{html.escape(title_value)}" style="{style_value}">{display_html}</td>'
            )
        row_group_attr = ""
        if row_group_modes:
            attrs = []
            for mode_id, _label, field_key in row_group_modes:
                attrs.append(
                    f' data-row-group-{html.escape(str(mode_id))}="{html.escape(str(row.get(field_key) or ""))}"'
                )
            if default_row_group_field:
                attrs.append(f' data-row-group="{html.escape(str(row.get(default_row_group_field) or ""))}"')
            row_group_attr = "".join(attrs)
        body_rows.append(f"<tr{row_group_attr}>" + "".join(cells) + "</tr>")

    mode_controls_html = ""
    if has_relative_mode:
        mode_controls_html = (
            '<div class="mode-toggle">'
            f'<button type="button" class="mode-btn{" active" if default_display_mode == "absolute" else ""}" data-mode="absolute">Absolute</button>'
            f'<button type="button" class="mode-btn{" active" if default_display_mode == "relative" else ""}" data-mode="relative" '
            'title="Relative vs the active baseline for the current grouping mode">Relative vs baseline</button>'
            "</div>"
        )

    row_group_controls_html = ""
    if len(row_group_modes) > 1:
        pieces = ['<div class="mode-toggle row-group-toggle">']
        for idx, (mode_id, label, _field_key) in enumerate(row_group_modes):
            active = " active" if str(mode_id) == str(default_row_group_mode) else ""
            pieces.append(
                f'<button type="button" class="mode-btn row-group-btn{active}" '
                f'data-row-group-mode="{html.escape(str(mode_id))}">{html.escape(str(label))}</button>'
            )
        pieces.append("</div>")
        row_group_controls_html = "".join(pieces)

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Results Table</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --panel-2: #e8eef5;
      --panel-3: #f8fafc;
      --text: #102133;
      --muted: #55687f;
      --border: #d0dae5;
      --border-2: #bcc9d6;
      --accent: #245f92;
      --sticky: #ffffff;
      --sticky-2: #f2f6fb;
      --group: #e1ebf4;
      --sticky-size-col-width: 4.5rem;
      --sticky-col-1-left: var(--sticky-size-col-width);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 13px/1.35 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .page {{
      padding: 12px 16px 16px;
      max-width: 100%;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin: 0 0 8px;
    }}
    .search {{
      min-width: 240px;
      flex: 1 1 280px;
      max-width: 420px;
      border: 1px solid var(--border-2);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      padding: 8px 10px;
      outline: none;
      font-size: 12px;
    }}
    .search:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(45, 111, 163, 0.12);
    }}
    .hint {{
      color: var(--muted);
      font-size: 11px;
    }}
    .copy-status {{
      color: #245f92;
      font-size: 11px;
      min-width: 9rem;
    }}
    .mode-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px;
      border: 1px solid var(--border-2);
      border-radius: 8px;
      background: var(--panel-3);
    }}
    .mode-btn {{
      appearance: none;
      border: 0;
      background: transparent;
      color: var(--muted);
      font: inherit;
      font-size: 11px;
      font-weight: 600;
      padding: 6px 9px;
      border-radius: 6px;
      cursor: pointer;
    }}
    .mode-btn.active {{
      background: #dbe7f3;
      color: #173c61;
    }}
    .mode-btn:hover {{
      color: #173c61;
    }}
    .table-wrap {{
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: auto;
      background: var(--panel);
      box-shadow: 0 1px 2px rgba(16, 33, 51, 0.04);
    }}
    table {{
      width: max-content;
      min-width: 100%;
      border-collapse: separate;
      border-spacing: 0;
    }}
    th, td {{
      padding: 5px 8px;
      border-right: 1px solid rgba(216, 224, 234, 0.95);
      border-bottom: 1px solid rgba(216, 224, 234, 0.95);
      white-space: nowrap;
      vertical-align: middle;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: var(--panel-2);
      color: #22364d;
      font-size: 11px;
      font-weight: 700;
      text-align: left;
      letter-spacing: 0.01em;
    }}
    th.numeric, td.numeric {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    th.sortable {{
      cursor: pointer;
      user-select: none;
    }}
    th.sortable::after {{
      content: "↕";
      color: #7a8da5;
      font-size: 10px;
      margin-left: 4px;
    }}
    tbody tr:nth-child(odd) td {{
      background: rgba(244, 247, 251, 0.92);
    }}
    tbody tr:hover td {{
      background: rgba(231, 239, 247, 0.98);
    }}
    .sticky-col-0 {{
      position: sticky;
      left: 0;
      z-index: 3;
      background: var(--sticky);
      width: var(--sticky-size-col-width);
      min-width: var(--sticky-size-col-width);
      max-width: var(--sticky-size-col-width);
    }}
    tbody tr:nth-child(odd) .sticky-col-0 {{
      background: var(--sticky-2);
    }}
    tbody tr:hover .sticky-col-0 {{
      background: #e6eef7;
    }}
    th.sticky-col-0 {{
      z-index: 7;
      background: #e7eef6;
    }}
    .sticky-col-1 {{
      position: sticky;
      left: var(--sticky-col-1-left);
      z-index: 3;
      background: var(--sticky);
    }}
    tbody tr:nth-child(odd) .sticky-col-1 {{
      background: var(--sticky-2);
    }}
    tbody tr:hover .sticky-col-1 {{
      background: #e6eef7;
    }}
    th.sticky-col-1 {{
      z-index: 7;
      background: #e7eef6;
    }}
    .group-band {{
      position: sticky;
      top: 0;
      z-index: 8;
      background: var(--group);
      color: #4f647d;
      text-align: center;
      font-size: 10px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      border-bottom: 1px solid var(--border-2);
    }}
    .group-gap {{
      position: sticky;
      top: 0;
      z-index: 8;
      background: var(--panel-2);
      border-bottom: 1px solid var(--border-2);
    }}
    .group-row td {{
      background: #e5edf6 !important;
      color: #50647b;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      padding: 6px 8px 4px;
      border-top: 1px solid var(--border-2);
    }}
    .repeated-header-row th {{
      position: static;
      top: auto;
      z-index: 2;
      background: #edf3f9;
      color: #273d55;
      border-bottom: 1px solid var(--border-2);
    }}
    .repeated-header-row .sticky-col-0 {{
      position: sticky;
      left: 0;
      z-index: 4;
      background: #edf3f9;
    }}
    .repeated-header-row .sticky-col-1 {{
      position: sticky;
      left: var(--sticky-col-1-left);
      z-index: 4;
      background: #edf3f9;
    }}
    .model-name {{
      min-width: 12rem;
      max-width: 18rem;
    }}
    td.copyable {{
      cursor: copy;
      text-decoration: underline dotted rgba(36, 95, 146, 0.45);
      text-underline-offset: 2px;
    }}
    td.copyable:hover {{
      color: #174b78;
      text-decoration-color: rgba(23, 75, 120, 0.9);
    }}
    .size-chip {{
      width: var(--sticky-size-col-width);
      min-width: var(--sticky-size-col-width);
      max-width: var(--sticky-size-col-width);
      color: #4d657f;
      font-size: 11px;
    }}
    .numeric {{
      min-width: 4.5rem;
      max-width: 7ch;
    }}
    #results-table[data-mode="absolute"] td.mode-aware {{
      background-color: var(--abs-bg, transparent) !important;
      border-color: var(--abs-border, rgba(216, 224, 234, 0.95)) !important;
    }}
    #results-table[data-mode="relative"] td.mode-aware {{
      background-color: var(--rel-bg, transparent) !important;
      border-color: var(--rel-border, rgba(216, 224, 234, 0.95)) !important;
    }}
    #results-table[data-mode="absolute"] .value-rel {{
      display: none;
    }}
    #results-table[data-mode="relative"] .value-abs {{
      display: none;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 11px;
      color: #24415f;
    }}
    .footer {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 11px;
    }}
    .hidden-row {{
      display: none;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="controls">
      <input id="table-filter" class="search" type="search" placeholder="Filter rows by text">
      {mode_controls_html}
      {row_group_controls_html}
      <span class="hint">Click a header to sort. Numeric columns sort numerically.</span>
      <span id="copy-status" class="copy-status" aria-live="polite"></span>
    </div>
    <div class="table-wrap">
      <table id="results-table" data-mode="{html.escape(str(default_display_mode))}" data-group-mode="{html.escape(str(default_row_group_mode or 'default'))}">
        <thead>
          {grouped_header_row}
          <tr>{header_cells}</tr>
        </thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </div>
    <div class="footer">
      Generated {html.escape(datetime.utcnow().isoformat(timespec="seconds") + "Z")}
    </div>
  </div>
  <script>
    const table = document.getElementById("results-table");
    const tbody = table.querySelector("tbody");
    const headers = Array.from(table.querySelectorAll("thead tr:last-child th.sortable"));
    const filterInput = document.getElementById("table-filter");
    const copyStatus = document.getElementById("copy-status");
    const modeButtons = Array.from(document.querySelectorAll(".mode-btn[data-mode]"));
    const rowGroupButtons = Array.from(document.querySelectorAll(".row-group-btn"));
    const modeCells = Array.from(table.querySelectorAll("tbody td[data-abs-display]"));
    let currentMode = {json.dumps(default_display_mode)};
    let currentRowGroupMode = {json.dumps(default_row_group_mode)};
    const hasGroupedRows = Boolean(tbody.querySelector("tr[data-row-group]"));
    const repeatedHeaderCells = {json.dumps(repeated_header_cells)};

    const showCopyStatus = (message) => {{
      if (!copyStatus) return;
      copyStatus.textContent = message;
      window.clearTimeout(showCopyStatus.timer);
      showCopyStatus.timer = window.setTimeout(() => {{
        copyStatus.textContent = "";
      }}, 1800);
    }};

    const copyText = async (text) => {{
      if (navigator.clipboard && window.isSecureContext) {{
        await navigator.clipboard.writeText(text);
        return;
      }}
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
    }};

    const getRowGroup = (row) => {{
      if (!currentRowGroupMode) return row.dataset.rowGroup || "";
      const attrName = `rowGroup${{currentRowGroupMode.charAt(0).toUpperCase()}}${{currentRowGroupMode.slice(1)}}`;
      return row.dataset[attrName] || row.dataset.rowGroup || "";
    }};

    const rebuildGroupRows = (rows = null) => {{
      if (!hasGroupedRows) return;
      const dataRows = rows ?? Array.from(tbody.querySelectorAll("tr[data-row-group]"));
      tbody.innerHTML = "";
      let currentGroup = null;
      const columnCount = table.querySelector("thead tr:last-child").cells.length;
      for (const row of dataRows) {{
        const group = getRowGroup(row);
        if (group !== currentGroup) {{
          const groupRow = document.createElement("tr");
          groupRow.className = "group-row";
          const cell = document.createElement("td");
          cell.colSpan = columnCount;
          cell.textContent = group;
          groupRow.appendChild(cell);
          tbody.appendChild(groupRow);
          const headerRow = document.createElement("tr");
          headerRow.className = "repeated-header-row";
          headerRow.innerHTML = repeatedHeaderCells;
          tbody.appendChild(headerRow);
          currentGroup = group;
        }}
        tbody.appendChild(row);
      }}
    }};

    const updateStickyOffsets = () => {{
      const stickySizeCell =
        table.querySelector("thead tr:last-child .sticky-col-0") ||
        table.querySelector("tbody .sticky-col-0");
      if (!stickySizeCell) return;
      const width = Math.ceil(stickySizeCell.getBoundingClientRect().width);
      table.style.setProperty("--sticky-col-1-left", `${{width}}px`);
    }};

    const syncModeCell = (cell) => {{
      const groupMode = currentRowGroupMode || "default";
      const relDisplay = cell.getAttribute(`data-rel-display-${{groupMode}}`) ?? cell.getAttribute("data-rel-display") ?? "";
      const relSort = cell.getAttribute(`data-rel-sort-${{groupMode}}`) ?? cell.getAttribute("data-rel-sort") ?? "";
      const relTitle = cell.getAttribute(`data-rel-title-${{groupMode}}`) ?? cell.getAttribute("data-rel-title") ?? "";
      const relGroup = cell.getAttribute(`data-rel-group-${{groupMode}}`) ?? cell.getAttribute("data-rel-group") ?? "";
      cell.setAttribute("data-rel-display", relDisplay);
      cell.setAttribute("data-rel-sort", relSort);
      cell.setAttribute("data-rel-title", relTitle);
      cell.setAttribute("data-rel-group", relGroup);
      const relSpan = cell.querySelector(".value-rel");
      if (relSpan) relSpan.textContent = relDisplay;
      const sortValue =
        currentMode === "relative"
          ? (relSort || cell.getAttribute("data-abs-sort") || "")
          : (cell.getAttribute("data-abs-sort") || "");
      const titleValue =
        currentMode === "relative"
          ? (relTitle || cell.getAttribute("data-abs-title") || "")
          : (cell.getAttribute("data-abs-title") || "");
      cell.setAttribute("data-sort", sortValue);
      cell.setAttribute("title", titleValue);
    }};

    const applyGroupMode = (groupMode) => {{
      currentRowGroupMode = groupMode || currentRowGroupMode;
      table.dataset.groupMode = currentRowGroupMode || "default";
      for (const cell of modeCells) {{
        const absBg = cell.style.getPropertyValue(`--abs-bg-${{currentRowGroupMode}}`);
        const absBorder = cell.style.getPropertyValue(`--abs-border-${{currentRowGroupMode}}`);
        const relBg = cell.style.getPropertyValue(`--rel-bg-${{currentRowGroupMode}}`);
        const relBorder = cell.style.getPropertyValue(`--rel-border-${{currentRowGroupMode}}`);
        if (absBg) cell.style.setProperty("--abs-bg", absBg);
        else cell.style.removeProperty("--abs-bg");
        if (absBorder) cell.style.setProperty("--abs-border", absBorder);
        else cell.style.removeProperty("--abs-border");
        if (relBg) cell.style.setProperty("--rel-bg", relBg);
        else cell.style.removeProperty("--rel-bg");
        if (relBorder) cell.style.setProperty("--rel-border", relBorder);
        else cell.style.removeProperty("--rel-border");
        syncModeCell(cell);
      }}
    }};

    const applyMode = (mode) => {{
      currentMode = mode;
      table.dataset.mode = mode;
      for (const cell of modeCells) {{
        syncModeCell(cell);
      }}
      for (const button of modeButtons) {{
        button.classList.toggle("active", button.dataset.mode === mode);
      }}
      updateStickyOffsets();
    }};

    filterInput.addEventListener("input", () => {{
      const query = filterInput.value.trim().toLowerCase();
      for (const row of tbody.querySelectorAll("tr[data-row-group]")) {{
        const text = row.innerText.toLowerCase();
        row.classList.toggle("hidden-row", query && !text.includes(query));
      }}
    }});

    modeButtons.forEach((button) => {{
      button.addEventListener("click", (event) => {{
        event.preventDefault();
        event.stopPropagation();
        applyMode(button.dataset.mode || "absolute");
      }});
    }});

    rowGroupButtons.forEach((button) => {{
      button.addEventListener("click", (event) => {{
        event.preventDefault();
        event.stopPropagation();
        currentRowGroupMode = button.dataset.rowGroupMode || currentRowGroupMode;
        rowGroupButtons.forEach((candidate) => {{
          candidate.classList.toggle("active", candidate === button);
        }});
        applyGroupMode(currentRowGroupMode);
        rebuildGroupRows();
      }});
    }});

    table.addEventListener("click", async (event) => {{
      const cell = event.target.closest("td.copyable[data-copy-command]");
      if (!cell || !table.contains(cell)) return;
      const command = cell.dataset.copyCommand || "";
      if (!command) return;
      event.preventDefault();
      event.stopPropagation();
      try {{
        await copyText(command);
        showCopyStatus("Copied view command");
      }} catch (_error) {{
        showCopyStatus(command);
      }}
    }});

    window.addEventListener("resize", updateStickyOffsets);

    headers.forEach((header) => {{
      const cellIndex = header.cellIndex;
      let ascending = true;
      header.addEventListener("click", () => {{
        const type = header.dataset.type || "text";
        const rows = hasGroupedRows
          ? Array.from(tbody.querySelectorAll("tr[data-row-group]"))
          : Array.from(tbody.rows).filter((row) => !row.classList.contains("group-row") && !row.classList.contains("repeated-header-row"));
        const compareRows = (a, b) => {{
          const aCell = a.cells[cellIndex];
          const bCell = b.cells[cellIndex];
          if (!aCell && !bCell) return 0;
          if (!aCell) return 1;
          if (!bCell) return -1;
          const aValue = aCell.dataset.sort || aCell.innerText;
          const bValue = bCell.dataset.sort || bCell.innerText;
          if (type === "number") {{
            const aNum = parseFloat(aValue);
            const bNum = parseFloat(bValue);
            const aOk = !Number.isNaN(aNum);
            const bOk = !Number.isNaN(bNum);
            if (!aOk && !bOk) return 0;
            if (!aOk) return 1;
            if (!bOk) return -1;
            return ascending ? aNum - bNum : bNum - aNum;
          }}
          return ascending
            ? aValue.localeCompare(bValue)
            : bValue.localeCompare(aValue);
        }};
        if (hasGroupedRows) {{
          const groupedRows = new Map();
          const groupOrder = [];
          for (const row of rows) {{
            const group = getRowGroup(row);
            if (!groupedRows.has(group)) {{
              groupedRows.set(group, []);
              groupOrder.push(group);
            }}
            groupedRows.get(group).push(row);
          }}
          const orderedRows = [];
          for (const group of groupOrder) {{
            const groupRows = groupedRows.get(group) || [];
            groupRows.sort(compareRows);
            orderedRows.push(...groupRows);
          }}
          rebuildGroupRows(orderedRows);
        }} else {{
          rows.sort(compareRows);
          tbody.innerHTML = "";
          rows.forEach((row) => tbody.appendChild(row));
        }}
        ascending = !ascending;
      }});
    }});

    applyMode(currentMode);
    applyGroupMode(currentRowGroupMode);
    rebuildGroupRows();
    updateStickyOffsets();
  </script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    target, proc = html_report_target(output_path)
    maybe_open_html(target)
    print(target)
    os.sys.stdout.flush()
    if proc is not None:
        wait_for_server(proc, target)


def parse_ts(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return 0.0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except Exception:
            pass
        try:
            text = text.replace("Z", "+00:00")
            return datetime.fromisoformat(text).timestamp()
        except Exception:
            return 0.0
    return 0.0


def value_text(value):
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def delta_text(value):
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:+.2f}"
    return str(value)


def normalize_relative_value(value, digits: int = 2):
    if value is None or not isinstance(value, (int, float)):
        return value
    threshold = 0.5 * (10 ** (-digits))
    if abs(float(value)) < threshold:
        return 0.0
    return float(value)


def model_size_group(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "other"
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*([bm])(?![A-Za-z])", text, re.IGNORECASE)
    if not match:
        return "other"
    return f"{match.group(1)}{match.group(2).upper()}"


def model_size_group_sort_key(group: str) -> Tuple[float, str]:
    if group == "other":
        return (float("inf"), group)
    match = re.match(r"(\d+(?:\.\d+)?)([BM])", group, re.IGNORECASE)
    if not match:
        return (float("inf"), group)
    scale = 1_000_000_000 if match.group(2).upper() == "B" else 1_000_000
    return (float(match.group(1)) * scale, group)


def model_sort_key(model: str) -> Tuple[float, str, str]:
    group = model_size_group(model)
    group_rank, _label = model_size_group_sort_key(group)
    return (group_rank, group.lower(), model.lower())


def display_model_size_group(group: str) -> str:
    return "Other" if group == "other" else group


def model_identity_text(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    parts = [part for part in text.split("/") if part]
    if len(parts) >= 2 and parts[0] in {"google", "local", "qwen", "swiss-ai"}:
        return "/".join(parts[:2])
    while parts and parts[-1] in {"final", "checkpoint", "checkpoints"}:
        parts.pop()
    if parts:
        return parts[-1]
    return text


def model_family_group(value: str) -> str:
    text = model_identity_text(value)
    if not text:
        return "other"
    name = text.split("/", 1)[-1]
    if re.search(r"ministr(?:al|am)[-_ ]?3", name):
        return "ministral3"
    if text.startswith("qwen/") or re.search(r"^(qwen|q35)(?:[0-9._-]|$)", name):
        return "qwen"
    if text.startswith("google/") and ("gemma" in name or re.search(r"^g3[-_]", name)):
        if re.search(r"gemma[-_]?4", name):
            return "gemma4"
        if re.search(r"gemma[-_]?3", name) or re.search(r"^g3[-_]", name):
            return "gemma3"
        return "gemma"
    if "gemma" in name or re.search(r"^g3[-_]", name):
        if re.search(r"gemma[-_]?4", name):
            return "gemma4"
        if re.search(r"gemma[-_]?3", name) or re.search(r"^g3[-_]", name):
            return "gemma3"
        return "gemma"
    if text.startswith("swiss-ai/apertus") or re.search(r"^apertus(?:[-_0-9]|$)", name):
        return "swiss-ai"
    if "llama" in name:
        return "llama"
    if "/" in text:
        head = text.split("/", 1)[0].strip()
        return head or "other"
    token = re.split(r"[-_ ]+", text.strip())[0]
    return token or "other"


def display_model_family_group(group: str) -> str:
    if not group or group == "other":
        return "Other"
    if group == "gemma3":
        return "Gemma 3"
    if group == "gemma4":
        return "Gemma 4"
    if group == "ministral3":
        return "Ministral 3"
    return group[:1].upper() + group[1:]


def eval_language_group(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    danish_markers = [
        "dala",
        "danish citizen tests",
        "gec",
        "talemaader",
        "ifeval da",
        "multi wiki qa",
        "multiwikiqa",
        "tournament",
        "wmt24",
    ]
    if any(marker in text for marker in danish_markers):
        return "danish"
    return "english"


def display_eval_language_group(group: str) -> str:
    return "Danish" if group == "danish" else "English"


def display_eval_language_median_header(group: str) -> str:
    return "Da med" if group == "danish" else "Eng med"


def is_gemma3_it_baseline_model(model: str, size_group: str) -> bool:
    if not model or size_group == "other":
        return False
    text = re.sub(r"[^a-z0-9.+-]+", "-", str(model).lower())
    size_token = size_group.lower()
    has_gemma3 = bool(re.search(r"gemma[-_]?3", text))
    has_it = bool(re.search(r"(?:^|-)it(?:$|-)", text))
    return has_gemma3 and has_it and size_token in text


def is_gemma4_e4b_it_baseline_model(model: str, size_group: str) -> bool:
    if not model or size_group == "other":
        return False
    text = re.sub(r"[^a-z0-9.+-]+", "-", str(model).lower())
    size_token = size_group.lower()
    has_gemma4 = bool(re.search(r"gemma[-_]?4", text))
    has_e4b = bool(re.search(r"(?:^|-)e?4b(?:$|-)", text))
    has_it = bool(re.search(r"(?:^|-)it(?:$|-)", text))
    return has_gemma4 and has_e4b and has_it and size_token in text


def is_qwen35_baseline_model(model: str, size_group: str) -> bool:
    if not model or size_group == "other":
        return False
    text = str(model).strip().lower()
    return text == f"qwen/qwen3.5-{size_group.lower()}"


def is_swiss_ai_apertus_baseline_model(model: str, size_group: str) -> bool:
    if not model or size_group == "other":
        return False
    text = re.sub(r"[^a-z0-9.+-]+", "-", str(model).lower()).strip("-")
    size_token = size_group.lower()
    return (
        text.startswith("swiss-ai-apertus-")
        and "instruct" in text
        and "2509" in text
        and size_token in text
    )


def is_ministral3_instruct_baseline_model(model: str, size_group: str) -> bool:
    if not model or size_group == "other":
        return False
    raw_text = str(model).strip().lower()
    text = re.sub(r"[^a-z0-9.+-]+", "-", raw_text).strip("-")
    size_token = size_group.lower()
    is_official_ref = raw_text.startswith("mistralai/") or "/" not in raw_text
    return (
        is_official_ref
        and bool(re.search(r"ministral[-_]?3", text))
        and "instruct" in text
        and size_token in text
    )


def family_baseline_missing_title(size_group: str, family_group: str) -> str:
    size_label = display_model_size_group(size_group)
    if family_group in {"gemma", "gemma3"}:
        return f"No Gemma 3 IT baseline for {size_label}"
    if family_group == "gemma4":
        return f"No Gemma 4 E4B IT baseline for {size_label}"
    if family_group == "ministral3":
        return f"No Ministral 3 Instruct baseline for {size_label}"
    if family_group == "qwen":
        return f"No Qwen baseline for {size_label}"
    return f"No {display_model_family_group(family_group)} baseline for {size_label}"


def compact_label(value: str, max_len: int = 10) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"

    for sep in ("/", ":", "|"):
        if sep in text:
            text = text.split(sep)[-1]

    text = text.replace("__", "_").replace("-", "_")
    tokens = [token for token in text.split("_") if token]
    if not tokens:
        return trim(text, max_len)

    if len(tokens) == 1:
        base = tokens[0]
    elif len(tokens[0]) >= max_len - 2:
        base = tokens[0]
    elif len(tokens) == 2:
        candidate = f"{tokens[0]}_{tokens[1]}"
        if len(candidate) <= max_len:
            base = candidate
        else:
            base = tokens[0]
    else:
        # Keep first token and initials from the rest.
        initials = "".join(token[0] for token in tokens[1:] if token)
        candidate = f"{tokens[0]}_{initials}" if initials else tokens[0]
        if len(candidate) <= max_len:
            base = candidate
        else:
            base = tokens[0]

    return base if len(base) <= max_len else base[:max_len]


def is_local_path_reference(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return text.startswith(("/", "./", "../", "~/"))


def model_label_from_ref(model_ref: str) -> str:
    ref = str(model_ref or "").strip()
    if not ref:
        return "model"

    for prefix in ("vllm/", "openai/"):
        if ref.startswith(prefix):
            ref = ref[len(prefix) :]
            break

    parts = [part for part in ref.split("/") if part]
    if len(parts) >= 2:
        base_name = parts[-1]
        parent_name = parts[-2]
    elif parts:
        base_name = parts[-1]
        parent_name = ""
    else:
        base_name = ref
        parent_name = ""

    if (
        base_name in {"final", "latest", "last"}
        or base_name.startswith("checkpoint-")
        or base_name.startswith("step-")
        or base_name.startswith("epoch-")
    ):
        label = f"{parent_name}-{base_name}" if parent_name else base_name
    else:
        label = base_name

    cleaned = []
    for char in label:
        if char.isalnum() or char in "._-":
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("._-")
    return result or "model"


def canonical_model_id(model_id: str, model_name: str) -> str:
    raw_id = str(model_id or "").strip()
    raw_name = str(model_name or "").strip()

    for candidate in (raw_name, raw_id):
        if candidate.startswith("vllm/"):
            suffix = candidate[len("vllm/") :]
            if is_local_path_reference(suffix):
                return f"local/{model_label_from_ref(candidate)}"
        if is_local_path_reference(candidate):
            return f"local/{model_label_from_ref(candidate)}"

    return raw_id or raw_name or "-"


def unique_compact_labels(values: List[str], max_len: int) -> List[str]:
    used: Set[str] = set()
    labels: List[str] = []
    for value in values:
        base = compact_label(value, max_len=max_len)
        label = base
        suffix = 2
        while label in used:
            tail = str(suffix)
            keep = max(1, max_len - len(tail))
            label = f"{base[:keep]}{tail}"
            suffix += 1
        used.add(label)
        labels.append(label)
    return labels


def trim(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "~"


def row_from_eval_result(record, eval_result, path, file_ts):
    evaluation_id = record.get("evaluation_id") or "-"
    source_metadata = record.get("source_metadata") or {}
    model_info = record.get("model_info") or {}
    score_details = eval_result.get("score_details") or {}
    details = score_details.get("details") or {}
    source_data = eval_result.get("source_data") or {}
    source_data_details = source_data.get("additional_details") or {}
    metric_config = eval_result.get("metric_config") or {}
    metric_config_details = metric_config.get("additional_details") or {}
    uncertainty = score_details.get("uncertainty") or {}

    evaluation_name = str(eval_result.get("evaluation_name") or "")
    parts = [p for p in evaluation_name.split("/") if p]

    task = details.get("task")
    scorer = details.get("scorer")
    metric = details.get("metric")

    if not task:
        task = parts[0] if parts else "<unknown>"

    if not scorer:
        scorer = parts[-2] if len(parts) >= 3 else "-"

    if not metric:
        if len(parts) >= 1:
            metric = parts[-1]
        else:
            metric = metric_config.get("evaluation_description") or "-"

    # EuroEval exports should be grouped by dataset in the task column.
    # Older exports may encode the coarse task-group (e.g. "knowledge") in
    # details.task while source_data carries the concrete dataset name.
    if str(source_metadata.get("source_name") or "") == "euroeval":
        dataset_name = source_data.get("dataset_name")
        if dataset_name:
            task = dataset_name
        if not scorer or scorer == "-":
            scorer = details.get("scorer") or source_data_details.get("task") or scorer

    ts = (
        parse_ts(eval_result.get("evaluation_timestamp"))
        or parse_ts(record.get("evaluation_timestamp"))
        or parse_ts(record.get("retrieved_timestamp"))
        or file_ts
    )

    n = uncertainty.get("num_samples")
    total = n
    run = source_metadata.get("source_name") or (parts[0] if parts else "-")
    view_run_label = extract_view_run_label(record, eval_result)
    view_command = view_command_for_run_label(view_run_label)

    raw_model = model_info.get("id") or model_info.get("name") or "-"
    reported_model = model_info.get("name") or raw_model
    model = canonical_model_id(str(raw_model), str(reported_model))
    preferred_for_display = str(metric_config_details.get("preferred_for_display") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    return {
        "run": run,
        "path": path,
        "ts": ts,
        "task": task,
        "scorer": scorer,
        "metric": metric,
        "value": score_details.get("score"),
        "n": n,
        "total": total,
        "model": model,
        "reported_model": reported_model,
        "evaluation_id": evaluation_id,
        "preferred_for_display": preferred_for_display,
        "view_run_label": view_run_label,
        "view_command": view_command,
    }


def extract_view_run_label(record, eval_result) -> Optional[str]:
    values: List[str] = []

    def collect(value):
        if value is None:
            return
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(eval_result.get("source_data") or {})
    collect((eval_result.get("generation_config") or {}).get("additional_details") or {})
    collect(record.get("detailed_evaluation_results") or {})

    for text in values:
        # LUMI eval runs set task caches below /overlay/cache/dfm-evals-<run-label>/xdg-cache.
        match = re.search(r"(?:^|/)dfm-evals-([^/]+)/xdg-cache(?:/|$)", text)
        if match:
            return match.group(1)
        # Server log provenance, when present, points back to artifacts/evals/logs/<run-label>/...
        match = re.search(r"/evals/logs/([^/]+)/", text)
        if match:
            return match.group(1)

    return None


def view_command_for_run_label(run_label: Optional[str]) -> Optional[str]:
    if not run_label:
        return None
    return f"./lumi/view.sh start --label {shlex.quote(run_label)}"


def latest_wmt24_comet_summary_path() -> Optional[str]:
    root = os.environ.get("POST_ARTIFACT_ROOT")
    if not root:
        return None
    pattern = os.path.join(root, "evals", "maintenance", "wmt24-comet-rescore-*", "summary.json")
    candidates = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: os.path.getmtime(path))


def should_include_wmt24_comet_summary() -> bool:
    root = os.environ.get("POST_ARTIFACT_ROOT")
    if not root:
        return False
    default_eee_root = os.path.normpath(os.path.join(root, "evals", "eee", "data"))
    for source_dir in source_dirs:
        norm_source = os.path.normpath(source_dir)
        if norm_source == default_eee_root or norm_source.startswith(default_eee_root + os.sep):
            return True
    return False


def rows_from_wmt24_comet_summary(path: str) -> List[dict]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            summaries = json.load(file)
    except Exception:
        return []

    try:
        file_ts = os.path.getmtime(path)
    except Exception:
        file_ts = 0.0

    result = []
    for summary in summaries or []:
        model_id = str(summary.get("model_id") or "").strip()
        value = summary.get("comet_mean")
        if not model_id or not isinstance(value, (int, float)):
            continue
        source_file = str(summary.get("source_file") or path)
        evaluation_id = summary.get("evaluation_id") or source_file
        row_ts = parse_ts(str(evaluation_id).rsplit("/", 1)[-1]) or file_ts
        result.append(
            {
                "run": "wmt24-comet-rescore",
                "path": path,
                "ts": row_ts,
                "task": "wmt24pp-en-da",
                "scorer": "comet",
                "metric": "mean",
                "value": float(value),
                "n": summary.get("n"),
                "total": summary.get("n"),
                "model": canonical_model_id(model_id, model_id),
                "reported_model": model_id,
                "evaluation_id": evaluation_id,
                "preferred_for_display": False,
            }
        )
    return result


files = []
for source_dir in source_dirs:
    pattern = os.path.join(source_dir, "**", "*.json")
    for path in glob.glob(pattern, recursive=True):
        base = os.path.basename(path)
        if base.startswith("_"):
            continue
        files.append(os.path.normpath(path))

files = sorted(set(files))
if not files:
    print("No readable EEE .json files found.")
    raise SystemExit(0)

file_stats = []
for path in files:
    try:
        file_stats.append((os.path.getmtime(path), path))
    except Exception:
        continue

if not file_stats:
    print("No readable EEE .json files found.")
    raise SystemExit(0)

if selector == "latest":
    newest_mtime = max(ts for ts, _ in file_stats)
    threshold = newest_mtime - latest_window_seconds
    selected = [path for ts, path in file_stats if ts >= threshold]
    selected_set = set(selected)
    file_stats = [(ts, path) for ts, path in file_stats if path in selected_set]

rows = []
for file_ts, path in sorted(file_stats, key=lambda x: x[1]):
    try:
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception:
        continue

    eval_results = record.get("evaluation_results") or []
    if not eval_results:
        model_info = record.get("model_info") or {}
        rows.append(
            {
                "run": (record.get("source_metadata") or {}).get("source_name") or "-",
                "path": path,
                "ts": parse_ts(record.get("evaluation_timestamp"))
                or parse_ts(record.get("retrieved_timestamp"))
                or file_ts,
                "task": "<unknown>",
                "scorer": "-",
                "metric": "-",
                "value": None,
                "n": None,
                "total": None,
                "model": model_info.get("id") or model_info.get("name") or "-",
                "reported_model": model_info.get("name") or "-",
                "evaluation_id": record.get("evaluation_id") or "-",
            }
        )
        continue

    for eval_result in eval_results:
        rows.append(row_from_eval_result(record, eval_result, path, file_ts))

wmt24_comet_summary_path = latest_wmt24_comet_summary_path() if should_include_wmt24_comet_summary() else None
if wmt24_comet_summary_path:
    rows.extend(rows_from_wmt24_comet_summary(wmt24_comet_summary_path))

if not rows:
    print("No rows parsed from EEE records.")
    raise SystemExit(0)

if not compare_models:
    rows.sort(key=lambda r: (r["run"], r["task"], r["scorer"], r["metric"], r["model"]))
    if fmt == "json":
        print(json.dumps(rows, indent=2))
        raise SystemExit(0)
    if fmt == "csv":
        fieldnames = ["run", "task", "scorer", "metric", "value", "n", "total", "model"]
        writer = csv.DictWriter(os.sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
        raise SystemExit(0)
    if fmt == "html":
        html_columns = [
            ("run", "Run", 10, "left"),
            ("task", "Task", 18, "left"),
            ("scorer", "Scorer", 10, "left"),
            ("metric", "Metric", 10, "left"),
            ("value", "Value", 7, "right"),
            ("n", "N", 5, "right"),
            ("total", "Total", 5, "right"),
            ("model", "Model", (12, 18), "left"),
        ]
        html_rows = []
        for row in rows:
            html_row = dict(row)
            if row.get("view_run_label"):
                html_row["run"] = row.get("view_run_label")
            if row.get("view_command"):
                html_row["__copy_run"] = row.get("view_command")
            html_rows.append(html_row)
        emit_html_report(
            html_columns,
            html_rows,
            numeric_keys={"value", "n", "total"},
            sticky_columns=1,
        )
        raise SystemExit(0)

    columns = [
        ("run", "Run", 14, "left"),
        ("task", "Task", 16, "left"),
        ("scorer", "Scorer", 14, "left"),
        ("metric", "Metric", 16, "left"),
        ("value", "Value", 6, "right"),
        ("n", "N", 4, "right"),
        ("total", "Total", 5, "right"),
        ("model", "Model", 72, "left"),
    ]
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "run": row.get("run", "-"),
                "task": row.get("task", "-"),
                "scorer": row.get("scorer", "-"),
                "metric": row.get("metric", "-"),
                "value": value_text(row.get("value")),
                "n": row.get("n", "-"),
                "total": row.get("total", "-"),
                "model": row.get("model", "-"),
            }
        )
    emit_string_table(columns, display_rows, title=f"EEE Results ({len(display_rows)} rows)")
    raise SystemExit(0)

# compare_models path
latest = {}
for row in rows:
    key = (row["model"], row["task"], row["scorer"], row["metric"])
    prev = latest.get(key)
    if prev is None or row["ts"] >= prev["ts"]:
        latest[key] = row

latest_rows = list(latest.values())
models = sorted({row["model"] for row in latest_rows}, key=model_sort_key)
tasks = sorted({row["task"] for row in latest_rows})
view_command_by_model: Dict[str, str] = {}
for row in sorted(latest_rows, key=lambda item: item.get("ts") or 0.0):
    command = row.get("view_command")
    model = row.get("model")
    if command and model:
        view_command_by_model[model] = command
model_size_labels = {model: model_size_group(model) for model in models}
baseline_models_by_size: Dict[str, str] = {}
baseline_models_by_size_family: Dict[Tuple[str, str], str] = {}
for model in models:
    size_group = model_size_labels[model]
    if size_group in baseline_models_by_size:
        continue
    if is_gemma3_it_baseline_model(model, size_group):
        baseline_models_by_size[size_group] = model
for model in models:
    size_group = model_size_labels[model]
    family_group = model_family_group(model)
    key = (size_group, family_group)
    if key in baseline_models_by_size_family:
        continue
    if family_group in {"gemma", "gemma3"} and is_gemma3_it_baseline_model(model, size_group):
        baseline_models_by_size_family[key] = model
    elif family_group == "gemma4" and is_gemma4_e4b_it_baseline_model(model, size_group):
        baseline_models_by_size_family[key] = model
    elif family_group == "qwen" and is_qwen35_baseline_model(model, size_group):
        baseline_models_by_size_family[key] = model
    elif family_group == "swiss-ai" and is_swiss_ai_apertus_baseline_model(model, size_group):
        baseline_models_by_size_family[key] = model
    elif family_group == "ministral3" and is_ministral3_instruct_baseline_model(model, size_group):
        baseline_models_by_size_family[key] = model

value_by_key = {
    (row["model"], row["task"], row["scorer"], row["metric"]): row.get("value")
    for row in latest_rows
}

combos_by_task = {}
for row in latest_rows:
    combos_by_task.setdefault(row["task"], set()).add((row["scorer"], row["metric"]))

preferred_combo_keys = {
    (row["task"], row["scorer"], row["metric"])
    for row in latest_rows
    if row.get("preferred_for_display")
}


def combo_rank(task: str, combo: Tuple[str, str]):
    scorer, metric = combo
    task_scorers = preferred_scorers_by_task.get(task, [])
    scorer_pri = task_scorers.index(scorer) if scorer in task_scorers else len(task_scorers)
    preferred_pri = 0 if (task, scorer, metric) in preferred_combo_keys else 1
    metric_pri = preferred_rank.get(metric, len(preferred_rank) + 100)
    coverage = 0
    for model in models:
        value = value_by_key.get((model, task, scorer, metric))
        if value is not None:
            coverage += 1
    return (scorer_pri, preferred_pri, metric_pri, -coverage, scorer, metric)


def unique_model_labels(model_names: List[str]) -> Dict[str, str]:
    parts = [m.split("/") for m in model_names]
    max_depth = max((len(p) for p in parts), default=1)
    labels: Dict[str, str] = {}
    depth = 1
    while depth <= max_depth:
        seen = {}
        collision = False
        for model, p in zip(model_names, parts):
            label = "/".join(p[-depth:]) if len(p) >= depth else "/".join(p)
            labels[model] = label
            seen[label] = seen.get(label, 0) + 1
            if seen[label] > 1:
                collision = True
        if not collision:
            return labels
        depth += 1
    return {m: m for m in model_names}


table_rows = []
if primary_only:
    for task in tasks:
        combos = sorted(combos_by_task.get(task, []), key=lambda c: combo_rank(task, c))
        required_scorers = required_primary_scorers_by_task.get(task, [])
        if required_scorers:
            combos = [combo for combo in combos if combo[0] in required_scorers]
        if not combos:
            continue
        scorer, metric = combos[0]
        row = {"task": task, "scorer": scorer, "metric": metric}
        for model in models:
            row[model] = value_by_key.get((model, task, scorer, metric))
            size_group = model_size_labels[model]
            absolute_group = f"{size_group}|{task}|{scorer}|{metric}"
            relative_group = f"{size_group}|{task}|{scorer}|{metric}"
            baseline_model = baseline_models_by_size.get(size_group)
            baseline_value = (
                value_by_key.get((baseline_model, task, scorer, metric))
                if baseline_model
                else None
            )
            current_value = row[model]
            if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                row[f"__relative_{model}"] = float(current_value) - float(baseline_value)
                row[f"__relative_title_{model}"] = f"{model} vs {baseline_model}"
            elif baseline_model:
                row[f"__relative_{model}"] = None
                row[f"__relative_title_{model}"] = f"{model} vs {baseline_model}"
            else:
                row[f"__relative_{model}"] = None
                row[f"__relative_title_{model}"] = f"No Gemma 3 IT baseline for {display_model_size_group(size_group)}"
            row[f"__absolute_group_{model}"] = absolute_group
            row[f"__relative_group_{model}"] = relative_group
        table_rows.append(row)
else:
    for task in tasks:
        combos = sorted(combos_by_task.get(task, []), key=lambda c: (c[0], c[1]))
        for scorer, metric in combos:
            row = {"task": task, "scorer": scorer, "metric": metric}
            for model in models:
                row[model] = value_by_key.get((model, task, scorer, metric))
                size_group = model_size_labels[model]
                absolute_group = f"{size_group}|{task}|{scorer}|{metric}"
                relative_group = f"{size_group}|{task}|{scorer}|{metric}"
                baseline_model = baseline_models_by_size.get(size_group)
                baseline_value = (
                    value_by_key.get((baseline_model, task, scorer, metric))
                    if baseline_model
                    else None
                )
                current_value = row[model]
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    row[f"__relative_{model}"] = float(current_value) - float(baseline_value)
                    row[f"__relative_title_{model}"] = f"{model} vs {baseline_model}"
                elif baseline_model:
                    row[f"__relative_{model}"] = None
                    row[f"__relative_title_{model}"] = f"{model} vs {baseline_model}"
                else:
                    row[f"__relative_{model}"] = None
                    row[f"__relative_title_{model}"] = f"No Gemma 3 IT baseline for {display_model_size_group(size_group)}"
                row[f"__absolute_group_{model}"] = absolute_group
                row[f"__relative_group_{model}"] = relative_group
            table_rows.append(row)

if orientation == "task-rows":
    if fmt == "json":
        print(json.dumps({"orientation": orientation, "models": models, "rows": table_rows}, indent=2))
        raise SystemExit(0)

    if fmt == "csv":
        fieldnames = ["task", "scorer", "metric"] + models
        writer = csv.DictWriter(os.sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_rows:
            out = {"task": row["task"], "scorer": row["scorer"], "metric": row["metric"]}
            for model in models:
                out[model] = row.get(model)
            writer.writerow(out)
        raise SystemExit(0)
    if fmt == "html":
        model_labels = unique_model_labels(models)
        model_size_labels = {model: model_size_group(model) for model in models}
        short_model_headers = dict(
            zip(models, unique_compact_labels([model_labels[model] for model in models], max_len=10))
        )
        group_spans = []
        current_group = None
        current_count = 0
        for model in models:
            group = model_size_labels[model]
            if group != current_group:
                if current_group is not None:
                    group_spans.append((display_model_size_group(current_group), current_count))
                current_group = group
                current_count = 1
            else:
                current_count += 1
        if current_group is not None:
            group_spans.append((display_model_size_group(current_group), current_count))
        html_columns = [
            ("task", "Task", 16, "left"),
            ("metric", "Metric", 9, "left"),
            ("scorer", "Scorer", 9, "left"),
        ]
        for model in models:
            html_columns.append((model, short_model_headers[model], 7, "right", model))
        emit_html_report(
            html_columns,
            table_rows,
            numeric_keys=set(models),
            relative_numeric_keys=set(models),
            heatmap_mode="row",
            sticky_columns=1,
            column_groups=group_spans,
        )
        raise SystemExit(0)

    columns = [
        ("task", "Task", 14, "left"),
        ("metric", "Metric", 12, "left"),
        ("scorer", "Scorer", 12, "left"),
    ]
    model_labels = unique_model_labels(models)
    for model in models:
        columns.append((model, model_labels[model], (14, 22), "right"))

    display_rows = []
    for row in table_rows:
        out = {
            "task": compact_label(row.get("task", "-"), 14),
            "metric": compact_label(row.get("metric", "-"), 12),
            "scorer": compact_label(row.get("scorer", "-"), 12),
        }
        for model in models:
            out[model] = value_text(row.get(model))
        display_rows.append(out)

    metric_mode = "primary" if primary_only else "all"
    emit_string_table(
        columns,
        display_rows,
        title=f"Model Comparison (task rows, {metric_mode} metrics)",
    )
    raise SystemExit(0)

# model-rows orientation
raw_col_defs = []
seen = set()
for row in table_rows:
    if primary_only:
        base = row["task"]
    else:
        base = f"{row['task']}|{row['scorer']}|{row['metric']}"
    col = base
    suffix = 2
    while col in seen:
        col = f"{base}#{suffix}"
        suffix += 1
    seen.add(col)
    raw_col_defs.append((col, base, row))

english_col_defs = []
danish_col_defs = []
for col_key, base, row in raw_col_defs:
    lang = eval_language_group(base)
    if lang == "danish":
        danish_col_defs.append((col_key, base, row, lang))
    else:
        english_col_defs.append((col_key, base, row, lang))

col_defs = []
if english_col_defs:
    col_defs.append(("__median_english", "English median", None, "english", "median"))
if danish_col_defs:
    col_defs.append(("__median_danish", "Danish median", None, "danish", "median"))
if english_col_defs:
    col_defs.extend((col_key, base, row, lang, "task") for col_key, base, row, lang in english_col_defs)
if danish_col_defs:
    col_defs.extend((col_key, base, row, lang, "task") for col_key, base, row, lang in danish_col_defs)

model_rows = []
for model in models:
    out = {"model": model}
    english_values = []
    danish_values = []
    for col_key, _base, source_row, lang, kind in col_defs:
        if kind == "task":
            value = source_row.get(model)
            out[col_key] = value
            if isinstance(value, (int, float)):
                if lang == "danish":
                    danish_values.append(float(value))
                else:
                    english_values.append(float(value))
        else:
            out[col_key] = None
    if english_values:
        out["__median_english"] = statistics.median(english_values)
    if danish_values:
        out["__median_danish"] = statistics.median(danish_values)
    model_rows.append(out)

model_rows_by_model = {row["model"]: row for row in model_rows}
for row in model_rows:
    row_model = row["model"]
    size_group = model_size_labels.get(row_model, "other")
    family_group = model_family_group(row_model)
    baseline_model = baseline_models_by_size.get(size_group)
    baseline_row = model_rows_by_model.get(baseline_model) if baseline_model else None
    family_baseline_model = baseline_models_by_size_family.get((size_group, family_group))
    family_baseline_row = model_rows_by_model.get(family_baseline_model) if family_baseline_model else None
    for col_key, _base, _source_row, lang, _kind in col_defs:
        absolute_group = f"{size_group}|{lang}|{col_key}"
        absolute_group_sizefamily = f"{size_group}|{family_group}|{lang}|{col_key}"
        relative_group = f"{size_group}|{lang}|{col_key}"
        relative_group_sizefamily = f"{size_group}|{family_group}|{lang}|{col_key}"
        current_value = row.get(col_key)
        baseline_value = baseline_row.get(col_key) if baseline_row else None
        family_baseline_value = family_baseline_row.get(col_key) if family_baseline_row else None
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            row[f"__relative_{col_key}"] = float(current_value) - float(baseline_value)
            row[f"__relative_size_{col_key}"] = row[f"__relative_{col_key}"]
            row[f"__relative_title_{col_key}"] = f"{row_model} vs {baseline_model}"
            row[f"__relative_title_size_{col_key}"] = row[f"__relative_title_{col_key}"]
        elif baseline_model:
            row[f"__relative_{col_key}"] = None
            row[f"__relative_size_{col_key}"] = None
            row[f"__relative_title_{col_key}"] = f"{row_model} vs {baseline_model}"
            row[f"__relative_title_size_{col_key}"] = row[f"__relative_title_{col_key}"]
        else:
            row[f"__relative_{col_key}"] = None
            row[f"__relative_size_{col_key}"] = None
            row[f"__relative_title_{col_key}"] = f"No Gemma 3 IT baseline for {display_model_size_group(size_group)}"
            row[f"__relative_title_size_{col_key}"] = row[f"__relative_title_{col_key}"]
        if isinstance(current_value, (int, float)) and isinstance(family_baseline_value, (int, float)):
            row[f"__relative_sizefamily_{col_key}"] = float(current_value) - float(family_baseline_value)
            row[f"__relative_title_sizefamily_{col_key}"] = f"{row_model} vs {family_baseline_model}"
        elif family_baseline_model:
            row[f"__relative_sizefamily_{col_key}"] = None
            row[f"__relative_title_sizefamily_{col_key}"] = f"{row_model} vs {family_baseline_model}"
        else:
            row[f"__relative_sizefamily_{col_key}"] = None
            row[f"__relative_title_sizefamily_{col_key}"] = family_baseline_missing_title(size_group, family_group)
        row[f"__absolute_group_{col_key}"] = absolute_group
        row[f"__absolute_group_size_{col_key}"] = absolute_group
        row[f"__absolute_group_sizefamily_{col_key}"] = absolute_group_sizefamily
        row[f"__relative_group_{col_key}"] = relative_group
        row[f"__relative_group_size_{col_key}"] = relative_group
        row[f"__relative_group_sizefamily_{col_key}"] = relative_group_sizefamily

if fmt == "json":
    print(json.dumps({"orientation": orientation, "columns": [c for c, _b, _r, _lang, _kind in col_defs], "rows": model_rows}, indent=2))
    raise SystemExit(0)

if fmt == "csv":
    fieldnames = ["model"] + [c for c, _b, _r, _lang, _kind in col_defs]
    writer = csv.DictWriter(os.sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in model_rows:
        writer.writerow(row)
    raise SystemExit(0)
if fmt == "html":
    model_labels = unique_model_labels(models)
    model_sizes = {model: model_size_group(model) for model in models}
    task_header_defs = [(col_key, base, lang, kind) for col_key, base, _r, lang, kind in col_defs]
    task_headers = [base for _k, base, _lang, kind in task_header_defs if kind == "task"]
    short_task_headers = unique_compact_labels(task_headers, max_len=9)
    short_header_by_key = {}
    task_idx = 0
    for col_key, _base, lang, kind in task_header_defs:
        if kind == "median":
            short_header_by_key[col_key] = display_eval_language_median_header(lang)
        else:
            short_header_by_key[col_key] = short_task_headers[task_idx]
            task_idx += 1
    group_spans = []
    median_count = int(bool(english_col_defs)) + int(bool(danish_col_defs))
    if median_count:
        group_spans.append(("Medians", median_count))
    if english_col_defs:
        group_spans.append(("English", len(english_col_defs)))
    if danish_col_defs:
        group_spans.append(("Danish", len(danish_col_defs)))
    html_columns = [("size", "Size", (3, 5), "left"), ("model", "Model", (12, 18), "left")]
    for col_key, base, _source_row, _lang, kind in col_defs:
        title = base
        header = short_header_by_key[col_key]
        html_columns.append((col_key, header, 7, "right", title))
    display_rows = []
    for row in model_rows:
        raw_model = row.get("model", "-")
        size_display = display_model_size_group(model_sizes.get(raw_model, "other"))
        family_display = display_model_family_group(model_family_group(raw_model))
        out = {
            "size": size_display,
            "size_family": f"{size_display} / {family_display}",
            "model": model_labels.get(raw_model, raw_model),
            "__title_model": raw_model,
            "__copy_model": view_command_by_model.get(raw_model),
        }
        for col_key, _base, _source_row, _lang, _kind in col_defs:
            out[col_key] = row.get(col_key)
            out[f"__absolute_group_{col_key}"] = row.get(f"__absolute_group_{col_key}")
            out[f"__absolute_group_size_{col_key}"] = row.get(f"__absolute_group_size_{col_key}")
            out[f"__absolute_group_sizefamily_{col_key}"] = row.get(f"__absolute_group_sizefamily_{col_key}")
            out[f"__relative_{col_key}"] = row.get(f"__relative_{col_key}")
            out[f"__relative_title_{col_key}"] = row.get(f"__relative_title_{col_key}")
            out[f"__relative_size_{col_key}"] = row.get(f"__relative_size_{col_key}")
            out[f"__relative_title_size_{col_key}"] = row.get(f"__relative_title_size_{col_key}")
            out[f"__relative_sizefamily_{col_key}"] = row.get(f"__relative_sizefamily_{col_key}")
            out[f"__relative_title_sizefamily_{col_key}"] = row.get(f"__relative_title_sizefamily_{col_key}")
            out[f"__relative_group_{col_key}"] = row.get(f"__relative_group_{col_key}")
            out[f"__relative_group_size_{col_key}"] = row.get(f"__relative_group_size_{col_key}")
            out[f"__relative_group_sizefamily_{col_key}"] = row.get(f"__relative_group_sizefamily_{col_key}")
        display_rows.append(out)
    emit_html_report(
        html_columns,
        display_rows,
        numeric_keys={c for c, _b, _r, _lang, _kind in col_defs},
        relative_numeric_keys={c for c, _b, _r, _lang, _kind in col_defs},
        heatmap_mode="column",
        sticky_columns=2,
        row_group_key="size",
        row_group_modes=[("size", "Size", "size"), ("sizefamily", "Size+Family", "size_family")],
        color_group_modes=["size", "sizefamily"],
        column_groups=group_spans,
    )
    raise SystemExit(0)

model_labels = unique_model_labels(models)
columns = [("model", "Model", (28, 64), "left")]
task_headers = [base for _k, base, _r, _lang, kind in col_defs if kind == "task"]
short_task_headers = unique_compact_labels(task_headers, max_len=6)
short_header_by_key = {}
task_idx = 0
for col_key, _base, _source_row, lang, kind in col_defs:
    if kind == "median":
        short_header_by_key[col_key] = display_eval_language_median_header(lang)
    else:
        short_header_by_key[col_key] = short_task_headers[task_idx]
        task_idx += 1
for col_key, _base, _source_row, _lang, _kind in col_defs:
    header = short_header_by_key[col_key]
    columns.append((col_key, header, 6, "right"))

display_rows = []
for row in model_rows:
    out = {"model": model_labels.get(row.get("model", "-"), row.get("model", "-"))}
    for col_key, _base, _source_row, _lang, _kind in col_defs:
        out[col_key] = value_text(row.get(col_key))
    display_rows.append(out)

metric_mode = "primary" if primary_only else "all"
emit_string_table(
    columns,
    display_rows,
    title=f"Model Comparison (model rows, {metric_mode} metrics)",
)
PY
