#!/bin/bash
# Print aggregate task/scorer/metric results from Every Eval Ever JSON artifacts.
# Supports model-comparison pivot table across multiple exports.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EEE_DATA_ROOT_HOST=${EEE_DATA_ROOT_HOST:-$REPO_ROOT/logs/every_eval_ever/data}
LATEST_WINDOW_SECONDS=${LATEST_WINDOW_SECONDS:-120}
RESULTS_TABLE_WIDTH=${RESULTS_TABLE_WIDTH:-140}

SELECTOR="latest"
SELECTOR_SET=0
RUN_LABEL=""
LOG_DIR=""

COMPARE_MODELS=0
PRIMARY_ONLY=1
FORMAT=${FORMAT:-table}
COMPARE_ORIENTATION=${COMPARE_ORIENTATION:-model-rows}

usage() {
  cat <<'USAGE'
Usage:
  ./lumi/results_table.sh [selector options] [view options]

Selector options:
  --latest             Use newest EEE export batch under data root (default)
  --all-runs           Use all EEE records under data root
  --run-label <label>  Use legacy logs/evals-runs/<label>/every_eval_ever if present,
                       otherwise EEE subdir <data_root>/<label> if present
  --log-dir <path>     Use explicit EEE directory path

View options:
  --compare-models     Pivot table by model (columns are models, rows are tasks)
  --primary-only       For compare mode: use one primary metric per task (default)
  --all-metrics        For compare mode: include every scorer+metric row per task
  --model-rows         For compare mode: rows=models, columns=tasks (default)
  --task-rows          For compare mode: rows=tasks, columns=models
  --format <fmt>       Output format: table|csv|json (default: table)
  --help               Show help

Environment overrides:
  EEE_DATA_ROOT_HOST     Host EEE data root (default: ./logs/every_eval_ever/data)
  LATEST_WINDOW_SECONDS  Window for --latest selection by file mtime (default: 120)
  RESULTS_TABLE_WIDTH    Preferred rich table width for --format table (default: 140)

Examples:
  ./lumi/results_table.sh --latest
  ./lumi/results_table.sh --all-runs
  ./lumi/results_table.sh --compare-models --all-runs
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
  local legacy="$REPO_ROOT/logs/evals-runs/$label/every_eval_ever"
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
  table|csv|json)
    ;;
  *)
    die "invalid --format: $FORMAT (expected table|csv|json)"
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
python3 - <<'PY'
import csv
import glob
import json
import os
from datetime import datetime

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
            max_width = cap if isinstance(cap, int) and cap > 0 else None
            table.add_column(
                str(header),
                justify=justify,
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
        widths[key] = min(max_len, cap)

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


def unique_compact_labels(values: list[str], max_len: int) -> list[str]:
    used: set[str] = set()
    labels: list[str] = []
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

    model = model_info.get("id") or model_info.get("name") or "-"
    reported_model = model_info.get("name") or model

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
    }


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

    columns = [
        ("run", "Run", 14, "left"),
        ("task", "Task", 16, "left"),
        ("scorer", "Scorer", 14, "left"),
        ("metric", "Metric", 16, "left"),
        ("value", "Value", 6, "right"),
        ("n", "N", 4, "right"),
        ("total", "Total", 5, "right"),
        ("model", "Model", 58, "left"),
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
models = sorted({row["model"] for row in latest_rows})
tasks = sorted({row["task"] for row in latest_rows})

value_by_key = {
    (row["model"], row["task"], row["scorer"], row["metric"]): row.get("value")
    for row in latest_rows
}

combos_by_task = {}
for row in latest_rows:
    combos_by_task.setdefault(row["task"], set()).add((row["scorer"], row["metric"]))


def combo_rank(task: str, combo: tuple[str, str]):
    scorer, metric = combo
    metric_pri = preferred_rank.get(metric, len(preferred_rank) + 100)
    coverage = 0
    for model in models:
        value = value_by_key.get((model, task, scorer, metric))
        if value is not None:
            coverage += 1
    return (metric_pri, -coverage, scorer, metric)


def unique_model_labels(model_names: list[str]) -> dict[str, str]:
    parts = [m.split("/") for m in model_names]
    max_depth = max((len(p) for p in parts), default=1)
    labels: dict[str, str] = {}
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
        if not combos:
            continue
        scorer, metric = combos[0]
        row = {"task": task, "scorer": scorer, "metric": metric}
        for model in models:
            row[model] = value_by_key.get((model, task, scorer, metric))
        table_rows.append(row)
else:
    for task in tasks:
        combos = sorted(combos_by_task.get(task, []), key=lambda c: (c[0], c[1]))
        for scorer, metric in combos:
            row = {"task": task, "scorer": scorer, "metric": metric}
            for model in models:
                row[model] = value_by_key.get((model, task, scorer, metric))
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

    columns = [
        ("task", "Task", 14, "left"),
        ("metric", "Metric", 12, "left"),
        ("scorer", "Scorer", 12, "left"),
    ]
    model_labels = unique_model_labels(models)
    for model in models:
        columns.append((model, model_labels[model], 20, "right"))

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
col_defs = []
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
    col_defs.append((col, base, row))

model_rows = []
for model in models:
    out = {"model": model}
    for col_key, _base, source_row in col_defs:
        out[col_key] = source_row.get(model)
    model_rows.append(out)

if fmt == "json":
    print(json.dumps({"orientation": orientation, "columns": [c for c, _b, _r in col_defs], "rows": model_rows}, indent=2))
    raise SystemExit(0)

if fmt == "csv":
    fieldnames = ["model"] + [c for c, _b, _r in col_defs]
    writer = csv.DictWriter(os.sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in model_rows:
        writer.writerow(row)
    raise SystemExit(0)

model_labels = unique_model_labels(models)
columns = [("model", "Model", 42, "left")]
short_col_headers = unique_compact_labels([base for _k, base, _r in col_defs], max_len=6)
for (col_key, _base, _source_row), header in zip(col_defs, short_col_headers):
    columns.append((col_key, header, 6, "right"))

display_rows = []
for row in model_rows:
    out = {"model": model_labels.get(row.get("model", "-"), row.get("model", "-"))}
    for col_key, _base, _source_row in col_defs:
        out[col_key] = value_text(row.get(col_key))
    display_rows.append(out)

metric_mode = "primary" if primary_only else "all"
emit_string_table(
    columns,
    display_rows,
    title=f"Model Comparison (model rows, {metric_mode} metrics)",
)
PY
