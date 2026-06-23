from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path(
    "/pfs/lustrep4/scratch/project_465002183/rasmus/artifacts/evals/eee/data/WMT24___en-da"
)
DEFAULT_MODEL = "Unbabel/wmt22-comet-da"


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = output_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        path
        for path in input_root.rglob("*.jsonl")
        if ".bak" not in path.name and path.is_file()
    )
    if args.limit_files is not None:
        paths = paths[: args.limit_files]

    from dfm_evals.scorers.comet import _get_comet_runtime, score_comet_texts

    runtime = _get_comet_runtime(
        model=args.model,
        model_storage_path=None,
        local_files_only=args.local_files_only,
        device=args.device,
    )

    summaries: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] rescoring {path}", flush=True)
        summaries.append(
            rescore_file(
                path=path,
                input_root=input_root,
                instances_dir=instances_dir,
                runtime=runtime,
                score_texts=score_comet_texts,
                batch_size=args.batch_size,
                limit_samples=args.limit_samples,
            )
        )

    summaries.sort(key=lambda row: row["comet_mean"], reverse=True)
    write_json(output_dir / "summary.json", summaries)
    write_summary_csv(output_dir / "summary.csv", summaries)
    write_domain_summary(output_dir / "domain_summary.csv", summaries)
    print(f"Wrote COMET rescoring output to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore existing WMT24++ every_eval_ever JSONL exports with COMET."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.limit_files is not None and args.limit_files < 1:
        parser.error("--limit-files must be >= 1")
    if args.limit_samples is not None and args.limit_samples < 1:
        parser.error("--limit-samples must be >= 1")
    return args


def rescore_file(
    *,
    path: Path,
    input_root: Path,
    instances_dir: Path,
    runtime: Any,
    score_texts: Any,
    batch_size: int,
    limit_samples: int | None,
) -> dict[str, Any]:
    output_path = instances_dir / f"{safe_relative_name(path, input_root)}.comet.jsonl"
    rows: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if limit_samples is not None and len(rows) + len(batch) >= limit_samples:
                break
            record = json.loads(line)
            batch.append(normalize_record(record=record, path=path, line_number=line_number))
            if len(batch) >= batch_size:
                rows.extend(
                    score_batch(runtime=runtime, score_texts=score_texts, batch=batch)
                )
                batch.clear()

    if batch:
        rows.extend(score_batch(runtime=runtime, score_texts=score_texts, batch=batch))

    with output_path.open("w", encoding="utf-8") as target_file:
        for row in rows:
            target_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            target_file.write("\n")

    summary = summarize_rows(rows)
    summary.update(
        {
            "source_file": str(path),
            "instance_output": str(output_path),
            "run_key": str(path.relative_to(input_root).with_suffix("")),
            "model_id": rows[0]["model_id"] if rows else None,
            "evaluation_id": rows[0]["evaluation_id"] if rows else None,
            "n": len(rows),
        }
    )
    return summary


def normalize_record(*, record: dict[str, Any], path: Path, line_number: int) -> dict[str, Any]:
    input_payload = as_dict(record.get("input"))
    output_payload = as_dict(record.get("output"))
    evaluation = as_dict(record.get("evaluation"))

    candidate = first_text(output_payload.get("raw"))
    reference = first_text(input_payload.get("reference"))
    source = extract_source(first_text(input_payload.get("raw")))
    chrf = maybe_float(evaluation.get("score"))

    return {
        "evaluation_id": record.get("evaluation_id"),
        "model_id": record.get("model_id"),
        "sample_id": record.get("sample_id"),
        "sample_hash": record.get("sample_hash"),
        "source": source,
        "candidate": candidate,
        "reference": reference,
        "chrf3pp": chrf,
        "domain": infer_domain(str(record.get("sample_id") or "")),
        "candidate_chars": len(candidate),
        "reference_chars": len(reference),
        "source_file": str(path),
        "line_number": line_number,
    }


def score_batch(
    *, runtime: Any, score_texts: Any, batch: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    scores = score_texts(
        runtime=runtime,
        sources=[row["source"] for row in batch],
        candidates=[row["candidate"] for row in batch],
        references=[row["reference"] for row in batch],
    )
    result: list[dict[str, Any]] = []
    for row, score in zip(batch, scores, strict=True):
        result.append(
            {
                "evaluation_id": row["evaluation_id"],
                "model_id": row["model_id"],
                "sample_id": row["sample_id"],
                "sample_hash": row["sample_hash"],
                "domain": row["domain"],
                "chrf3pp": row["chrf3pp"],
                "comet": float(score),
                "candidate_chars": row["candidate_chars"],
                "reference_chars": row["reference_chars"],
                "source_file": row["source_file"],
                "line_number": row["line_number"],
            }
        )
    return result


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comet = [float(row["comet"]) for row in rows]
    chrf = [float(row["chrf3pp"]) for row in rows if row["chrf3pp"] is not None]
    domains: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        domains[str(row["domain"] or "unknown")].append(float(row["comet"]))

    return {
        "comet_mean": mean(comet),
        "comet_stderr": stderr(comet),
        "comet_min": min(comet) if comet else None,
        "comet_p25": quantile(comet, 0.25),
        "comet_median": quantile(comet, 0.50),
        "comet_p75": quantile(comet, 0.75),
        "comet_max": max(comet) if comet else None,
        "chrf3pp_mean": mean(chrf),
        "chrf3pp_stderr": stderr(chrf),
        "chrf_comet_pearson": pearson(chrf, comet) if len(chrf) == len(comet) else None,
        "verbose_outputs": sum(1 for row in rows if int(row["candidate_chars"]) > 600),
        "domains": {
            domain: {
                "n": len(scores),
                "comet_mean": mean(scores),
                "comet_stderr": stderr(scores),
            }
            for domain, scores in sorted(domains.items())
        },
    }


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    columns = [
        "run_key",
        "model_id",
        "n",
        "comet_mean",
        "comet_stderr",
        "comet_median",
        "chrf3pp_mean",
        "chrf3pp_stderr",
        "chrf_comet_pearson",
        "verbose_outputs",
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in summaries:
            writer.writerow({column: row.get(column) for column in columns})


def write_domain_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    columns = ["run_key", "model_id", "domain", "n", "comet_mean", "comet_stderr"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            for domain, values in summary["domains"].items():
                writer.writerow(
                    {
                        "run_key": summary["run_key"],
                        "model_id": summary["model_id"],
                        "domain": domain,
                        "n": values["n"],
                        "comet_mean": values["comet_mean"],
                        "comet_stderr": values["comet_stderr"],
                    }
                )


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def extract_source(raw_input: str) -> str:
    marker = "\n\nEnglish text:\n"
    if marker in raw_input:
        return raw_input.split(marker, 1)[1].strip()

    prefix = "English text:"
    if raw_input.startswith(prefix):
        text = raw_input[len(prefix) :].lstrip()
        for trailer in (
            "\n\nTranslate the above text into Danish.",
            "\n\nTranslate the above text",
        ):
            if trailer in text:
                text = text.split(trailer, 1)[0]
                break
        return text.strip()

    raise ValueError("Could not extract source text from EEE input.raw")


def infer_domain(sample_id: str) -> str:
    match = re.match(r"test-[a-z]+-([^_:]+)", sample_id)
    return match.group(1) if match else "unknown"


def safe_relative_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return "__".join(part for part in relative.with_suffix("").parts)


def first_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return first_text(value[0])
    return ""


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def maybe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stderr(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_den = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_den = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    denominator = left_den * right_den
    return numerator / denominator if denominator else None


if __name__ == "__main__":
    main()
