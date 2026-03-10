from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

SCHEMA_VERSION = "0.2.1"
INSTANCE_SCHEMA_VERSION = "instance_level_eval_0.2.1"
RELATIONSHIPS = {"first_party", "third_party", "collaborative", "other"}
ENGINE_PREFIXES = {"vllm", "sglang", "hf", "ollama", "llamacpp", "llama-cpp-python"}
API_PREFIXES = {
    "openai",
    "anthropic",
    "google",
    "grok",
    "xai",
    "mistral",
    "deepseek",
    "perplexity",
    "openrouter",
    "azure",
    "azure-ai",
    "bedrock",
}
LOWER_IS_BETTER_HINTS = (
    "error",
    "loss",
    "latency",
    "time",
    "toxicity",
    "perplexity",
    "rmse",
    "mae",
    "wer",
    "cer",
)
CORRECT_TEXT_LABELS = {"C", "CORRECT", "TRUE", "PASS", "YES"}
INCORRECT_TEXT_LABELS = {"I", "INCORRECT", "FALSE", "FAIL", "NO"}


def _compact(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _compact(v) for k, v in data.items() if v is not None}
    if isinstance(data, list):
        return [_compact(v) for v in data if v is not None]
    return data


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _string_dict(value: Mapping[str, Any] | None) -> dict[str, str] | None:
    if not value:
        return None
    return {str(k): str(v) for k, v in value.items()}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _to_unix_timestamp(value: str | float | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return str(parsed) if math.isfinite(parsed) else None

    text = value.strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return str(dt.timestamp())


def _now_unix_timestamp() -> str:
    return str(time.time())


def _sanitize_path_component(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in "._-":
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("._-")
    return result or "unknown"


def _model_label_from_ref(model_ref: str) -> str:
    ref = (model_ref or "").strip()
    if not ref:
        return "model"

    if ref.startswith("vllm/"):
        ref = ref[len("vllm/") :]
    if ref.startswith("openai/"):
        ref = ref[len("openai/") :]

    if "/" in ref:
        base_name = Path(ref).name
        parent_name = Path(ref).parent.name
        if (
            base_name in {"final", "latest", "last"}
            or base_name.startswith("checkpoint-")
            or base_name.startswith("step-")
            or base_name.startswith("epoch-")
        ):
            if parent_name and parent_name not in {".", "/"}:
                label = f"{parent_name}-{base_name}"
            else:
                label = base_name
        else:
            label = base_name
    else:
        label = ref

    cleaned = []
    for char in label:
        if char.isalnum() or char in "._-":
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("._-")
    return result or "model"


def _extract_lora_context_from_server_logs(eval_log_path: Path) -> dict[str, str] | None:
    server_dir = eval_log_path.parent / "_vllm_server"
    if not server_dir.is_dir():
        return None

    try:
        log_files = sorted(server_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return None

    lora_module_pattern = re.compile(r"LoRAModulePath\(name='([^']+)', path='([^']+)'")
    model_tag_pattern = re.compile(r"'model_tag': '([^']+)'")
    served_model_pattern = re.compile(r"'served_model_name': \['([^']+)'\]")

    for log_file in log_files:
        try:
            with log_file.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "LoRAModulePath(" not in line:
                        continue

                    module_match = lora_module_pattern.search(line)
                    if module_match is None:
                        continue

                    context: dict[str, str] = {
                        "lora_module_name": module_match.group(1).strip(),
                        "lora_adapter_path": module_match.group(2).strip(),
                        "server_log_path": str(log_file),
                    }

                    model_tag_match = model_tag_pattern.search(line)
                    if model_tag_match is not None:
                        context["lora_base_model"] = model_tag_match.group(1).strip()

                    served_model_match = served_model_pattern.search(line)
                    if served_model_match is not None:
                        context["served_model_name"] = served_model_match.group(1).strip()

                    return context
        except OSError:
            continue

    return None


def _infer_lora_model_ref(
    *,
    model_ref: str,
    sample_output_model_ref: str | None,
    lora_context: Mapping[str, str] | None,
) -> tuple[str, bool]:
    if not lora_context:
        return model_ref, False

    if not model_ref.startswith("vllm/"):
        return model_ref, False

    adapter_path = str(lora_context.get("lora_adapter_path", "")).strip()
    if not adapter_path:
        return model_ref, False

    suffix = model_ref[len("vllm/") :]
    sample_model = (sample_output_model_ref or "").strip()
    module_name = str(lora_context.get("lora_module_name", "")).strip()
    base_model = str(lora_context.get("lora_base_model", "")).strip()

    should_infer = False
    if sample_model and suffix == sample_model:
        should_infer = True
    elif module_name and base_model and module_name == base_model and suffix == module_name:
        should_infer = True

    if not should_infer:
        return model_ref, False

    adapter_label = _model_label_from_ref(adapter_path)
    if not adapter_label:
        return model_ref, False

    developer = "unknown"
    if "/" in suffix:
        developer = suffix.split("/", 1)[0]

    if developer and developer != "unknown":
        inferred = f"vllm/{developer}/{adapter_label}"
    else:
        inferred = f"vllm/{adapter_label}"

    if inferred == model_ref:
        return model_ref, False
    return inferred, True


def _merge_generation_additional_details(
    generation_config: Mapping[str, Any] | None,
    *,
    extra_details: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    details = {str(k): str(v) for k, v in (extra_details or {}).items() if v is not None}
    if not generation_config and not details:
        return None

    merged: dict[str, Any] = {}
    if generation_config:
        merged.update(dict(generation_config))

    existing_details = _as_mapping(merged.get("additional_details"))
    existing_details.update(details)

    if existing_details:
        merged["additional_details"] = existing_details

    return _compact(merged)


def _split_model_id(model_id: str) -> tuple[str, str]:
    if "/" in model_id:
        developer, model_name = model_id.split("/", 1)
        return _sanitize_path_component(developer), _sanitize_path_component(model_name)
    return "unknown", _sanitize_path_component(model_id)


def _parse_model_info(
    model_ref: str,
    *,
    fallback_vllm_version: str | None = None,
) -> dict[str, Any]:
    reference = (model_ref or "unknown_model").strip() or "unknown_model"
    parts = reference.split("/")
    prefix = parts[0].lower() if parts else "unknown"

    inference_platform: str | None = None
    inference_engine: dict[str, str] | None = None

    if prefix in ENGINE_PREFIXES:
        if len(parts) >= 3:
            developer = parts[1]
            model_name = parts[2].split("#", 1)[0].split("@", 1)[0]
            model_id = f"{developer}/{model_name}"
        elif len(parts) == 2:
            developer = parts[0]
            model_name = parts[1].split("#", 1)[0].split("@", 1)[0]
            model_id = f"{developer}/{model_name}"
        else:
            developer = "unknown"
            model_id = reference

        inference_engine = {"name": prefix}
    elif prefix in API_PREFIXES:
        developer = prefix
        model_id = reference.split("#", 1)[0]
        inference_platform = prefix
    elif len(parts) >= 2:
        developer = parts[0]
        model_name = parts[1].split("#", 1)[0]
        model_id = f"{developer}/{model_name}"
    else:
        developer = "unknown"
        model_id = reference.split("#", 1)[0]

    if inference_engine is not None and fallback_vllm_version and inference_engine["name"] == "vllm":
        inference_engine["version"] = fallback_vllm_version
    elif fallback_vllm_version and inference_engine is None:
        inference_engine = {"name": "vllm", "version": fallback_vllm_version}

    model_info: dict[str, Any] = {
        "name": reference,
        "id": model_id,
        "developer": developer,
        "inference_platform": inference_platform,
        "inference_engine": inference_engine,
    }
    return _compact(model_info)


def _infer_lower_is_better(metric_name: str) -> bool:
    lowered = metric_name.lower()
    return any(hint in lowered for hint in LOWER_IS_BETTER_HINTS)


def _build_source_metadata(
    *,
    source_name: str,
    source_organization_name: str,
    evaluator_relationship: str,
    source_organization_url: str | None,
    source_organization_logo_url: str | None,
) -> dict[str, Any]:
    relationship = evaluator_relationship.strip()
    if relationship not in RELATIONSHIPS:
        allowed = ", ".join(sorted(RELATIONSHIPS))
        raise ValueError(
            f"Invalid evaluator relationship `{evaluator_relationship}`. Expected one of: {allowed}."
        )

    return _compact(
        {
            "source_name": source_name,
            "source_type": "evaluation_run",
            "source_organization_name": source_organization_name,
            "source_organization_url": source_organization_url,
            "source_organization_logo_url": source_organization_logo_url,
            "evaluator_relationship": relationship,
        }
    )


def _write_record(
    *,
    output_dir: Path,
    benchmark_name: str,
    model_info: Mapping[str, Any],
    record: Mapping[str, Any],
) -> Path:
    model_id = str(model_info.get("id", "unknown"))
    model_dev, model_name = _split_model_id(model_id)
    benchmark_dir = _sanitize_path_component(benchmark_name)
    file_uuid = str(uuid.uuid4())

    destination_dir = output_dir / benchmark_dir / model_dev / model_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{file_uuid}.json"

    with destination.open("w", encoding="utf-8") as f:
        json.dump(_compact(dict(record)), f, ensure_ascii=False, indent=2)
        f.write("\n")

    return destination


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_compact(dict(row)), ensure_ascii=False))
            f.write("\n")


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_to_text(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _parse_message_content(content: Any) -> tuple[str | None, str | None]:
    if content is None:
        return None, None
    if isinstance(content, str):
        return content, None
    if not isinstance(content, list):
        text = _to_text(content)
        return text if text else None, None

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for part in content:
        part_map = _as_mapping(part)
        part_type = str(part_map.get("type") or getattr(part, "type", "")).lower()
        if part_type == "reasoning":
            reasoning_text = _to_text(
                part_map.get("reasoning")
                or part_map.get("summary")
                or part_map.get("text")
                or part_map.get("content")
            )
            if reasoning_text:
                reasoning_parts.append(reasoning_text)
            continue

        text = _to_text(part_map.get("text") or part_map.get("content") or part)
        if text:
            text_parts.append(text)

    text_value = "\n".join(text_parts) if text_parts else None
    reasoning_value = "\n".join(reasoning_parts) if reasoning_parts else None
    return text_value, reasoning_value


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return []

    serialized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        arguments = getattr(tool_call, "arguments", None)
        arguments_map: dict[str, str] | None = None
        if isinstance(arguments, Mapping):
            arguments_map = {}
            for key, value in arguments.items():
                if isinstance(value, str):
                    arguments_map[str(key)] = value
                else:
                    arguments_map[str(key)] = json.dumps(
                        value, ensure_ascii=False, sort_keys=True, default=str
                    )

        serialized.append(
            _compact(
                {
                    "id": str(getattr(tool_call, "id", "") or ""),
                    "name": str(
                        getattr(tool_call, "function", None)
                        or getattr(tool_call, "name", "")
                        or ""
                    ),
                    "arguments": arguments_map,
                }
            )
        )
    return [item for item in serialized if item.get("id") and item.get("name")]


def _message_tool_call_id(message: Any) -> list[str] | None:
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id is None:
        return None
    if isinstance(tool_call_id, str):
        return [tool_call_id]
    if isinstance(tool_call_id, list):
        values = [str(item) for item in tool_call_id if item is not None]
        return values or None
    return [str(tool_call_id)]


def _extract_response(sample: Any) -> tuple[str, str | None]:
    output = getattr(sample, "output", None)
    completion = _to_text(getattr(output, "completion", None)).strip()
    if completion:
        return completion, None

    choices = getattr(output, "choices", None)
    if isinstance(choices, list) and len(choices) > 0:
        choice_message = getattr(choices[0], "message", None)
        if choice_message is not None:
            content, reasoning = _parse_message_content(
                getattr(choice_message, "content", None)
            )
            if content:
                return content, reasoning
            if reasoning:
                return "", reasoning

    sample_scores = _as_mapping(getattr(sample, "scores", None))
    for score in sample_scores.values():
        answer_text = _to_text(getattr(score, "answer", None)).strip()
        if answer_text:
            return answer_text, None
        explanation = _to_text(getattr(score, "explanation", None)).strip()
        if explanation:
            return explanation, None

    return "", None


def _extract_references(target: Any) -> list[str]:
    if target is None:
        return []
    if isinstance(target, str):
        return [target]
    if isinstance(target, (list, tuple, set)):
        return [str(value) for value in target]
    return [str(target)]


def _extract_score(sample: Any, extracted_answer: str, references: list[str]) -> tuple[float, bool]:
    sample_scores = _as_mapping(getattr(sample, "scores", None))

    numeric_score: float | None = None
    bool_correct: bool | None = None

    for score in sample_scores.values():
        raw_value = getattr(score, "value", None)

        if isinstance(raw_value, bool):
            bool_correct = raw_value
            numeric_score = 1.0 if raw_value else 0.0
            break

        if isinstance(raw_value, str):
            upper = raw_value.strip().upper()
            if upper in CORRECT_TEXT_LABELS:
                bool_correct = True
                numeric_score = 1.0
                break
            if upper in INCORRECT_TEXT_LABELS:
                bool_correct = False
                numeric_score = 0.0
                break

        as_number = _maybe_float(raw_value)
        if as_number is not None and numeric_score is None:
            numeric_score = as_number
            if 0.0 <= as_number <= 1.0:
                bool_correct = as_number >= 0.5

    if bool_correct is None:
        bool_correct = extracted_answer in references if references else False

    if numeric_score is None:
        numeric_score = 1.0 if bool_correct else 0.0

    return numeric_score, bool_correct


def _extract_token_usage(sample: Any) -> dict[str, Any] | None:
    output = getattr(sample, "output", None)
    usage = getattr(output, "usage", None)
    if usage is None:
        return None

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if input_tokens is None or output_tokens is None or total_tokens is None:
        return None

    return _compact(
        {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
            "input_tokens_cache_write": getattr(usage, "input_tokens_cache_write", None),
            "input_tokens_cache_read": getattr(usage, "input_tokens_cache_read", None),
            "reasoning_tokens": getattr(usage, "reasoning_tokens", None),
        }
    )


def _extract_performance(sample: Any) -> dict[str, Any] | None:
    total_time = _maybe_float(getattr(sample, "total_time", None))
    working_time = _maybe_float(getattr(sample, "working_time", None))
    if total_time is None and working_time is None:
        return None

    return _compact(
        {
            "latency_ms": max(0.0, (total_time or 0.0) * 1000.0),
            "generation_time_ms": (
                max(0.0, working_time * 1000.0) if working_time is not None else None
            ),
        }
    )


def _extract_error(sample: Any) -> str | None:
    error = getattr(sample, "error", None)
    if error is None:
        return None
    message = _to_text(getattr(error, "message", None)).strip()
    traceback = _to_text(getattr(error, "traceback", None)).strip()
    if message and traceback:
        return f"{message}\n{traceback}"
    return message or traceback or None


def _extract_metadata(sample: Any) -> dict[str, str] | None:
    metadata: dict[str, str] = {}
    output = getattr(sample, "output", None)
    stop_reason: Any = None
    if output is not None:
        try:
            stop_reason = getattr(output, "stop_reason", None)
        except IndexError:
            # Some Inspect logs can have an output object with zero choices.
            # Treat stop reason as unknown instead of failing export.
            stop_reason = None
    if stop_reason is not None:
        metadata["stop_reason"] = str(stop_reason)
    epoch = getattr(sample, "epoch", None)
    if epoch is not None:
        metadata["epoch"] = str(epoch)
    return metadata or None


def _build_inspect_instance_rows(
    *,
    evaluation_id: str,
    model_id: str,
    evaluation_name: str,
    samples: list[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        sample_id = str(getattr(sample, "id", index))
        input_raw = _to_text(getattr(sample, "input", ""))
        references = _extract_references(getattr(sample, "target", None))
        choices_raw = getattr(sample, "choices", None)
        choices = (
            [str(choice) for choice in choices_raw]
            if isinstance(choices_raw, list)
            else None
        )

        processed_messages: list[dict[str, Any]] = []
        raw_messages = getattr(sample, "messages", None)
        if isinstance(raw_messages, list):
            for turn_idx, message in enumerate(raw_messages):
                content, reasoning = _parse_message_content(
                    getattr(message, "content", None)
                )
                processed_messages.append(
                    _compact(
                        {
                            "turn_idx": turn_idx,
                            "role": str(getattr(message, "role", "unknown")),
                            "content": content,
                            "reasoning_trace": reasoning,
                            "tool_calls": _message_tool_calls(message) or None,
                            "tool_call_id": _message_tool_call_id(message),
                        }
                    )
                )

        has_tool_activity = any(
            message.get("role") == "tool" or (message.get("tool_calls") is not None)
            for message in processed_messages
        )
        assistant_turns = sum(
            1 for message in processed_messages if message.get("role") == "assistant"
        )
        if has_tool_activity:
            interaction_type = "agentic"
        elif assistant_turns > 1:
            interaction_type = "multi_turn"
        else:
            interaction_type = "single_turn"

        extracted_answer, reasoning_trace = _extract_response(sample)
        score, is_correct = _extract_score(sample, extracted_answer, references)

        if interaction_type == "single_turn":
            output = _compact(
                {
                    "raw": [extracted_answer],
                    "reasoning_trace": [reasoning_trace] if reasoning_trace else None,
                }
            )
            messages = None
            answer_source = "output.raw[0]"
            answer_turn_idx = 0
        else:
            output = None
            messages = processed_messages
            answer_turn_idx = 0
            answer_source = "messages[0].content"
            for message in reversed(processed_messages):
                role = message.get("role")
                content = _to_text(message.get("content", "")).strip()
                if role == "assistant" and content:
                    answer_turn_idx = int(message.get("turn_idx", 0))
                    answer_source = f"messages[{answer_turn_idx}].content"
                    break

        rows.append(
            _compact(
                {
                    "schema_version": INSTANCE_SCHEMA_VERSION,
                    "evaluation_id": evaluation_id,
                    "model_id": model_id,
                    "evaluation_name": evaluation_name,
                    "sample_id": sample_id,
                    "sample_hash": _sha256_text(input_raw + "".join(references)),
                    "interaction_type": interaction_type,
                    "input": {
                        "raw": input_raw,
                        "reference": references,
                        "choices": choices,
                    },
                    "output": output,
                    "messages": messages,
                    "answer_attribution": [
                        {
                            "turn_idx": answer_turn_idx,
                            "source": answer_source,
                            "extracted_value": extracted_answer,
                            "extraction_method": "identity",
                            "is_terminal": True,
                        }
                    ],
                    "evaluation": {
                        "score": score,
                        "is_correct": is_correct,
                        "num_turns": len(processed_messages) if processed_messages else 1,
                        "tool_calls_count": sum(
                            len(message.get("tool_calls", []))
                            for message in processed_messages
                        ),
                    },
                    "token_usage": _extract_token_usage(sample),
                    "performance": _extract_performance(sample),
                    "error": _extract_error(sample),
                    "metadata": _extract_metadata(sample),
                }
            )
        )

    return rows


def _extract_inspect_source_data(dataset: Any, task_name: str) -> tuple[str, dict[str, Any]]:
    dataset_map = _as_mapping(dataset)
    dataset_name = str(dataset_map.get("name") or task_name or "unknown_dataset")
    location = dataset_map.get("location")
    sample_count = dataset_map.get("samples")
    sample_ids = dataset_map.get("sample_ids")
    shuffled = dataset_map.get("shuffled")

    if isinstance(location, str) and location.startswith(("http://", "https://")):
        source_data: dict[str, Any] = {
            "dataset_name": dataset_name,
            "source_type": "url",
            "url": [location],
        }
    elif isinstance(location, str) and "/" in location and not Path(location).exists():
        source_data = {
            "dataset_name": dataset_name,
            "source_type": "hf_dataset",
            "hf_repo": location,
            "samples_number": int(sample_count) if isinstance(sample_count, int) else None,
            "sample_ids": (
                [str(sample_id) for sample_id in sample_ids]
                if isinstance(sample_ids, list)
                else None
            ),
            "additional_details": (
                {"shuffled": str(shuffled)} if shuffled is not None else None
            ),
        }
    else:
        details: dict[str, str] = {}
        if location is not None:
            details["location"] = str(location)
        if sample_count is not None:
            details["samples_number"] = str(sample_count)
        if sample_ids is not None:
            details["sample_ids"] = json.dumps(sample_ids, ensure_ascii=False, default=str)
        if shuffled is not None:
            details["shuffled"] = str(shuffled)

        source_data = {
            "dataset_name": dataset_name,
            "source_type": "other",
            "additional_details": details or None,
        }

    return dataset_name, _compact(source_data)


def _extract_inspect_inference_base_url(eval_spec: Any) -> str | None:
    model_base_url = getattr(eval_spec, "model_base_url", None)
    if isinstance(model_base_url, str) and model_base_url.strip():
        return model_base_url.strip()

    model_args = _as_mapping(getattr(eval_spec, "model_args", None))
    for key in ("base_url", "model_base_url", "api_base", "endpoint", "openai_base_url"):
        value = model_args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    generate_cfg = _as_mapping(getattr(eval_spec, "model_generate_config", None))
    for key in ("base_url", "api_base", "openai_base_url"):
        value = generate_cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _extract_inspect_generation_config(
    eval_spec: Any,
    *,
    inference_base_url: str | None = None,
    inference_provider_name: str | None = None,
) -> dict[str, Any] | None:
    generate_cfg = _as_mapping(getattr(eval_spec, "model_generate_config", None))
    task_args = _as_mapping(getattr(eval_spec, "task_args", None))
    limits_cfg = _as_mapping(getattr(eval_spec, "config", None))

    generation_args: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_k", "max_tokens"):
        if generate_cfg.get(key) is not None:
            generation_args[key] = generate_cfg[key]

    reasoning_effort = generate_cfg.get("reasoning_effort")
    if isinstance(reasoning_effort, str):
        generation_args["reasoning"] = reasoning_effort.lower() != "none"

    max_attempts = task_args.get("max_attempts", generate_cfg.get("max_retries"))
    if isinstance(max_attempts, int):
        generation_args["max_attempts"] = max_attempts

    sandbox = task_args.get("sandbox")
    if sandbox is not None:
        if isinstance(sandbox, list):
            sandbox_list = list(sandbox)
        else:
            sandbox_list = [sandbox]
        sandbox_type = str(sandbox_list[0]) if len(sandbox_list) > 0 else None
        sandbox_cfg = str(sandbox_list[1]) if len(sandbox_list) > 1 else None
        if sandbox_type is not None or sandbox_cfg is not None:
            generation_args["sandbox"] = _compact({"type": sandbox_type, "config": sandbox_cfg})

    eval_limits = _compact(
        {
            "time_limit": limits_cfg.get("time_limit"),
            "message_limit": limits_cfg.get("message_limit"),
            "token_limit": limits_cfg.get("token_limit"),
        }
    )
    if eval_limits:
        generation_args["eval_limits"] = eval_limits

    additional_details: dict[str, str] = {}
    if generate_cfg:
        additional_details["model_generate_config"] = json.dumps(
            generate_cfg, ensure_ascii=False, sort_keys=True, default=str
        )
    if task_args:
        additional_details["task_args"] = json.dumps(
            task_args, ensure_ascii=False, sort_keys=True, default=str
        )
    if limits_cfg:
        additional_details["eval_config"] = json.dumps(
            limits_cfg, ensure_ascii=False, sort_keys=True, default=str
        )
    if inference_base_url:
        additional_details["inference_base_url"] = inference_base_url
        parsed_base = urlparse(inference_base_url)
        if parsed_base.hostname:
            additional_details["inference_host"] = parsed_base.hostname
    if inference_provider_name:
        additional_details["inference_provider_name"] = inference_provider_name

    if not generation_args and not additional_details:
        return None

    return _compact(
        {
            "generation_args": generation_args or None,
            "additional_details": additional_details or None,
        }
    )


def _extract_inspect_results(
    *,
    eval_log: Any,
    task_name: str,
    source_data: Mapping[str, Any],
    evaluation_timestamp: str | None,
    generation_config: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    results = getattr(eval_log, "results", None)
    scorers = list(getattr(results, "scores", []) or [])
    sample_count = len(getattr(eval_log, "samples", []) or [])

    evaluation_results: list[dict[str, Any]] = []
    for scorer in scorers:
        scorer_name = str(
            getattr(scorer, "name", None) or getattr(scorer, "scorer", "unknown_scorer")
        )
        params = _as_mapping(getattr(scorer, "params", None))
        metrics = _as_mapping(getattr(scorer, "metrics", None))

        stderr_value: float | None = None
        stddev_value: float | None = None
        for metric in metrics.values():
            metric_name = str(getattr(metric, "name", "")).lower()
            metric_value = _maybe_float(getattr(metric, "value", None))
            if metric_name == "stderr":
                stderr_value = metric_value
            elif metric_name in {"std", "stddev", "standard_deviation"}:
                stddev_value = metric_value

        grader_model = params.get("grader_model")
        llm_scoring: dict[str, Any] | None = None
        if grader_model:
            llm_scoring = {
                "judges": [{"model_info": _parse_model_info(str(grader_model))}],
                "input_prompt": str(params.get("grader_template", "")),
            }

        for metric_key, metric in metrics.items():
            metric_name = str(getattr(metric, "name", metric_key))
            lowered = metric_name.lower()
            if lowered in {"stderr", "std", "stddev", "standard_deviation"}:
                continue

            metric_value = _maybe_float(getattr(metric, "value", None))
            if metric_value is None:
                continue

            details = {
                "scorer": scorer_name,
                "metric": metric_name,
                "task": task_name,
                "scorer_params": json.dumps(params, ensure_ascii=False, sort_keys=True, default=str),
            }
            score_details: dict[str, Any] = {
                "score": metric_value,
                "details": _string_dict(details),
            }

            uncertainty = _compact(
                {
                    "standard_error": (
                        {"value": stderr_value} if stderr_value is not None else None
                    ),
                    "standard_deviation": stddev_value,
                    "num_samples": sample_count if sample_count > 0 else None,
                }
            )
            if uncertainty:
                score_details["uncertainty"] = uncertainty

            metric_config = {
                "evaluation_description": f"{scorer_name}:{metric_name}",
                "lower_is_better": _infer_lower_is_better(metric_name),
                "score_type": "continuous",
                "min_score": min(0.0, metric_value),
                "max_score": max(1.0, metric_value),
                "llm_scoring": llm_scoring,
            }

            result = _compact(
                {
                    "evaluation_name": f"{task_name}/{scorer_name}/{metric_name}",
                    "source_data": dict(source_data),
                    "evaluation_timestamp": evaluation_timestamp,
                    "metric_config": metric_config,
                    "score_details": score_details,
                    "generation_config": dict(generation_config)
                    if generation_config is not None
                    else None,
                }
            )
            evaluation_results.append(result)

    return evaluation_results


def export_inspect_logs(
    *,
    log_path: str | Path,
    output_dir: str | Path,
    source_organization_name: str = "unknown",
    evaluator_relationship: str = "third_party",
    source_organization_url: str | None = None,
    source_organization_logo_url: str | None = None,
    eval_library_name: str = "inspect_ai",
    eval_library_version: str | None = None,
    inference_base_url: str | None = None,
    inference_provider_name: str | None = None,
) -> list[Path]:
    try:
        from inspect_ai.log import list_eval_logs, read_eval_log
    except ImportError as exc:
        raise RuntimeError("inspect_ai is required for Inspect EEE export.") from exc

    resolved_input = Path(log_path)
    destination_root = Path(output_dir)
    source_metadata = _build_source_metadata(
        source_name="inspect_ai",
        source_organization_name=source_organization_name,
        evaluator_relationship=evaluator_relationship,
        source_organization_url=source_organization_url,
        source_organization_logo_url=source_organization_logo_url,
    )

    candidates: list[Path] = []
    if resolved_input.is_file():
        candidates = [resolved_input]
    elif resolved_input.is_dir():
        try:
            log_infos = list_eval_logs(resolved_input.absolute().as_posix())
        except Exception:
            log_infos = []

        for info in log_infos:
            log_name = getattr(info, "name", "")
            parsed = urlparse(log_name)
            if parsed.scheme == "file":
                candidates.append(Path(unquote(parsed.path)))
            elif log_name:
                candidates.append(Path(log_name))

        if not candidates:
            candidates.extend(sorted(resolved_input.rglob("*.eval")))
            candidates.extend(sorted(resolved_input.rglob("*.json")))
    else:
        raise FileNotFoundError(f"Inspect log path does not exist: {resolved_input}")

    unique_candidates = list(dict.fromkeys(candidates))
    written: list[Path] = []
    lora_context_cache: dict[Path, dict[str, str] | None] = {}
    for candidate in unique_candidates:
        try:
            eval_log = read_eval_log(str(candidate), header_only=False)
        except Exception:
            continue

        eval_spec = getattr(eval_log, "eval", None)
        if eval_spec is None:
            continue

        task_full_name = str(getattr(eval_spec, "task", "inspect_eval"))
        task_name = task_full_name.split("/")[-1] if "/" in task_full_name else task_full_name

        dataset_name, source_data = _extract_inspect_source_data(
            getattr(eval_spec, "dataset", None), task_name
        )
        benchmark_name = dataset_name or task_name

        eval_started_at = getattr(getattr(eval_log, "stats", None), "started_at", None)
        eval_created_at = getattr(eval_spec, "created", None)
        evaluation_timestamp = _to_unix_timestamp(eval_started_at) or _to_unix_timestamp(
            eval_created_at
        )
        retrieved_timestamp = _now_unix_timestamp()

        model_ref = str(getattr(eval_spec, "model", "unknown_model")).strip() or "unknown_model"
        configured_model_ref = model_ref
        samples = list(getattr(eval_log, "samples", []) or [])
        detailed_model_ref: str | None = None
        if samples:
            first_sample_output = getattr(samples[0], "output", None)
            detailed_model_value = getattr(first_sample_output, "model", None)
            if isinstance(detailed_model_value, str) and detailed_model_value.strip():
                detailed_model_ref = detailed_model_value.strip()
                # Keep explicitly qualified eval.model (e.g., vllm/<adapter>) as
                # authoritative. sample.output.model can reflect the base model.
                if model_ref == "unknown_model" or "/" not in model_ref:
                    model_ref = detailed_model_ref

        run_dir = candidate.parent.resolve()
        if run_dir not in lora_context_cache:
            lora_context_cache[run_dir] = _extract_lora_context_from_server_logs(candidate)
        lora_context = lora_context_cache[run_dir]
        model_ref, lora_model_ref_inferred = _infer_lora_model_ref(
            model_ref=model_ref,
            sample_output_model_ref=detailed_model_ref,
            lora_context=lora_context,
        )

        packages = _as_mapping(getattr(eval_spec, "packages", None))
        inspect_version = str(packages.get("inspect_ai", "unknown"))
        library_version = eval_library_version or inspect_version

        model_info = _parse_model_info(model_ref)
        resolved_inference_base_url = (
            inference_base_url or _extract_inspect_inference_base_url(eval_spec)
        )
        generation_config = _extract_inspect_generation_config(
            eval_spec,
            inference_base_url=resolved_inference_base_url,
            inference_provider_name=inference_provider_name,
        )
        generation_extra_details: dict[str, Any] = {}
        if configured_model_ref:
            generation_extra_details["configured_model_ref"] = configured_model_ref
        if detailed_model_ref:
            generation_extra_details["sample_output_model_ref"] = detailed_model_ref
        if lora_context:
            generation_extra_details.update(
                {
                    "lora_module_name": lora_context.get("lora_module_name"),
                    "lora_base_model": lora_context.get("lora_base_model"),
                    "lora_adapter_path": lora_context.get("lora_adapter_path"),
                    "lora_server_log_path": lora_context.get("server_log_path"),
                }
            )
        if lora_model_ref_inferred:
            generation_extra_details["model_ref_inferred_from_lora"] = "true"
        generation_config = _merge_generation_additional_details(
            generation_config,
            extra_details=generation_extra_details,
        )
        evaluation_results = _extract_inspect_results(
            eval_log=eval_log,
            task_name=task_name,
            source_data=source_data,
            evaluation_timestamp=evaluation_timestamp,
            generation_config=generation_config,
        )

        if len(evaluation_results) == 0:
            continue

        effective_timestamp = evaluation_timestamp or retrieved_timestamp
        evaluation_id = f"{_sanitize_path_component(benchmark_name)}/{model_info['id']}/{effective_timestamp}"

        record = _compact(
            {
                "schema_version": SCHEMA_VERSION,
                "evaluation_id": evaluation_id,
                "evaluation_timestamp": evaluation_timestamp,
                "retrieved_timestamp": retrieved_timestamp,
                "source_metadata": source_metadata,
                "eval_library": {
                    "name": eval_library_name,
                    "version": library_version,
                },
                "model_info": model_info,
                "evaluation_results": evaluation_results,
            }
        )

        destination = _write_record(
            output_dir=destination_root,
            benchmark_name=benchmark_name,
            model_info=model_info,
            record=record,
        )

        if samples:
            instance_rows = _build_inspect_instance_rows(
                evaluation_id=evaluation_id,
                model_id=str(model_info["id"]),
                evaluation_name=benchmark_name,
                samples=samples,
            )
            if instance_rows:
                instance_path = destination.with_suffix(".jsonl")
                _write_jsonl(instance_path, instance_rows)
                record["detailed_evaluation_results"] = {
                    "format": "jsonl",
                    "file_path": str(instance_path),
                    "hash_algorithm": "sha256",
                    "checksum": _sha256_file(instance_path),
                    "total_rows": len(instance_rows),
                }
                with destination.open("w", encoding="utf-8") as f:
                    json.dump(_compact(record), f, ensure_ascii=False, indent=2)
                    f.write("\n")

        written.append(destination)

    if len(written) == 0:
        raise ValueError(
            f"No Inspect evaluation logs were converted from input path: {resolved_input}"
        )

    return written


def _extract_euroeval_generation_config(
    entry: Mapping[str, Any],
    *,
    inference_base_url: str | None = None,
    inference_provider_name: str | None = None,
) -> dict[str, Any] | None:
    details: dict[str, str] = {}
    detail_fields = (
        "task",
        "few_shot",
        "generative",
        "generative_type",
        "validation_split",
        "merge",
        "num_model_parameters",
        "max_sequence_length",
        "vocabulary_size",
        "transformers_version",
        "torch_version",
        "vllm_version",
        "xgrammar_version",
    )
    for field in detail_fields:
        value = entry.get(field)
        if value is not None:
            details[field] = str(value)

    languages = entry.get("languages")
    if isinstance(languages, list):
        details["languages"] = ",".join(str(language) for language in languages)
    if inference_base_url:
        details["inference_base_url"] = inference_base_url
        parsed_base = urlparse(inference_base_url)
        if parsed_base.hostname:
            details["inference_host"] = parsed_base.hostname
    if inference_provider_name:
        details["inference_provider_name"] = inference_provider_name

    if not details:
        return None
    return {"additional_details": details}


def _extract_euroeval_results(
    *,
    entry: Mapping[str, Any],
    source_data: Mapping[str, Any],
    generation_config: Mapping[str, Any] | None,
    dataset_name: str,
    task_name: str,
) -> list[dict[str, Any]]:
    results = _as_mapping(entry.get("results"))
    totals = _as_mapping(results.get("total"))
    raw_scores = results.get("raw")
    sample_count = len(raw_scores) if isinstance(raw_scores, list) else None
    dataset_label = str(dataset_name or task_name or "unknown_dataset")
    task_group = str(task_name or dataset_name or "unknown_task")

    evaluation_results: list[dict[str, Any]] = []
    for key, value in sorted(totals.items()):
        if not key.startswith("test_") or key.endswith("_se"):
            continue

        metric_name = key[len("test_") :]
        score = _maybe_float(value)
        if score is None:
            continue

        ci_radius = _maybe_float(totals.get(f"test_{metric_name}_se"))

        uncertainty: dict[str, Any] = {}
        if ci_radius is not None:
            lower_ci = score - ci_radius
            upper_ci = score + ci_radius
            uncertainty["confidence_interval"] = {
                "lower": lower_ci,
                "upper": upper_ci,
                "confidence_level": 0.95,
                "method": "normal_approx",
            }
            uncertainty["standard_error"] = {
                "value": ci_radius / 1.96,
                "method": "derived_from_95ci_radius",
            }
        if sample_count is not None:
            uncertainty["num_samples"] = sample_count

        score_details: dict[str, Any] = {
            "score": score,
            "details": _string_dict(
                {
                    "task": dataset_label,
                    "scorer": task_group,
                    "metric": metric_name,
                }
            ),
            "uncertainty": uncertainty or None,
        }

        evaluation_results.append(
            _compact(
                {
                    "evaluation_name": f"{dataset_label}/{task_group}/{metric_name}",
                    "source_data": dict(source_data),
                    "metric_config": {
                        "evaluation_description": (
                            f"{dataset_label}:{task_group}:{metric_name}"
                        ),
                        "lower_is_better": _infer_lower_is_better(metric_name),
                        "score_type": "continuous",
                        "min_score": min(0.0, score, score - ci_radius)
                        if ci_radius is not None
                        else min(0.0, score),
                        "max_score": max(1.0, score, score + ci_radius)
                        if ci_radius is not None
                        else max(1.0, score),
                    },
                    "score_details": score_details,
                    "generation_config": (
                        dict(generation_config) if generation_config is not None else None
                    ),
                }
            )
        )

    return evaluation_results


def export_euroeval_results(
    *,
    results_file: str | Path,
    output_dir: str | Path,
    source_organization_name: str = "unknown",
    evaluator_relationship: str = "third_party",
    source_organization_url: str | None = None,
    source_organization_logo_url: str | None = None,
    eval_library_name: str = "euroeval",
    eval_library_version: str | None = None,
    inference_base_url: str | None = None,
    inference_provider_name: str | None = None,
) -> list[Path]:
    results_path = Path(results_file)
    if not results_path.is_file():
        raise FileNotFoundError(f"EuroEval results file does not exist: {results_path}")

    destination_root = Path(output_dir)
    source_metadata = _build_source_metadata(
        source_name="euroeval",
        source_organization_name=source_organization_name,
        evaluator_relationship=evaluator_relationship,
        source_organization_url=source_organization_url,
        source_organization_logo_url=source_organization_logo_url,
    )

    entries: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in EuroEval results file {results_path} at line {line_number}: {exc}"
                ) from exc
            if isinstance(parsed, Mapping):
                entries.append(dict(parsed))

    written: list[Path] = []
    for entry in entries:
        dataset_name = str(entry.get("dataset") or entry.get("task") or "unknown_dataset")
        task_name = str(entry.get("task") or dataset_name)

        source_details: dict[str, str] = {"task": task_name}
        source_details["dataset"] = dataset_name
        languages = entry.get("languages")
        if isinstance(languages, list):
            source_details["languages"] = ",".join(str(language) for language in languages)

        source_data: dict[str, Any] = _compact(
            {
                "dataset_name": dataset_name,
                "source_type": "other",
                "additional_details": source_details,
            }
        )

        model_ref = str(entry.get("model") or "unknown_model")
        vllm_version = (
            str(entry.get("vllm_version"))
            if entry.get("vllm_version") is not None
            else None
        )
        model_info = _parse_model_info(model_ref, fallback_vllm_version=vllm_version)

        generation_config = _extract_euroeval_generation_config(
            entry,
            inference_base_url=inference_base_url,
            inference_provider_name=inference_provider_name,
        )
        evaluation_results = _extract_euroeval_results(
            entry=entry,
            source_data=source_data,
            generation_config=generation_config,
            dataset_name=dataset_name,
            task_name=task_name,
        )
        if len(evaluation_results) == 0:
            continue

        retrieved_timestamp = _now_unix_timestamp()
        library_version = (
            str(entry.get("euroeval_version"))
            if entry.get("euroeval_version") is not None
            else None
        )
        if library_version is None:
            library_version = eval_library_version or "unknown"

        evaluation_id = (
            f"{_sanitize_path_component(dataset_name)}/{model_info['id']}/{retrieved_timestamp}"
        )
        record = _compact(
            {
                "schema_version": SCHEMA_VERSION,
                "evaluation_id": evaluation_id,
                "retrieved_timestamp": retrieved_timestamp,
                "source_metadata": source_metadata,
                "eval_library": {
                    "name": eval_library_name,
                    "version": library_version,
                },
                "model_info": model_info,
                "evaluation_results": evaluation_results,
            }
        )

        destination = _write_record(
            output_dir=destination_root,
            benchmark_name=dataset_name,
            model_info=model_info,
            record=record,
        )
        written.append(destination)

    if len(written) == 0:
        raise ValueError(
            f"No EuroEval records were converted from results file: {results_path}"
        )

    return written


def _tournament_benchmark_name(project_id: str | None) -> str:
    value = (project_id or "").strip()
    return value if value else "tournament"


def _extract_tournament_generation_config(
    config: Any,
    status: Any,
) -> dict[str, Any] | None:
    contestant_generate_config = _as_mapping(
        getattr(config, "contestant_generate_config", None)
    )
    judge_generate_config = _as_mapping(getattr(config, "judge_generate_config", None))

    generation_args: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_k", "max_tokens"):
        value = contestant_generate_config.get(key)
        if value is not None:
            generation_args[key] = value

    additional_details = _compact(
        {
            "project_id": getattr(status, "project_id", None),
            "run_status": getattr(status, "run_status", None),
            "converged": str(getattr(status, "converged", False)).lower(),
            "stop_reasons": (
                json.dumps(
                    getattr(status, "stop_reasons", []),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if getattr(status, "stop_reasons", None)
                else None
            ),
            "total_models": str(getattr(status, "total_models", "")),
            "total_prompts": str(getattr(status, "total_prompts", "")),
            "total_matches": str(getattr(status, "total_matches", "")),
            "rated_matches": str(getattr(status, "rated_matches", "")),
            "judge_model": getattr(config, "judge_model", None),
            "run_dir": (
                str(getattr(config, "run_dir"))
                if getattr(config, "run_dir", None) is not None
                else None
            ),
            "state_dir": (
                str(getattr(config, "state_dir"))
                if getattr(config, "state_dir", None) is not None
                else None
            ),
            "contestant_generate_config": (
                json.dumps(
                    contestant_generate_config,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                if contestant_generate_config
                else None
            ),
            "judge_generate_config": (
                json.dumps(
                    judge_generate_config,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                if judge_generate_config
                else None
            ),
        }
    )

    if not generation_args and not additional_details:
        return None

    return _compact(
        {
            "generation_args": generation_args or None,
            "additional_details": additional_details or None,
        }
    )


def _extract_tournament_source_data(
    *,
    config: Any,
    status: Any,
    benchmark_name: str,
) -> dict[str, Any]:
    return _compact(
        {
            "dataset_name": benchmark_name,
            "source_type": "other",
            "additional_details": {
                "project_id": getattr(status, "project_id", None),
                "run_status": getattr(status, "run_status", None),
                "converged": str(getattr(status, "converged", False)).lower(),
                "stop_reasons": (
                    json.dumps(
                        getattr(status, "stop_reasons", []),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if getattr(status, "stop_reasons", None)
                    else None
                ),
                "total_models": str(getattr(status, "total_models", "")),
                "total_prompts": str(getattr(status, "total_prompts", "")),
                "response_count": str(getattr(status, "response_count", "")),
                "expected_responses": str(getattr(status, "expected_responses", "")),
                "missing_responses": str(getattr(status, "missing_responses", "")),
                "total_matches": str(getattr(status, "total_matches", "")),
                "rated_matches": str(getattr(status, "rated_matches", "")),
                "judge_model": getattr(config, "judge_model", None),
            },
        }
    )


def _build_tournament_llm_scoring(config: Any) -> dict[str, Any] | None:
    judge_model = getattr(config, "judge_model", None)
    if not isinstance(judge_model, str) or not judge_model.strip():
        return None

    return _compact(
        {
            "judges": [{"model_info": _parse_model_info(judge_model)}],
            "input_prompt": getattr(config, "judge_prompt_template", None),
        }
    )


def _extract_tournament_results(
    *,
    benchmark_name: str,
    source_data: Mapping[str, Any],
    generation_config: Mapping[str, Any] | None,
    llm_scoring: Mapping[str, Any] | None,
    standing: Any,
    rank: int,
) -> list[dict[str, Any]]:
    metrics: list[tuple[str, float]] = [
        ("rank", float(rank)),
        ("conservative", float(getattr(standing, "conservative"))),
        ("elo_like", float(getattr(standing, "elo_like"))),
        ("mu", float(getattr(standing, "mu"))),
        ("sigma", float(getattr(standing, "sigma"))),
        ("games", float(getattr(standing, "games"))),
        ("wins", float(getattr(standing, "wins"))),
        ("losses", float(getattr(standing, "losses"))),
        ("ties", float(getattr(standing, "ties"))),
    ]

    games = int(getattr(standing, "games"))
    if games > 0:
        metrics.extend(
            [
                ("win_rate", float(getattr(standing, "wins")) / games),
                ("tie_rate", float(getattr(standing, "ties")) / games),
            ]
        )

    lower_is_better_metrics = {"rank", "sigma", "losses"}
    count_metrics = {"games", "wins", "losses", "ties"}
    shared_details = {
        "task": benchmark_name,
        "scorer": "tournament",
        "rank": str(rank),
        "model_name": getattr(standing, "model_name", None),
        "games": str(getattr(standing, "games")),
        "wins": str(getattr(standing, "wins")),
        "losses": str(getattr(standing, "losses")),
        "ties": str(getattr(standing, "ties")),
    }

    results: list[dict[str, Any]] = []
    for metric_name, score in metrics:
        score_details = {
            "score": score,
            "details": _string_dict({**shared_details, "metric": metric_name}),
            "uncertainty": (
                {"num_samples": games}
                if games > 0 and metric_name not in count_metrics
                else None
            ),
        }
        results.append(
            _compact(
                {
                    "evaluation_name": f"{benchmark_name}/tournament/{metric_name}",
                    "source_data": dict(source_data),
                    "metric_config": {
                        "evaluation_description": f"tournament:{metric_name}",
                        "lower_is_better": metric_name in lower_is_better_metrics,
                        "score_type": (
                            "count" if metric_name in count_metrics else "continuous"
                        ),
                        "min_score": min(0.0, score),
                        "max_score": max(1.0, score),
                        "llm_scoring": dict(llm_scoring) if llm_scoring else None,
                    },
                    "score_details": score_details,
                    "generation_config": (
                        dict(generation_config) if generation_config is not None else None
                    ),
                }
            )
        )

    return results


def export_tournament_results(
    *,
    target: str | Path,
    output_dir: str | Path,
    source_organization_name: str = "unknown",
    evaluator_relationship: str = "third_party",
    source_organization_url: str | None = None,
    source_organization_logo_url: str | None = None,
    eval_library_name: str = "dfm_evals.tournament",
    eval_library_version: str | None = None,
) -> list[Path]:
    from dfm_evals.tournament.exports import load_export_snapshot

    snapshot = load_export_snapshot(target)
    if len(snapshot.status.standings) == 0:
        raise ValueError(f"Tournament has no standings to export: {target}")

    destination_root = Path(output_dir)
    source_metadata = _build_source_metadata(
        source_name="tournament",
        source_organization_name=source_organization_name,
        evaluator_relationship=evaluator_relationship,
        source_organization_url=source_organization_url,
        source_organization_logo_url=source_organization_logo_url,
    )
    benchmark_name = _tournament_benchmark_name(snapshot.status.project_id)
    generation_config = _extract_tournament_generation_config(
        snapshot.config,
        snapshot.status,
    )
    source_data = _extract_tournament_source_data(
        config=snapshot.config,
        status=snapshot.status,
        benchmark_name=benchmark_name,
    )
    llm_scoring = _build_tournament_llm_scoring(snapshot.config)
    retrieved_timestamp = _now_unix_timestamp()
    library_version = eval_library_version or "unknown"

    written: list[Path] = []
    for rank, standing in enumerate(snapshot.status.standings, start=1):
        model_ref = snapshot.names_by_id.get(
            getattr(standing, "model_id"),
            getattr(standing, "model_name", None) or getattr(standing, "model_id"),
        )
        model_info = _parse_model_info(str(model_ref))
        evaluation_results = _extract_tournament_results(
            benchmark_name=benchmark_name,
            source_data=source_data,
            generation_config=generation_config,
            llm_scoring=llm_scoring,
            standing=standing,
            rank=rank,
        )
        evaluation_id = (
            f"{_sanitize_path_component(benchmark_name)}/{model_info['id']}/{retrieved_timestamp}"
        )
        record = _compact(
            {
                "schema_version": SCHEMA_VERSION,
                "evaluation_id": evaluation_id,
                "retrieved_timestamp": retrieved_timestamp,
                "source_metadata": source_metadata,
                "eval_library": {
                    "name": eval_library_name,
                    "version": library_version,
                },
                "model_info": model_info,
                "evaluation_results": evaluation_results,
            }
        )
        destination = _write_record(
            output_dir=destination_root,
            benchmark_name=benchmark_name,
            model_info=model_info,
            record=record,
        )
        written.append(destination)

    return written
