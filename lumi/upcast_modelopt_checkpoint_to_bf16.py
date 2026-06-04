#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


CONFIG_KEYS_TO_DROP = ("quantization_config",)
PARAMS_KEYS_TO_DROP = ("quantization",)
SUPPORT_FILES_TO_DROP = ("hf_quant_config.json",)
MISTRAL_FP8_SCALE_SUFFIXES = (".qscale_weight", ".qscale_act")
MODEL_OPT_AUX_SUFFIXES = (
    ".input_scale",
    ".weight_scale",
    ".weight_scale_2",
    ".k_scale",
    ".v_scale",
)
FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)

FP4_MAGNITUDE_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


@dataclass(frozen=True)
class ConversionStats:
    shard_name: str
    converted_fp8_tensors: int
    converted_nvfp4_tensors: int
    copied_tensors: int
    skipped_auxiliary_tensors: int
    bytes_written: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upcast a ModelOpt mixed FP8/NVFP4 safetensors checkpoint into BF16"
            " shard-by-shard."
        )
    )
    parser.add_argument("--source", required=True, help="Source checkpoint directory")
    parser.add_argument("--output", required=True, help="Output checkpoint directory")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip shards that already exist in the output directory",
    )
    parser.add_argument(
        "--only-shards",
        nargs="*",
        default=None,
        help="Optional list of shard filenames to convert",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    return parser.parse_args()


def load_index(index_path: Path) -> dict:
    with index_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"output directory is not empty: {output_dir} (pass --overwrite to reuse)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def patch_config(config_path: Path, output_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    for key in CONFIG_KEYS_TO_DROP:
        config.pop(key, None)
    config["dtype"] = "bfloat16"
    config["torch_dtype"] = "bfloat16"
    write_json(output_path, config)


def patch_params(params_path: Path, output_path: Path) -> None:
    with params_path.open("r", encoding="utf-8") as fh:
        params = json.load(fh)
    for key in PARAMS_KEYS_TO_DROP:
        params.pop(key, None)
    write_json(output_path, params)


def copy_support_files(source_dir: Path, output_dir: Path) -> None:
    for dropped_name in SUPPORT_FILES_TO_DROP:
        dropped_path = output_dir / dropped_name
        if dropped_path.exists():
            dropped_path.unlink()

    for entry in source_dir.iterdir():
        if entry.name.endswith(".safetensors"):
            continue
        if entry.name.endswith(".index.json"):
            continue
        if entry.name in SUPPORT_FILES_TO_DROP:
            continue
        if entry.name == "config.json":
            patch_config(entry, output_dir / entry.name)
            continue
        if entry.name == "params.json":
            patch_params(entry, output_dir / entry.name)
            continue
        target = output_dir / entry.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if entry.is_file():
            shutil.copy2(entry, target)


def select_shards(index_data: dict, only_shards: list[str] | None) -> list[str]:
    shards = sorted(set(index_data["weight_map"].values()))
    if not only_shards:
        return shards
    requested = set(only_shards)
    missing = sorted(requested - set(shards))
    if missing:
        raise SystemExit(f"unknown shards requested: {missing}")
    return [shard for shard in shards if shard in requested]


def is_auxiliary_quant_tensor(key: str) -> bool:
    if key.endswith(MISTRAL_FP8_SCALE_SUFFIXES):
        return True
    if key.endswith(MODEL_OPT_AUX_SUFFIXES):
        return True
    return False


def dequantize_mistral_fp8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.to(torch.bfloat16) * scale.to(torch.bfloat16)


def dequantize_modelopt_fp8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.to(torch.bfloat16) * scale.max().to(torch.bfloat16)


def break_fp4_bytes(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError(f"expected packed NVFP4 weights as uint8, got {packed.dtype}")
    rows, cols = packed.shape
    flattened = packed.reshape(-1)
    low = flattened & 0x0F
    high = (flattened & 0xF0) >> 4
    combined = torch.stack((low, high), dim=1).reshape(-1)
    signs = (combined & 0x08).to(torch.bool)
    magnitudes = (combined & 0x07).to(torch.long)
    values = FP4_MAGNITUDE_TABLE.to(packed.device)[magnitudes]
    values = values * torch.where(signs, -1.0, 1.0)
    return values.reshape(rows, cols * 2)


def convert_swizzled_scale_to_linear(
    swizzled: torch.Tensor,
    output_size: int,
    input_size: int,
    group_size: int,
) -> torch.Tensor:
    if swizzled.ndim != 2:
        raise ValueError(
            "expected 2D NVFP4 weight scale tensor, got "
            f"{tuple(swizzled.shape)}"
        )
    output_tiles = (output_size + 128 - 1) // 128
    packed_tile_width = group_size * 4
    input_tiles = (input_size + packed_tile_width - 1) // packed_tile_width
    reshaped = swizzled.reshape(1, output_tiles, input_tiles, 32, 4, 4)
    linear = reshaped.permute(0, 1, 4, 3, 2, 5).reshape(
        output_tiles * 128,
        input_tiles * packed_tile_width // group_size,
    )
    return linear[:output_size, : input_size // group_size]


def infer_nvfp4_group_size(weight: torch.Tensor, weight_scale: torch.Tensor) -> int:
    input_size = weight.shape[1] * 2
    if weight_scale.ndim != 2:
        raise ValueError(
            "expected 2D NVFP4 weight_scale tensor, got "
            f"{tuple(weight_scale.shape)}"
        )
    if weight_scale.shape[1] == 0:
        raise ValueError("NVFP4 weight_scale has zero columns")
    group_size, remainder = divmod(input_size, weight_scale.shape[1])
    if remainder != 0 or group_size <= 0:
        raise ValueError(
            "failed to infer NVFP4 group size from weight shape "
            f"{tuple(weight.shape)} and weight_scale shape {tuple(weight_scale.shape)}"
        )
    return group_size


def dequantize_nvfp4_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    group_size = infer_nvfp4_group_size(weight, weight_scale)
    unpacked = break_fp4_bytes(weight)
    linear_scale = convert_swizzled_scale_to_linear(
        weight_scale,
        output_size=weight.shape[0],
        input_size=unpacked.shape[1],
        group_size=group_size,
    )
    dequant_scale = linear_scale.to(torch.float32) * weight_scale_2.max().to(torch.float32)
    restored = unpacked.reshape(weight.shape[0], -1, group_size) * dequant_scale.unsqueeze(-1)
    return restored.reshape(weight.shape[0], unpacked.shape[1]).to(torch.bfloat16)


def convert_tensor(
    *,
    fh,
    key: str,
    tensor: torch.Tensor,
    key_set: set[str],
) -> tuple[torch.Tensor, str] | None:
    if is_auxiliary_quant_tensor(key):
        return None

    if key.endswith(".weight") and tensor.dtype in FP8_DTYPES:
        scale_key = key[: -len(".weight")] + ".weight_scale"
        mistral_scale_key = key[: -len(".weight")] + ".qscale_weight"
        if scale_key in key_set:
            scale = fh.get_tensor(scale_key)
            return dequantize_modelopt_fp8_weight(tensor, scale), "fp8"
        if mistral_scale_key in key_set:
            scale = fh.get_tensor(mistral_scale_key)
            return dequantize_mistral_fp8_weight(tensor, scale), "fp8"
        raise RuntimeError(f"missing FP8 scale tensor for {key}")

    if key.endswith(".weight") and tensor.dtype == torch.uint8:
        scale_key = key[: -len(".weight")] + ".weight_scale"
        scale2_key = key[: -len(".weight")] + ".weight_scale_2"
        if scale_key not in key_set or scale2_key not in key_set:
            raise RuntimeError(
                f"missing NVFP4 scale tensors for {key}: "
                f"{scale_key}, {scale2_key}"
            )
        return (
            dequantize_nvfp4_weight(
                tensor,
                fh.get_tensor(scale_key),
                fh.get_tensor(scale2_key),
            ),
            "nvfp4",
        )

    return tensor, "copied"


def convert_shard(source_path: Path, output_path: Path) -> ConversionStats:
    tensors: dict[str, torch.Tensor] = {}
    converted_fp8 = 0
    converted_nvfp4 = 0
    copied = 0
    skipped_auxiliary = 0

    with safe_open(source_path, framework="pt") as fh:
        keys = list(fh.keys())
        key_set = set(keys)
        metadata = fh.metadata()
        for key in keys:
            tensor = fh.get_tensor(key)
            converted = convert_tensor(fh=fh, key=key, tensor=tensor, key_set=key_set)
            if converted is None:
                skipped_auxiliary += 1
                continue

            converted_tensor, status = converted
            tensors[key] = converted_tensor
            if status == "fp8":
                converted_fp8 += 1
            elif status == "nvfp4":
                converted_nvfp4 += 1
            else:
                copied += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=output_path.parent,
        prefix=output_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp_fh:
        tmp_path = Path(tmp_fh.name)
    try:
        save_file(tensors, str(tmp_path), metadata=metadata)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return ConversionStats(
        shard_name=source_path.name,
        converted_fp8_tensors=converted_fp8,
        converted_nvfp4_tensors=converted_nvfp4,
        copied_tensors=copied,
        skipped_auxiliary_tensors=skipped_auxiliary,
        bytes_written=output_path.stat().st_size,
    )


def build_output_index(
    source_dir: Path,
    output_dir: Path,
    index_data: dict,
    shard_names: list[str],
) -> dict:
    new_weight_map: dict[str, str] = {}
    total_size = 0

    for shard_name in shard_names:
        source_path = source_dir / shard_name
        output_path = output_dir / shard_name
        if not output_path.exists():
            raise RuntimeError(f"missing converted shard: {output_path}")

        total_size += output_path.stat().st_size
        with safe_open(source_path, framework="pt") as fh:
            for key in fh.keys():
                if is_auxiliary_quant_tensor(key):
                    continue
                new_weight_map[key] = shard_name

    return {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }


def single_safetensors_name(source_dir: Path) -> str | None:
    candidates = sorted(
        entry.name
        for entry in source_dir.iterdir()
        if entry.name.endswith(".safetensors") and entry.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if "consolidated.safetensors" in candidates:
        return "consolidated.safetensors"
    return None


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()

    if not source_dir.is_dir():
        raise SystemExit(f"source directory does not exist: {source_dir}")

    index_candidates = [
        source_dir / "consolidated.safetensors.index.json",
        source_dir / "model.safetensors.index.json",
    ]
    index_path = next((candidate for candidate in index_candidates if candidate.exists()), None)

    ensure_output_dir(output_dir, overwrite=args.overwrite)
    copy_support_files(source_dir, output_dir)

    if index_path is None:
        shard_name = single_safetensors_name(source_dir)
        if shard_name is None:
            raise SystemExit(f"no safetensors index or single safetensors file found in {source_dir}")
        if args.only_shards and shard_name not in set(args.only_shards):
            raise SystemExit(f"unknown shard requested for single-file checkpoint: {args.only_shards}")
        index_data = None
        shard_names = [shard_name]
    else:
        index_data = load_index(index_path)
        shard_names = select_shards(index_data, args.only_shards)
    conversion_manifest: list[dict[str, int | str]] = []

    for shard_name in shard_names:
        source_path = source_dir / shard_name
        output_path = output_dir / shard_name
        if args.skip_existing and output_path.exists():
            print(f"SKIP {shard_name} existing={output_path}")
            continue

        print(f"CONVERT {source_path} -> {output_path}")
        stats = convert_shard(source_path, output_path)
        conversion_manifest.append(
            {
                "shard_name": stats.shard_name,
                "converted_fp8_tensors": stats.converted_fp8_tensors,
                "converted_nvfp4_tensors": stats.converted_nvfp4_tensors,
                "copied_tensors": stats.copied_tensors,
                "skipped_auxiliary_tensors": stats.skipped_auxiliary_tensors,
                "bytes_written": stats.bytes_written,
            }
        )
        print(
            "DONE"
            f" shard={stats.shard_name}"
            f" fp8={stats.converted_fp8_tensors}"
            f" nvfp4={stats.converted_nvfp4_tensors}"
            f" copied={stats.copied_tensors}"
            f" skipped_aux={stats.skipped_auxiliary_tensors}"
            f" bytes={stats.bytes_written}"
        )

    if index_data is not None and index_path is not None:
        output_index = build_output_index(source_dir, output_dir, index_data, shard_names)
        write_json(output_dir / index_path.name, output_index)
    write_json(
        output_dir / "upcast_manifest.json",
        {
            "source": str(source_dir),
            "output": str(output_dir),
            "source_index": index_path.name if index_path is not None else None,
            "converted_shards": shard_names,
            "manifest": conversion_manifest,
        },
    )

    print(f"OUTPUT {output_dir}")


if __name__ == "__main__":
    main()
