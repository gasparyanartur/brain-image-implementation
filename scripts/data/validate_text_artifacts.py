"""Validate Qwen-caption and downstream text-embedding artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from brain_image.data.io import get_image_paths


DEFAULT_MODELS = ("t5_base", "clip_vitl14_text", "gemma_embedding_300m")


def load_caption_records(caption_path: Path) -> list[dict]:
    records = []
    with caption_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            if not record.get("path") or not record.get("caption", "").strip():
                raise ValueError(f"Missing path or caption on line {line_number}")
            records.append(record)
    return records


def latent_path(cache_dir: Path, image_path: Path, model_name: str, split: str) -> Path:
    cache_split = "train" if split == "train" else "test"
    return (cache_dir / model_name / cache_split / image_path).with_suffix(".pt")


def validate_text_artifacts(
    caption_path: Path,
    cache_dir: Path,
    stats_dir: Path,
    image_dir: Path,
    models: tuple[str, ...] = DEFAULT_MODELS,
    check_tensor_shapes: bool = False,
) -> dict[str, int]:
    records = load_caption_records(caption_path)
    paths = {record["path"] for record in records}
    if len(paths) != len(records):
        raise ValueError("Caption file contains duplicate image paths")

    summary: dict[str, int] = {"captions": len(records)}
    for split in ("train", "test"):
        image_paths = get_image_paths(image_dir, split, extensions=(".jpg",))
        missing_captions = [path for path in image_paths if str(path) not in paths]
        if missing_captions:
            raise FileNotFoundError(f"Missing captions for {len(missing_captions)} {split} images")
        summary[f"{split}_images"] = len(image_paths)

        for model_name in models:
            missing_latents = [
                path for path in image_paths if not latent_path(cache_dir, path, model_name, split).exists()
            ]
            if missing_latents:
                raise FileNotFoundError(f"Missing {model_name} latents for {len(missing_latents)} {split} images")

            stats_path = stats_dir / "datasets" / "things-eeg2" / ("train" if split == "train" else "test") / f"{model_name}.pt"
            if not stats_path.exists():
                raise FileNotFoundError(f"Missing statistics: {stats_path}")

            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            if tuple(stats["mean"].shape) != tuple(stats["std"].shape):
                raise ValueError(f"Statistics shape mismatch for {model_name}/{split}")

            if check_tensor_shapes:
                sample = torch.load(latent_path(cache_dir, image_paths[0], model_name, split), map_location="cpu", weights_only=True)
                if tuple(sample.shape) != tuple(stats["mean"].shape):
                    raise ValueError(f"Latent/statistics dimension mismatch for {model_name}/{split}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-path", type=Path, default=Path("data/things-eeg2/captions/local.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("tensorcache"))
    parser.add_argument("--stats-dir", type=Path, default=Path("statistics"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/things-eeg2/imgs"))
    parser.add_argument("--check-tensor-shapes", action="store_true")
    args = parser.parse_args()
    summary = validate_text_artifacts(**vars(args))
    print("Text artifacts are valid:", summary)


if __name__ == "__main__":
    main()
