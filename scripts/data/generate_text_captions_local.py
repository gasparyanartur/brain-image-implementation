from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import hydra
import torch
import tqdm
import yaml
from omegaconf import DictConfig
from PIL import Image

from brain_image.configs import BaseConfig, GlobalConfig, get_device_str
from brain_image.data.datamodule import EEGDataModule
from brain_image.data.dataset.union import EEGDatasetConfigType
from brain_image.utils import setup


def extract_label(path: Path) -> str:
    return " ".join(path.stem.split("_")[:-1])


def load_existing_paths(caption_path: Path) -> set[str]:
    if not caption_path.exists():
        return set()
    existing: set[str] = set()
    with open(caption_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                existing.add(entry["path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return existing


class LocalCaptionConfig(BaseConfig):
    dataset: EEGDatasetConfigType
    splits: list[Literal["train", "test"]] = ["train", "test"]
    caption_path: Path = Path("data/things-eeg2/captions/local.jsonl")
    provenance_path: Path | None = None
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    batch_size: int = 8
    dtype: str = "bfloat16"
    device: str | None = None
    system_prompt: str = (
        "You are given an image. Generate a single, detailed caption for the image. "
        "The caption must be visually precise, descriptive, and neutral. "
        "Do not assume any prior context. "
        "Avoid storytelling, interpretation, or emotional language. "
        "Describe only what is clearly visible."
    )
    max_new_tokens: int = 256


def run_captioning(
    dataset,
    config: LocalCaptionConfig,
    split: str,
    model,
    processor,
    device: str,
) -> None:
    from qwen_vl_utils import process_vision_info

    existing_paths = load_existing_paths(config.caption_path)
    img_paths = dataset.get_image_paths()
    pending_paths = [p for p in img_paths if str(p) not in existing_paths]

    if not pending_paths:
        logging.info(f"[{split}] All {len(img_paths)} images already captioned, skipping.")
        return

    logging.info(f"[{split}] Captioning {len(pending_paths)} images ({len(existing_paths)} already done)")
    config.caption_path.parent.mkdir(parents=True, exist_ok=True)

    n_batches = (len(pending_paths) + config.batch_size - 1) // config.batch_size
    with open(config.caption_path, "a") as out_file, torch.no_grad():
        pbar = tqdm.tqdm(
            range(0, len(pending_paths), config.batch_size),
            total=n_batches,
            desc=f"[{split}]",
            unit="batch",
            dynamic_ncols=True,
        )
        for i in pbar:
            batch_paths = pending_paths[i : i + config.batch_size]
            pbar.set_postfix(imgs=f"{min(i + config.batch_size, len(pending_paths))}/{len(pending_paths)}", label=extract_label(batch_paths[0]))

            # Build per-image message lists
            batch_messages = [
                [
                    {"role": "system", "content": config.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": Image.open(p).convert("RGB")},
                            {"type": "text", "text": f"Label: {extract_label(p)}. Describe this image."},
                        ],
                    },
                ]
                for p in batch_paths
            ]

            texts = [processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in batch_messages]

            # Flatten messages so process_vision_info can collect all images
            flat_messages = [msg for msgs in batch_messages for msg in msgs]
            image_inputs, video_inputs = process_vision_info(flat_messages)

            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            ).to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for path, caption in zip(batch_paths, output_texts):
                entry = {"path": str(path), "split": split, "label": extract_label(path), "caption": caption.strip()}
                out_file.write(json.dumps(entry) + "\n")

            out_file.flush()


def generate_local_captions(config: LocalCaptionConfig) -> None:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = config.device or get_device_str()
    provenance_path = config.provenance_path or config.caption_path.with_suffix(".yaml")
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "model_name": config.model_name,
        "system_prompt": config.system_prompt,
        "max_new_tokens": config.max_new_tokens,
        "batch_size": config.batch_size,
        "dtype": config.dtype,
        "device": device,
        "splits": config.splits,
        "caption_path": str(config.caption_path),
    }
    provenance_path.write_text(yaml.safe_dump(provenance, sort_keys=False))
    logging.info(f"Saved caption provenance to {provenance_path}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[config.dtype]

    logging.info(f"Loading model {config.model_name} on {device} ({config.dtype})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(config.model_name)
    processor.tokenizer.padding_side = "left"  # Required for correct batch inference with decoder-only models

    for split in tqdm.tqdm(config.splits, desc="Splits", unit="split"):
        logging.info(f"Processing split: {split}")
        dataset = EEGDataModule(config.dataset).create_dataset(
            split,
            preload_cache=False,
            embeddings_to_compute_stats=[],
            compute_stats=False,
        )
        logging.info(f"Dataset for split '{split}' has {len(dataset)} samples.")
        run_captioning(dataset, config, split, model, processor, device)


@hydra.main(config_path=str(GlobalConfig.CONFIGS_DIR), config_name="generate_text_captions_local", version_base=None)
def main(cfg: DictConfig) -> None:
    setup()
    config = LocalCaptionConfig.from_hydra_config(cfg)
    for key, value in config.model_dump(mode="json").items():
        logging.info(f"{key}: {value}")
    generate_local_captions(config)


if __name__ == "__main__":
    main()
