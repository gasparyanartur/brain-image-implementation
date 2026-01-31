import argparse
import logging
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel
import torch
import tqdm

from brain_image.data.tensorcache import TensorCache
from brain_image.model.encoder.img_encoder import ImageEncoderName
from brain_image.utils import casttensor, flatten_configs, setup_logging


class Arguments(BaseModel):
    embedding_types: list[ImageEncoderName]
    data_path: Path
    cache_dir: Path
    img_dir: Path
    stats_dir: Path
    split: Literal["train", "test"]


@torch.no_grad()
def get_embeddings_stats(
    tensorcache: TensorCache,
    img_paths: list[Path],
    embedding_names: list[ImageEncoderName],
    split: Literal["train", "test"],
) -> dict[str, dict[str, torch.Tensor]]:
    logging.info(f"Getting embedding stats for {embedding_names} - {len(img_paths)} images")
    _running_embeddings = {}

    for emb_name in embedding_names:
        arg_list = ((str(img_path), emb_name, split) for img_path in img_paths)
        _running_embeddings[emb_name] = tensorcache.batch_get(arg_list)
    

    logging.info(f"Keys gathered: {_running_embeddings.keys()}")

    _running_latents = _running_embeddings

    logging.info(f"Finished getting embeddings {_running_latents.keys()}")

    embedding_stats: dict[str, dict[str, torch.Tensor]] = {
        k: {
            "mean": torch.mean(v, dim=0),
            "std": torch.std(v, dim=0),
            "min": torch.min(v, dim=0).values,
            "max": torch.max(v, dim=0).values,
            "norm": v.norm(dim=-1).mean(),
        }
        for k, v in _running_latents.items()
    }

    logging.info(f"Finished getting embedding stats")
    return embedding_stats


def main(args: Arguments):
    setup_logging()
    logging.warning("Deprecated: The stats are computed inside of the datamodule now.")

    logging.info(f"Computing stats with arguments:")
    for k, v in flatten_configs(args).items():
        logging.info(f"\t{k}: {v}")


    tensor_cache = TensorCache(args.cache_dir)
    img_dir_path = (
        args.data_path / args.img_dir / "training_images"
        if args.split == "train"
        else args.data_path / args.img_dir / "test_images"
    )
    stats_dir_path = args.data_path / args.stats_dir / args.split
    stats_dir_path.mkdir(parents=True, exist_ok=True)

    img_paths = list(img_dir_path.rglob("*.jpg"))

    embedding_stats = get_embeddings_stats(
        tensorcache=tensor_cache,
        img_paths=img_paths,
        embedding_names=args.embedding_types,
        split=args.split,
    )

    logging.info(f"Gathered stats for {embedding_stats.keys()}")
    logging.info(f"Saving embedding stats to {stats_dir_path}")
    for emb_name, emb_stats in embedding_stats.items():
        emb_path = stats_dir_path / f"{emb_name}.pt"
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb_stats, emb_path)

        logging.info(f"Saved embedding stats to {emb_path}")
        for stat_name, stat in emb_stats.items():
            logging.info(f"\t{stat_name}: {stat.shape} | {stat.mean()} ± {stat.std() if len(stat.shape) > 0 else 0} | {stat}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "embedding_types",
        type=str,
        nargs="*",
        choices=[
            "clip_vitl14",
            "clip_vith14",
            "sd_variations_v2",
            "synclr_vitb16",
            "aligned_synclr_vitb16",
            "unaligned_synclr_vitb16",
        ],
    )
    parser.add_argument("--data_path", type=Path, default="data/things-eeg2")
    parser.add_argument("--cache_dir", type=Path, default="tensorcache")
    parser.add_argument("--img_dir", type=Path, default="imgs")
    parser.add_argument("--stats_dir", type=Path, default="stats")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])

    args = parser.parse_args()
    main(Arguments(**vars(args)))