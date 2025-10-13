import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import Literal, TypedDict
from pydantic import BaseModel
from torch import Tensor
import torch

from brain_image.data import get_image_paths, load_all_eeg_data


class SampleType(TypedDict):
    img_path: str
    eeg: Tensor
    sub: int


def merge_data(
    sub: int, img_paths: list[Path], eeg_data: Tensor, idxs: Tensor
) -> list[SampleType]:
    merged_data = []

    for i in range(eeg_data.size(0)):
        idx = idxs[i]
        img_path = img_paths[int(idx)]
        eeg = eeg_data[i]

        joined_object = {"img_path": str(img_path), "eeg": eeg, "sub": sub, "idx": idx}

        merged_data.append(joined_object)

    return merged_data


class Arguments(BaseModel):
    splits: list[Literal["train", "test"]]
    eeg_dir: Path
    image_dir: Path
    prepared_data_dir: Path
    subs: list[int]


def main(args: Arguments):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Arguments: {args}")

    for split in args.splits:
        logging.info(f"Starting data preparation for split: {split}")
        logging.info(f"Loading subs: {args.subs}")
        sub_paths = [
            Path(
                args.eeg_dir
                / f"sub-{sub:02}/preprocessed_eeg_{'training' if split == 'train' else 'test'}.npy"
            )
            for sub in args.subs
        ]
        img_paths = get_image_paths(
            image_dir=args.image_dir,
            split=split,
        )
        logging.info(f"Loaded {len(img_paths)} image paths")
        
        preprocess_configs = {
            "unpack_repetitions": True,
            "normalize": False
        } if split == "train" else {
            "unpack_repetitions": False, 
            "normalize": False
        }
            
        eeg_data, idxs, *_ = load_all_eeg_data(eeg_paths=sub_paths, preprocess_configs=preprocess_configs)   # <sub, i, s, t>
        logging.info(f"Loaded EEG data with shape: {eeg_data.shape}")

        for i_sub, sub in enumerate(args.subs):
            logging.info(f"Creating data for sub: {sub}")
            merged_data = merge_data(sub, img_paths, eeg_data[i_sub], idxs[i_sub])
            dst_path = args.prepared_data_dir / f"sub-{sub:02}" / f"{split}.pt"
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            logging.info(f"Saving data to {dst_path}")
            torch.save(merged_data, dst_path)


if __name__ == "__main__":
    parser = ArgumentParser("Prepare data for easy loading during runtime")

    parser.add_argument(
        "--eeg_dir",
        "-d",
        default="data/things-eeg2/eeg",
        type=Path,
        help="Path to preprocessed EEG data directory",
    )
    parser.add_argument(
        "--image_dir",
        "-i",
        default="data/things-eeg2/imgs",
        type=Path,
        help="Path to target images",
    )
    parser.add_argument(
        "--prepared_data_dir",
        "-o",
        default="data/things-eeg2/prepared",
        type=Path,
        help="Output directory",
    )
    parser.add_argument("--splits", type=str, nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument(
        "--subs", "-s", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    args = Arguments(**vars(parser.parse_args()))
    main(args)
