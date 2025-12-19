from argparse import ArgumentParser
import logging
import sys
import tarfile
from typing import Any, Literal, TypedDict
import gdown
from pathlib import Path
import zipfile
import huggingface_hub
from pydantic import BaseModel
import requests
import torch
import tqdm
import pandas as pd

from data.data import EEGDataset, SampleType, get_image_paths, load_all_eeg_data



class SetupDataArguments(BaseModel):
    subs: list[int] | None
    data_path: Path
    raw_dir: str
    img_dir: str
    preprocessed_eeg_dir: str
    prepared_data_dir: str
    get_eeg: bool
    get_img: bool
    get_all: bool
    download_file: bool
    extract_file: bool
    preprocess_data: bool
    remove_extracted: bool
    prepare_data: bool
    all_stages: bool
    skip_existing: bool
    seed: int
    data_source: Literal["hf"]


def main(args: SetupDataArguments):
    if args.all_stages or (
        not args.download_file
        and not args.extract_file
        and not args.remove_extracted
        and not args.prepare_data
    ):
        args.download_file = True
        args.extract_file = True
        args.remove_extracted = True
        args.prepare_data = True
        args.all_stages = True

    if args.get_all or (not args.get_eeg and not args.get_img):
        args.get_eeg = True
        args.get_img = True
        args.get_all = True

    if args.subs is None:
        args.subs = list(range(1, 21))

    logging.info(f"Preparing data, arguments:")
    for k, v in vars(args).items():
        logging.info(f"\t{k}: {v}")

    args.data_path.mkdir(parents=True, exist_ok=True)
    (args.data_path / args.preprocessed_eeg_dir).mkdir(exist_ok=True)

    preprocessed_eeg_dir = args.data_path / "preprocessed_eeg"
    img_dir = args.data_path / "stimuli/images"

    if args.get_eeg:
        huggingface_hub.snapshot_download("Alljoined/Alljoined-1.6M", allow_patterns="preprocessed_eeg/*", repo_type="dataset", local_dir=args.data_path)

    # Get image
    if args.get_img:
        imgs_zip = args.data_path / "stimuli.zip"

        if args.download_file and not imgs_zip.exists():
            huggingface_hub.snapshot_download("Alljoined/Alljoined-1.6M", allow_patterns="stimuli.zip", repo_type="dataset", local_dir=args.data_path)
        else:
            logging.info(f"File already exists: {imgs_zip}, skipping download...")

        if args.extract_file:
            img_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Extracting file to {img_dir}")
            with zipfile.ZipFile(imgs_zip, "r") as zf:
                zf.extractall(img_dir)

        else:
            logging.info(f"Skipping extraction...")

        if args.remove_extracted:
            imgs_zip.unlink()

    if args.prepare_data:
        logging.info(f"Preparing data...")
        prepared_dir = args.data_path / args.prepared_data_dir
        prepared_dir.mkdir(parents=True, exist_ok=True)


        all_metadatas = [
            [pd.read_parquet(preprocessed_eeg_dir / f"sub-{s:02}") / "experiment_metadata_categories.parquet" for s in args.subs]
        ]


        for split in ["train", "test"]:
            logging.info(f"Starting data preparation for split: {split}")
            logging.info(f"Loading subs: {args.subs}")
            sub_paths = [
                Path(
                    preprocessed_eeg_dir
                    / f"sub-{sub:02}/preprocessed_eeg_{'training' if split == 'train' else 'test'}.npy"
                )
                for sub in args.subs
            ]
            img_paths = get_image_paths(
                image_dir=img_dir,
                split=split,  # type: ignore
            )
            logging.info(f"Loaded {len(img_paths)} image paths")

            preprocess_configs = (
                {"unpack_repetitions": True, "normalize": False}
                if split == "train"
                else {"unpack_repetitions": False, "normalize": False}
            )

            eeg_data, idxs, *_ = load_all_eeg_data(
                eeg_paths=sub_paths, preprocess_configs=preprocess_configs
            )  # <sub, i, s, t>
            logging.info(f"Loaded EEG data with shape: {eeg_data.shape}")

            for i_sub, sub in enumerate(args.subs):
                logging.info(f"Creating data for sub: {sub}")
                merged_data = _merge_data(sub, img_paths, eeg_data[i_sub], idxs[i_sub])
                dst_path = prepared_dir / f"sub-{sub:02}" / f"{split}.pt"
                dst_path.parent.mkdir(parents=True, exist_ok=True)

                logging.info(f"Saving data to {dst_path}")
                torch.save(merged_data, dst_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        "Downloads and preprocesses image and EEG data. Run with -all and -s* to run the full pipeline"
    )
    parser.add_argument("--subs", "-s", type=int, nargs="*", default=None)
    parser.add_argument("--data_path", type=Path, default="data/alljoined-1_6m")
    parser.add_argument("--raw_dir", type=str, default="raw_eeg")
    parser.add_argument("--img_dir", type=str, default="imgs")
    parser.add_argument("--preprocessed_eeg_dir", type=str, default="eeg")
    parser.add_argument("--prepared_data_dir", type=str, default="prepared")
    parser.add_argument("--get_eeg", "-eeg", action="store_true")
    parser.add_argument("--get_img", "-img", action="store_true")
    parser.add_argument("--get_all", "-all", action="store_true")
    parser.add_argument("--download_file", "-s1", action="store_true")
    parser.add_argument("--extract_file", "-s2", action="store_true")
    parser.add_argument("--preprocess_data", "-s3", action="store_true")
    parser.add_argument("--remove_extracted", "-s4", action="store_true")
    parser.add_argument("--prepare_data", "-s5", action="store_true")
    parser.add_argument("--all_stages", "-s*", action="store_true")
    parser.add_argument("--skip_existing", "-skip", action="store_true")
    parser.add_argument("--seed", type=int, default=20200220)
    parser.add_argument(
        "--data_source", type=str, default="alignvis", choices=["things", "alignvis"]
    )

    args = parser.parse_args()
    print(args)
    main(SetupDataArguments(**vars(args)))
