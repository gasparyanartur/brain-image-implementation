from argparse import ArgumentParser
import logging
import sys
import tarfile
from typing import Any, Literal, TypedDict
import gdown
from pathlib import Path
import zipfile
from pydantic import BaseModel
import requests
import torch
import tqdm

from brain_image.data import EEGDataset, SampleType, get_image_paths, load_all_eeg_data


"""
Raw EEG Data taken from https://drive.google.com/drive/folders/1KnOcV38RthPcpZR2vtiSm0jtZ6p63RNt
Find more information here: https://osf.io/crxs4/overview"""
_THINGS_DATA_URL: dict[str, str] = {"sub-08": "1mkOEFmoSyEZiIqa7fZ47Q00V0PDJxqjQ"}


def _get_alignvis_hf_url(sub: int, split: Literal["train", "test"]):
    idx = 2 * (sub - 1) + (1 if split == "train" else 0)
    base_url = (
        "https://huggingface.co/datasets/nonarjb/alignvis/resolve/main/things_eeg_2"
    )
    return f"{base_url}/things_eeg_2-Preprocessed_data_250Hz-{idx:06d}.tar"


_IMG_URL = "https://files.de-1.osf.io/v1/resources/y63gw/providers/osfstorage/?zip="


def _download_to_file(
    url,
    file_path,
    verbose: bool = True,
    progress_bar: bool = True,
    chunk_size: int = 1024,
):
    if verbose:
        logging.info(f"Downloading file from {url} to {file_path}")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    response = requests.get(url, stream=True, headers=headers)
    total_size = int(response.headers.get("content-length", 0))
    written_size = 0
    with open(file_path, "wb") as f:
        with tqdm.tqdm(
            response.iter_content(chunk_size=chunk_size),
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Downloading",
            disable=not progress_bar,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                chunk_size = len(chunk)
                if chunk_size == 0:
                    continue

                f.write(chunk)
                written_size += chunk_size
                pbar.update(chunk_size)

    if total_size > 0 and (written_size != total_size):
        raise ValueError(
            f"Downloaded size does not match expected size: {written_size} != {total_size}"
        )


def _merge_data(
    sub: int, img_paths: list[Path], eeg_data: torch.Tensor, idxs: torch.Tensor
) -> list[SampleType]:
    merged_data = []

    for i in range(eeg_data.size(0)):
        idx = idxs[i]
        img_path = img_paths[int(idx)]
        eeg = eeg_data[i]

        joined_object = {"img_path": str(img_path), "eeg": eeg, "sub": sub, "idx": idx}

        merged_data.append(joined_object)

    return merged_data


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
    get_stats: bool
    all_stages: bool
    skip_existing: bool
    seed: int
    data_source: Literal["things", "alignvis"]


def main(args: SetupDataArguments):
    if args.all_stages or (
        not args.download_file
        and not args.extract_file
        and not args.preprocess_data
        and not args.remove_extracted
        and not args.prepare_data
    ):
        args.download_file = True
        args.extract_file = True
        args.preprocess_data = True
        args.remove_extracted = True
        args.prepare_data = True
        args.all_stages = True

    if args.get_all or (not args.get_eeg and not args.get_img):
        args.get_eeg = True
        args.get_img = True
        args.get_all = True

    if args.subs is None:
        args.subs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    logging.info(f"Preparing data, arguments:")
    for k, v in vars(args).items():
        logging.info(f"\t{k}: {v}")

    args.data_path.mkdir(parents=True, exist_ok=True)
    (args.data_path / args.preprocessed_eeg_dir).mkdir(exist_ok=True)

    preprocessed_eeg_dir = args.data_path / args.preprocessed_eeg_dir
    img_dir = args.data_path / args.img_dir

    if args.get_eeg:
        logging.info(f"Processing EEG data for subs: {args.subs}")
        for sub in args.subs:
            logging.info(f"Processing subject {sub}")
            sub_name = f"sub-{sub:02d}"

            if args.data_source == "things":
                raw_file_path = args.data_path / args.raw_dir / f"{sub_name}.zip"
                extracted_path = args.data_path / args.raw_dir / sub_name

                url = _THINGS_DATA_URL[sub_name]

                if args.download_file and not (
                    args.skip_existing and raw_file_path.exists()
                ):
                    logging.info(f"Downloading file from {url} to {raw_file_path}")
                    gdown.download(output=str(raw_file_path), id=url)
                elif args.skip_existing and raw_file_path.exists():
                    logging.info(
                        f"File already exists: {raw_file_path}, skipping this step..."
                    )
                else:
                    raise FileNotFoundError(f"File not found: {raw_file_path}")

                if args.extract_file and not (
                    args.skip_existing and extracted_path.exists()
                ):
                    logging.info(f"Extracting file to {extracted_path}")
                    with zipfile.ZipFile(raw_file_path, "r") as zf:
                        zf.extractall(extracted_path.parent)
                elif args.skip_existing and extracted_path.exists():
                    logging.info(
                        f"Directory already exists: {extracted_path}, skipping this step..."
                    )
                else:
                    raise FileNotFoundError(f"Directory not found: {extracted_path}")

                if args.preprocess_data:
                    from brain_image.data_preprocessing import (
                        generate_preprocessed_dataset,
                    )

                    generate_preprocessed_dataset(args.data_path, sub, seed=args.seed)

                if args.remove_extracted:
                    raw_file_path.unlink()

            elif args.data_source == "alignvis":
                url_train = _get_alignvis_hf_url(sub, "train")
                url_test = _get_alignvis_hf_url(sub, "test")

                raw_file_train_path = (
                    args.data_path / args.raw_dir / f"{sub_name}_train.tar"
                )
                raw_file_test_path = (
                    args.data_path / args.raw_dir / f"{sub_name}_test.tar"
                )

                if args.download_file:
                    if not (args.skip_existing and raw_file_train_path.exists()):
                        _download_to_file(url_train, raw_file_train_path)
                    else:
                        logging.info(
                            f"File already exists: {raw_file_train_path}, skipping this step..."
                        )

                    if not raw_file_test_path.exists():
                        _download_to_file(url_test, raw_file_test_path)
                    else:
                        logging.info(
                            f"File already exists: {raw_file_test_path}, skipping this step..."
                        )

                else:
                    logging.info(
                        f"Files already exist for {sub_name}, skipping download..."
                    )

                if args.extract_file:
                    preprocessed_eeg_dir.mkdir(parents=True, exist_ok=True)

                    logging.info(f"Extracting train file to {preprocessed_eeg_dir}")
                    with tarfile.open(raw_file_train_path, "r") as tar:
                        tar.extractall(preprocessed_eeg_dir)
                    logging.info(f"Extracting test file to {preprocessed_eeg_dir}")
                    with tarfile.open(raw_file_test_path, "r") as tar:
                        tar.extractall(preprocessed_eeg_dir)
                else:
                    logging.info(f"Skipping extraction for {sub_name}...")

                if args.remove_extracted:
                    raw_file_train_path.unlink()
                    raw_file_test_path.unlink()

            else:
                raise ValueError(f"Unknown data source: {args.data_source}")

            logging.info(f"Finished processing subject {sub}")

    # Get image
    if args.get_img:
        imgs_zip = args.data_path / args.img_dir / "raw.zip"
        training_img_path_zip = img_dir / "training_images.zip"
        test_img_path_zip = img_dir / "test_images.zip"

        if args.download_file and not imgs_zip.exists():
            _download_to_file(_IMG_URL, imgs_zip)
        else:
            logging.info(f"File already exists: {imgs_zip}, skipping download...")

        if args.extract_file:
            img_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Extracting file to {img_dir}")
            with zipfile.ZipFile(imgs_zip, "r") as zf:
                zf.extractall(img_dir)

            with zipfile.ZipFile(training_img_path_zip, "r") as zf:
                zf.extractall(img_dir)

            with zipfile.ZipFile(test_img_path_zip, "r") as zf:
                zf.extractall(img_dir)
        else:
            logging.info(f"Skipping extraction...")

        if args.remove_extracted:
            imgs_zip.unlink()
            training_img_path_zip.unlink()
            test_img_path_zip.unlink()

    if args.prepare_data:
        logging.info(f"Preparing data...")
        prepared_dir = args.data_path / args.prepared_data_dir
        prepared_dir.mkdir(parents=True, exist_ok=True)

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
    parser.add_argument("--data_path", type=Path, default="data/things-eeg2")
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
