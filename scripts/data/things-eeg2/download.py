from pathlib import Path
import argparse
import logging
from typing import cast
import zipfile

import huggingface_hub

from brain_image.utils import setup_logging
import tempfile


THINGS_URL = "gasparyanartur/things-eeg2"
SUB_MIN = 1
SUB_MAX = 10


def main(args):
    setup_logging()
    logging.info(f"Downloaded Things EEG dataset with configs:")

    for k, v in args.items():
        logging.info(f"{k}: {v}")

    data_path = args["data_path"]
    data_path.mkdir(parents=True, exist_ok=True)

    subs = list(args["subs"]) if args["subs"] else list(range(SUB_MIN, SUB_MAX))
    sub_names = [f"sub-{sub:02d}" for sub in subs]

    download_types = set(args["download_types"])

    if "imgs" in download_types:
        logging.info("Downloading images...")
        imgs_path = data_path / args["img_dir"]

        train_unzipped_path = imgs_path / "training_images"
        test_unzipped_path = imgs_path / "test_images"

        if train_unzipped_path.exists() and test_unzipped_path.exists():
            logging.info("Images already downloaded and extracted, skipping...")

        else:
            train_zip_path = imgs_path / "training_images.zip"
            test_zip_path = imgs_path / "test_images.zip"

            if train_zip_path.exists() and test_zip_path.exists():
                logging.info("Images already downloaded")
            else:
                logging.info("Downloading images from Hugging Face Hub...")
                imgs_path.mkdir(parents=True, exist_ok=True)

                huggingface_hub.snapshot_download(
                        THINGS_URL,
                        allow_patterns="imgs*",
                        repo_type="dataset",
                        local_dir=data_path,
                    )

            assert train_zip_path.exists() and test_zip_path.exists(), "Images not downloaded"
            logging.info("Extracting images...")
            with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
                zip_ref.extractall(imgs_path)
            
            with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
                zip_ref.extractall(imgs_path)

    if "eeg" in download_types:
        logging.info("Downloading EEG data...")

        for sub in sub_names:
            sub_path = data_path / args["raw_eeg_dir"] / sub
            if (
                len(list(sub_path.rglob("*/raw_eeg_training.py"))) > 0
                and len(list(sub_path.rglob("*/raw_eeg_test.py"))) > 0
            ):
                logging.info(f"EEG data for {sub} already exists. Skipping download.")
            else:
                logging.info(f"Downloading {sub}...")
                huggingface_hub.snapshot_download(
                    THINGS_URL,
                    allow_patterns=f"raw-eeg/{sub}/*",
                    repo_type="dataset",
                    local_dir=data_path,
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subs", type=int, nargs="+", default=None)
    parser.add_argument(
        "-d", "--data_path", type=Path, default=Path("data/things-eeg2")
    )
    parser.add_argument("--raw_eeg_dir", type=str, default="raw-eeg")
    parser.add_argument("--img_dir", type=str, default="imgs")
    parser.add_argument(
        "-t",
        "--download_types",
        type=str,
        nargs="+",
        choices=["eeg", "imgs"],
        default=["eeg", "imgs"],
    )
    args = parser.parse_args()

    main(vars(args))
