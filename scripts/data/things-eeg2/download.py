"""
Raw EEG Data taken from https://drive.google.com/drive/folders/1KnOcV38RthPcpZR2vtiSm0jtZ6p63RNt
Find more information here: https://osf.io/crxs4/overview"""

from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import Literal
import zipfile

import gdown

from brain_image.data.data import download_to_file
from brain_image.utils import setup_logging

# From the Things EEG2 dataset Google Drive folder
# https://drive.google.com/drive/folders/1KnOcV38RthPcpZR2vtiSm0jtZ6p63RNt
_THINGS_DATA_URL: dict[str, str] = {
    "sub-01": "1GCEoU_VFAnxwhX3wOXgzpcqdMkzK2j4d",
    "sub-02": "1fmzu5I_sP11zmARpG4up_inn8wbG4GQE",
    "sub-03": "1gKB-9AuueH9pfbT0hIKe0hstMuCbC9m4",
    "sub-04": "1hEJuZbw9EAXsdZk7G8Joif5V64-mrC3x",
    "sub-05": "19Q0s9oZdlxt1Ct0VuGVwCJVo8uXMnwuS",
    "sub-06": "1puOoIkZjWXCNWf3iIzYackAOFxmwqSH0",
    "sub-07": "1Z-FtP6kR02N-5G9p24mdfY12z9XUhUEB",
    "sub-08": "1mkOEFmoSyEZiIqa7fZ47Q00V0PDJxqjQ",
    "sub-09": "1NV9bL_M2jSlL8iZ2qI69azbxiW8Pptfb",
    "sub-10": "1f29e8A5Pr3Iu8el7aPkhJSRfd-rrAE0W",
}

_IMG_URL = "https://files.de-1.osf.io/v1/resources/y63gw/providers/osfstorage/?zip="


def main(args):
    setup_logging()
    logging.info(f"Downloading THINGS-EEG2 dataset with configs:")
    for k, v in args.items():
        logging.info(f"{k}: {v}")

    data_path = args["data_path"]
    data_path.mkdir(parents=True, exist_ok=True)

    download_types = set(args["download_types"])

    subs = list(args["subs"]) if args["subs"] else list(range(1, 11))
    sub_names = [f"sub-{sub:02d}" for sub in subs]

    if "eeg" in download_types:
        raw_eeg_path = data_path / args["raw_eeg_dir"]
        raw_eeg_path.mkdir(parents=True, exist_ok=True)
        for sub in sub_names:
            logging.info(f"Downloading {sub}...")

            raw_file_path = raw_eeg_path / f"{sub}.zip"
            extracted_path = raw_eeg_path / sub

            url = _THINGS_DATA_URL[sub]

            download_to_file(url, raw_file_path, verbose=True, skip_if_exists=True, backend="gdown")

            logging.info(f"Extracting file to {extracted_path}")
            with zipfile.ZipFile(raw_file_path, "r") as zf:
                zf.extractall(extracted_path.parent)

    if "imgs" in download_types:
        logging.info("Downloading images...")

        img_dir = data_path / args["img_dir"]
        imgs_zip = img_dir / "raw.zip"
        training_img_path_zip = img_dir / "training_images.zip"
        test_img_path_zip = img_dir / "test_images.zip"

        download_to_file(_IMG_URL, imgs_zip)

        img_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Extracting file to {img_dir}")
        with zipfile.ZipFile(imgs_zip, "r") as zf:
            zf.extractall(img_dir)

        with zipfile.ZipFile(training_img_path_zip, "r") as zf:
            zf.extractall(img_dir)

        with zipfile.ZipFile(test_img_path_zip, "r") as zf:
            zf.extractall(img_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
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
