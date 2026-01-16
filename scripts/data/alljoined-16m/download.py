from pathlib import Path
import argparse
import logging
from typing import cast
import zipfile

import huggingface_hub

from brain_image.utils import setup_logging
import tempfile


ALLJOINED_URL = "Alljoined/Alljoined-1.6M"
STIM_ORDER_REVISION = "bd55487200c4c712dac9be2ee6fc8f593095b454"


def main(args):
    setup_logging()
    logging.info(f"Downloaded Alljoined 1.6M dataset with configs:")
    for k, v in args.items():
        logging.info(f"{k}: {v}")

    data_path = args["data_path"]
    data_path.mkdir(parents=True, exist_ok=True)

    subs = list(args["subs"]) or list(range(1, 21))
    sub_names = [f"sub-{sub:02d}" for sub in subs]

    download_types = set(args["download_types"])

    if "stim" in download_types:
        logging.info("Downloading stimulus data...")
        stim_path = data_path / args["stim_dir"]

        stim_zip_path = data_path / "stimuli.zip"
        if not stim_zip_path.exists():
            huggingface_hub.snapshot_download(
                ALLJOINED_URL,
                allow_patterns="stimuli*",
                repo_type="dataset",
                local_dir=data_path,
            )
        else:
            logging.info(
                f"Stimulus zip file already exists at {stim_zip_path}. Skipping download."
            )

        assert (
            stim_zip_path.exists()
        ), f"Stimulus zip file not found at {stim_zip_path} after download."

        with zipfile.ZipFile(stim_zip_path, "r") as zip_ref:
            logging.info(f"Extracting stimulus data to {stim_path}")
            zip_ref.extractall(stim_path)

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
                    ALLJOINED_URL,
                    allow_patterns=f"raw_eeg/{sub}/*",
                    repo_type="dataset",
                    local_dir=data_path,
                )

            # The stim-order file is not downloaded with the EEG data, so we need to download it separately
            stim_order_path = sub_path / "stim_order.parquet"
            if stim_order_path.exists():
                logging.info(
                    f"Stim-order file for {sub} already exists. Skipping download."
                )

            else:
                logging.info("Downloading stim-order files...")

                with tempfile.TemporaryDirectory(dir=data_path) as tmp_dir:
                    stim_path = Path(tmp_dir)
                    huggingface_hub.snapshot_download(
                        ALLJOINED_URL,
                        allow_patterns=f"*/{sub}*/stim_order.parquet",
                        repo_type="dataset",
                        local_dir=stim_path,
                        revision=STIM_ORDER_REVISION,
                    )

                    # Move the stim_order.parquet file to the correct location
                    _found_path = list(stim_path.rglob(f"*/{sub}*/stim_order.parquet"))
                    assert (
                        len(_found_path) == 1
                    ), f"Expected exactly one stim_order.parquet file for {sub}, found {len(_found_path)}"

                    _found_path = cast(Path, _found_path[0])
                    _found_path.rename(stim_order_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--subs", type=int, nargs="+", default=None)
    parser.add_argument("-d", "--data_path", type=Path, default="data/alljoined-1.6m")
    parser.add_argument("--raw_eeg_dir", type=str, default="raw-eeg")
    parser.add_argument("--stim_dir", type=str, default="stimuli")

    parser.add_argument(
        "-t",
        "--download_types",
        nargs="+",
        choices=["eeg", "stim", "stim-order"],
        default=["eeg", "stim", "stim-order"],
    )

    args = parser.parse_args()

    main(vars(args))
