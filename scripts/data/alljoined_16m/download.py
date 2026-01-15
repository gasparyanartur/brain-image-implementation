from pathlib import Path
import argparse
import logging
import zipfile

import huggingface_hub

from brain_image.utils import setup_logging
import tempfile


STIM_ORDER_REVISION="bd55487200c4c712dac9be2ee6fc8f593095b454"


def main(args):

    setup_logging()
    logging.info(f"Downloaded Alljoined 1.6M dataset with configs:")
    for k, v in args.items():
        logging.info(f"{k}: {v}")

    data_path = Path(args["data_path"])
    data_path.mkdir(parents=True, exist_ok=True)

    if args["subs"] is None:
        args["subs"] = list(range(1, 21))

    if "stim" in args["download_types"]:
        logging.info("Downloading stimulus data...")

        stim_path = data_path / args["stim_dir"]
        #stim_path.mkdir(parents=True, exist_ok=True)

        stim_zip_path = data_path / "stimuli.zip"
        if not stim_zip_path.exists():
            huggingface_hub.snapshot_download("Alljoined/Alljoined-1.6M", allow_patterns="stimuli*", repo_type="dataset", local_dir=data_path)
        else:
            logging.info(f"Stimulus zip file already exists at {stim_zip_path}. Skipping download.")

        assert stim_zip_path.exists(), f"Stimulus zip file not found at {stim_zip_path} after download."

        with zipfile.ZipFile(stim_zip_path, "r") as zip_ref:
            logging.info(f"Extracting stimulus data to {stim_path}")
            zip_ref.extractall(stim_path)

    if "eeg" in args["download_types"]:
        logging.info("Downloading EEG data...")
        raw_eeg_path = data_path / args["raw_eeg_dir"]

        raw_eeg_path.mkdir(parents=True, exist_ok=True)
        for sub in args["subs"]:
            logging.info(f"Downloading sub-{sub:02}...")
            huggingface_hub.snapshot_download("Alljoined/Alljoined-1.6M", allow_patterns=f"raw_eeg/sub-{sub:02}/*", repo_type="dataset", local_dir=raw_eeg_path)

    if "stim-order" in args["download_types"] or "eeg" in args["download_types"]:
        logging.info("Downloading stim-order files...")
        # The stim-order file is not downloaded with the EEG data, so we need to download it separately
        
        with tempfile.TemporaryDirectory(dir=data_path) as tmp_dir:
            tmp_path = Path(tmp_dir)
            for sub in args["subs"]:
                huggingface_hub.snapshot_download("Alljoined/Alljoined-1.6M", allow_patterns=f"*{sub:02}*/stim_order.parquet", repo_type="dataset", local_dir=tmp_path, revision=STIM_ORDER_REVISION)
        
                # Move the stim_order.parquet file to the correct location
                _found_path = list(tmp_path.rglob(f"*/sub-{sub:02}*/stim_order.parquet"))
                assert len(_found_path) == 1, f"Expected exactly one stim_order.parquet file for sub-{sub:02}, found {len(_found_path)}"
                
                _found_path[0].rename(data_path / args["raw_eeg_dir"] / f"sub-{sub:02}" / "stim_order.parquet")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 

    parser.add_argument("-s", "--subs", type=int, nargs="+", default=None)
    parser.add_argument("-d", "--data_path", type=str, default="data/alljoined-1.6m")
    parser.add_argument("--raw_eeg_dir", type=str, default="raw_eeg")
    parser.add_argument("--stim_dir", type=str, default="stimuli")

    parser.add_argument("-t", "--download_types", nargs='+', choices=["eeg", "stim", "stim-order"], default=["eeg", "stim", "stim-order"])

    args = parser.parse_args()

    main(vars(args))