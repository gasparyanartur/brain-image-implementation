from concurrent.futures import ThreadPoolExecutor
import functools
import logging
from typing import Iterable, Literal
import gdown
import numpy as np
import requests
from torch import Tensor
import torchvision
import tqdm
from brain_image.data.data import Tensor, preprocess_eeg_data


import torch


from pathlib import Path


def load_eeg_data(
    eeg_path: Path,
) -> tuple[Tensor, Tensor, Tensor, list[str]]:
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG data not found: {eeg_path}")

    # Load the EEG data
    eeg_pickle = np.load(eeg_path, allow_pickle=True)
    raw_eeg = eeg_pickle["preprocessed_eeg_data"]
    channel_names = eeg_pickle["ch_names"]
    times = eeg_pickle["times"]

    raw_eeg = torch.from_numpy(raw_eeg).float()
    times = torch.from_numpy(times).float()
    idxs = torch.arange(len(raw_eeg))

    return raw_eeg, idxs, times, channel_names


def load_all_eeg_data(
    eeg_paths: list[Path], preprocess_configs: dict | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    preprocess_configs = preprocess_configs or {}

    all_eeg_data = []
    all_idxs = []
    all_times = None
    all_ch_names = []

    for eeg_path in eeg_paths:
        eeg_data, idxs, times, ch_names = load_eeg_data(eeg_path)
        eeg_data, idxs = preprocess_eeg_data(eeg_data, idxs, **preprocess_configs)

        all_eeg_data.append(eeg_data)

        if all_times is None:
            all_times = times

        if not all_ch_names:
            all_ch_names = ch_names

        all_idxs.append(idxs)

    if all_times is None:
        all_times = torch.tensor([])

    return torch.stack(all_eeg_data), torch.stack(all_idxs), all_times, all_ch_names


def get_image_paths(
    image_dir: Path,
    split: Literal["train", "test"],
    extensions: tuple[str, ...] = (".jpg", ".png", ".jpeg"),
) -> list[Path]:
    """Get all image paths from a directory."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    if split == "train":
        image_dir = image_dir / "training_images"

    elif split == "test":
        image_dir = image_dir / "test_images"

    img_paths = [
        img_path
        for concept_dir in sorted(image_dir.iterdir())
        for img_path in sorted(concept_dir.iterdir())
        if img_path.is_file() and img_path.suffix in extensions
    ]

    return img_paths


def download_to_file(
    url,
    file_path,
    verbose: bool = True,
    progress_bar: bool = True,
    chunk_size: int = 1024,
    skip_if_exists: bool = True,
    backend: Literal["gdown", "requests"] = "requests",
):
    def _log(s):
        if verbose:
            logging.info(s)

    if skip_if_exists and file_path.exists():
        _log(f"File {file_path} already exists, skipping download")
        return

    _log(f"Downloading file from {url} to {file_path} with backend {backend}")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    match backend:
        case "requests":
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            }
            response = requests.get(url, stream=True, headers=headers)
            total_size = int(response.headers.get("content-length", 0))
            written_size = 0

            _log(f"Using requests backend, saving in {file_path}")
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
        case "gdown":
            _log(f"Using gdown backend, saving in {file_path}")
            gdown.download(output=str(file_path), id=url)
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    if not file_path.exists():
        raise ValueError(f"Failed to download {url} to {file_path}")
    else:
        _log(f"Successfully downloaded {url} to {file_path}")


def load_image_from_path(path: Path | str, mode: str | None = None) -> Tensor:
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    match mode:
        case "rgb":
            mode_value = torchvision.io.ImageReadMode.RGB
        case None:
            mode_value = torchvision.io.ImageReadMode.UNCHANGED
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    img = torchvision.io.decode_image(str(path), mode=mode_value)
    return img


def batch_load_images(
    paths: Iterable[Path | str],
    parallel: bool = False,
    progressbar: bool = False,
    mode: str | None = None,
) -> Tensor:
    if parallel:
        with ThreadPoolExecutor() as pool:
            imgs = list(
                pool.map(
                    functools.partial(load_image_from_path, mode=mode),
                    paths,
                    timeout=10,
                )
            )
    else:
        imgs = [
            load_image_from_path(path, mode=mode)
            for path in tqdm.tqdm(
                list(paths), disable=not progressbar, desc="Loading images"
            )
        ]

    imgs = torch.stack(imgs, dim=0)
    return imgs