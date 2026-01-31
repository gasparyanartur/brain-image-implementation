from functools import lru_cache


import torch
from torch import Tensor


import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Literal, Sequence, cast

from brain_image.model.encoder.encoder import EncoderName


def _encode_tensor_keys(keys: tuple[str, ...]) -> str:
    return "/".join(keys)


@lru_cache(maxsize=1024 * 1024)
def _get_cached_tensor_path(cache_path: Path, keys: tuple[str, ...]) -> Path:
    encoded_path = _encode_tensor_keys(keys)
    full_path = cache_path / encoded_path
    full_path = full_path.with_suffix(".pt")
    return full_path


@lru_cache(maxsize=1024 * 1024)
def _load_cached_tensor_from_path(path: Path) -> Tensor:
    tensor = torch.load(path)
    return tensor


class TensorCache:
    def __init__(
        self,
        cache_path: Path = Path("tensorcache"),
    ):
        self.cache_path = cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def save(self, tensor: torch.Tensor, *keys: str):
        keys = tuple(keys)
        path = _get_cached_tensor_path(self.cache_path, keys)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)

    def batch_save(self, args: Iterable[tuple], parallel: bool = True) -> None:
        if parallel:
            with mp.Pool() as pool:
                pool.starmap(self.save, args)

        else:
            for arg in args:
                tensor, *keys = arg
                self.save(tensor, *cast(Sequence[str], keys))

    def batch_get(
        self, items: Iterable[Iterable[str]], parallel: bool = True
    ) -> torch.Tensor:
        def _get(key_list: list[str]) -> torch.Tensor:
            return self.get(*key_list)

        if parallel:
            with ThreadPoolExecutor() as executor:
                item_list = list(executor.map(_get, items))
        else:
            item_list = [self.get(*item) for item in items]

        return torch.stack(item_list, dim=0)

    def get(self, *keys: str) -> torch.Tensor:
        keys = tuple(keys)
        path = _get_cached_tensor_path(self.cache_path, keys)
        tensor = _load_cached_tensor_from_path(path)

        return tensor

    def get_latent(
        self,
        source_path: Path,
        model_name: EncoderName,
        split: Literal["train", "val", "test"],
    ) -> Tensor:
        split = "train" if split == "train" else "test"
        return self.get(str(source_path), model_name, split)