import logging
import re
from pathlib import Path
from typing import Literal


def find_checkpoint_in_run(
    checkpoint_dir: Path,
    checkpoint_selection: Literal["last", "max", "min"],
    checkpoint_metric: str,
) -> Path:
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")

    cp_candidates = list(checkpoint_dir.glob("*.ckpt"))
    if len(cp_candidates) == 0:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    name_pattern = re.compile(
        rf".+epoch_(\d+)-{checkpoint_metric}_([0-9]*\.?[0-9]+)\.ckpt"
    )

    parsed_infos = []
    logging.info(f"Found {len(cp_candidates)} checkpoints in {checkpoint_dir}")
    for cp in cp_candidates:
        if cp.name == "last.ckpt":
            if checkpoint_selection == "last":
                return cp
            else:
                continue

        logging.debug(f"Parsing checkpoint {cp.name}")
        parsed_info = name_pattern.match(cp.name)
        if parsed_info is None:
            logging.warning(
                f"Checkpoint {cp} does not match the pattern {name_pattern} - Ignoring this one as candidate"
            )
            continue

        epoch = int(parsed_info.group(1))
        value = float(parsed_info.group(2))
        parsed_infos.append({"epoch": epoch, "value": value, "path": cp})

    for info in parsed_infos:
        logging.debug(
            f"Checkpoint {info['path']} has epoch {info['epoch']} and value {info['value']}"
        )

    match checkpoint_selection:
        case "last":
            sort_strategy = lambda info: -info["epoch"]
        case "max":
            sort_strategy = lambda info: -info["value"]
        case "min":
            sort_strategy = lambda info: info["value"]

    parsed_infos.sort(key=sort_strategy)
    return parsed_infos[0]["path"]