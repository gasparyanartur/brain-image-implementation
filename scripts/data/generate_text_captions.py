"""
generate_text_captions.py

Supports two modes controlled by ``use_batch`` in the Hydra config:

  Streaming (use_batch=false)
    Sends requests concurrently via the Responses API.
    Rate-limited; use max_concurrency=1 and max_retries to stay within quotas.

  Batch (use_batch=true)
    Uses the OpenAI Batch API (/v1/chat/completions) for 50 % cheaper requests
    and much higher throughput limits.  Results arrive within 24 h.

    1st run  – prepares chunked JSONL request files, uploads them, submits batch
               jobs, and saves state (batch IDs + custom_id→path map) to
               ``batch_state_path``.
    2nd run  – checks status of every pending batch; downloads + appends completed
               captions to ``caption_path``; removes completed batches from state.
"""
import asyncio
import base64
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import dotenv
import hydra
import tqdm
from omegaconf import DictConfig
from openai import AsyncOpenAI, OpenAI

from brain_image.configs import BaseConfig, GlobalConfig
from brain_image.data.datamodule import EEGDataModule
from brain_image.data.dataset.union import EEGDatasetConfigType
from brain_image.utils import setup


# ── Config ────────────────────────────────────────────────────────────────────

class TextCaptionConfig(BaseConfig):
    dataset: EEGDatasetConfigType
    splits: list[Literal["train", "test"]] = ["test"]
    caption_path: Path = Path("data/things-eeg2/captions/default.jsonl")
    model: str = "gpt-4o-mini"
    max_tokens: int = 512
    system_prompt: str = (
        "You are given an image. Generate a single, detailed caption for the image.\n"
        "You are given an object class label for the image, which is the object in the image that the caption should describe.\n"
        "The caption must be visually precise, descriptive, and neutral.\n"
        "Do not assume any prior context.\n"
        "Avoid storytelling, interpretation, or emotional language.\n"
        "Describe only what is clearly visible.\n"
        "Example: An [X] in a [Y]. The [X] is a [object] with [visual attributes] located at [position] of [Y] in the [local position] of the image. [Y] is an [object] with [visual attributes]."
    )
    # Streaming mode
    max_concurrency: int = 1
    max_retries: int = 5
    # Batch mode
    use_batch: bool = False
    batch_state_path: Path = Path("data/things-eeg2/captions/batch_state.json")
    batch_chunk_size: int = 500  # images per batch API call (keep files well under 200 MB)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_label(image_path: Path) -> str:
    return " ".join(image_path.stem.split("_")[:-1])


def load_existing_paths(caption_path: Path) -> set[str]:
    if not caption_path.exists():
        return set()
    existing: set[str] = set()
    with open(caption_path, "r") as f:
        for line in f:
            existing.add(json.loads(line)["path"])
    return existing


def path_to_custom_id(path: Path) -> str:
    """Short stable ID for a path; fits within the 64-char custom_id limit."""
    return hashlib.md5(str(path).encode()).hexdigest()[:32]


# ── Streaming mode ─────────────────────────────────────────────────────────────

async def _caption_one_streaming(
    client: AsyncOpenAI,
    image_path: Path,
    config: TextCaptionConfig,
) -> str:
    base64_image = encode_image(image_path)
    label = extract_label(image_path)
    logging.debug(f"Captioning {image_path}, label: {label}")

    response = await client.responses.create(
        model=config.model,
        instructions=config.system_prompt,
        max_output_tokens=config.max_tokens,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": label},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                ],
            }
        ],
    )
    logging.debug(f"Response status: {response.status}, output_text: {repr(response.output_text)}")
    return response.output_text


async def _run_streaming(
    client: AsyncOpenAI,
    img_paths: list[Path],
    config: TextCaptionConfig,
) -> None:
    config.caption_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_paths(config.caption_path)
    todo = [p for p in img_paths if str(p) not in existing]
    logging.info(f"Streaming: {len(todo)} remaining out of {len(img_paths)}.")

    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def process_one(path: Path) -> tuple[str, str]:
        async with semaphore:
            text = await _caption_one_streaming(client, path, config)
            return str(path), text

    tasks = [asyncio.create_task(process_one(p)) for p in todo]

    with open(config.caption_path, "a") as f, tqdm.tqdm(total=len(tasks), desc="Streaming captions") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                path_str, text = await coro
                json.dump({"path": path_str, "caption": text}, f)
                f.write("\n")
                f.flush()
            except Exception as e:
                logging.error(f"Streaming request failed: {e}")
            pbar.update(1)


# ── Batch mode ─────────────────────────────────────────────────────────────────

def _make_batch_request(image_path: Path, config: TextCaptionConfig) -> dict:
    """Build one line of the batch JSONL (Chat Completions format)."""
    return {
        "custom_id": path_to_custom_id(image_path),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": extract_label(image_path)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
        },
    }


def _load_batch_state(state_path: Path) -> list[dict]:
    if not state_path.exists():
        return []
    with open(state_path) as f:
        return json.load(f)["pending_batches"]


def _save_batch_state(state_path: Path, pending: list[dict]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump({"pending_batches": pending}, f, indent=2)


def _retrieve_completed_batches(
    client: OpenAI,
    pending: list[dict],
    config: TextCaptionConfig,
) -> list[dict]:
    """Poll each pending batch.  Write results for completed ones; return still-pending list."""
    still_pending: list[dict] = []

    for record in pending:
        batch_id = record["batch_id"]
        batch = client.batches.retrieve(batch_id)
        logging.info(f"Batch {batch_id}: status={batch.status} "
                     f"({batch.request_counts.completed}/{batch.request_counts.total} completed)")

        if batch.status == "completed":
            logging.info(f"Downloading results for batch {batch_id}…")
            id_to_path: dict[str, str] = record["id_to_path"]
            output_content = client.files.content(batch.output_file_id).text
            config.caption_path.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with open(config.caption_path, "a") as f:
                for line in output_content.splitlines():
                    result = json.loads(line)
                    custom_id = result["custom_id"]
                    if result.get("error"):
                        logging.warning(f"Request {custom_id} failed: {result['error']}")
                        continue
                    text = result["response"]["body"]["choices"][0]["message"]["content"]
                    img_path = id_to_path.get(custom_id, custom_id)
                    json.dump({"path": img_path, "caption": text}, f)
                    f.write("\n")
                    written += 1
            logging.info(f"Wrote {written} captions from batch {batch_id}.")

        elif batch.status in ("failed", "expired", "cancelled"):
            logging.error(f"Batch {batch_id} ended with status '{batch.status}'. Discarding.")

        else:  # in_progress, validating, finalizing, …
            still_pending.append(record)

    return still_pending


def _submit_batch_chunk(
    client: OpenAI,
    chunk: list[Path],
    config: TextCaptionConfig,
) -> dict:
    """Build the JSONL for *chunk*, upload it, create a batch, return a state record."""
    id_to_path = {path_to_custom_id(p): str(p) for p in chunk}

    logging.info(f"Preparing batch JSONL for {len(chunk)} images…")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for path in tqdm.tqdm(chunk, desc="Building batch JSONL"):
            tmp.write(json.dumps(_make_batch_request(path, config)) + "\n")
        tmp_path = Path(tmp.name)

    logging.info(f"Uploading {tmp_path} ({tmp_path.stat().st_size / 1e6:.1f} MB)…")
    with open(tmp_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    tmp_path.unlink()

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    logging.info(f"Submitted batch {batch.id} for {len(chunk)} images.")

    return {
        "batch_id": batch.id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "n_images": len(chunk),
        "id_to_path": id_to_path,
    }


def _run_batch(
    client: OpenAI,
    img_paths: list[Path],
    config: TextCaptionConfig,
) -> None:
    state_path = config.batch_state_path
    pending = _load_batch_state(state_path)

    # 1. Retrieve / download any completed batches.
    if pending:
        logging.info(f"Checking {len(pending)} pending batch(es)…")
        pending = _retrieve_completed_batches(client, pending, config)
        _save_batch_state(state_path, pending)

    # 2. Determine which images still need captioning.
    existing = load_existing_paths(config.caption_path)
    in_flight = {path for record in pending for path in record["id_to_path"].values()}
    todo = [p for p in img_paths if str(p) not in existing and str(p) not in in_flight]

    if not todo:
        if pending:
            logging.info(f"All images are either captioned or in {len(pending)} pending batch(es). Re-run to retrieve results.")
        else:
            logging.info("All images are already captioned.")
        return

    # 3. Submit new batches in chunks.
    logging.info(f"Submitting {len(todo)} uncaptioned images in chunks of {config.batch_chunk_size}…")
    for i in range(0, len(todo), config.batch_chunk_size):
        chunk = todo[i : i + config.batch_chunk_size]
        record = _submit_batch_chunk(client, chunk, config)
        pending.append(record)
        _save_batch_state(state_path, pending)

    logging.info(
        f"Submitted {len(pending)} active batch(es). "
        f"Re-run this script to check status and download results."
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def generate_text_captions(config: TextCaptionConfig) -> None:
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Add it to .env or the environment.")

    dataset_module = EEGDataModule(config.dataset)
    all_img_paths: list[Path] = []
    for split in config.splits:
        dataset = dataset_module.create_dataset(split, preload_cache=False, embeddings_to_compute_stats=[], compute_stats=False)
        paths = [Path(p) for p in dataset.get_image_paths()]
        logging.info(f"Split '{split}': {len(paths)} images.")
        all_img_paths.extend(paths)

    # Deduplicate (train + test may share no images, but be safe).
    seen: set[str] = set()
    unique_paths = [p for p in all_img_paths if not (str(p) in seen or seen.add(str(p)))]
    logging.info(f"Total unique images: {len(unique_paths)}")

    if config.use_batch:
        sync_client = OpenAI(api_key=api_key, max_retries=config.max_retries)
        _run_batch(sync_client, unique_paths, config)
    else:
        async_client = AsyncOpenAI(api_key=api_key, max_retries=config.max_retries)
        asyncio.run(_run_streaming(async_client, unique_paths, config))


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="generate_text_captions",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    setup()

    config = TextCaptionConfig.from_hydra_config(cfg)
    logging.info("Starting text caption generation")
    for key, value in config.model_dump(mode="json").items():
        logging.info(f"  {key}: {value}")

    generate_text_captions(config)


if __name__ == "__main__":
    main()
