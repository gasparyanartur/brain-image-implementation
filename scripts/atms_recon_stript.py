import itertools as it
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

import torch
import tqdm
import matplotlib.pyplot as plt
import datetime

from data.data import TensorCache
from brain_image.model.eeg_encoder import atms

ALIGN_LOSS_SCALE = 1
REGRESS_LOSS_SCALE = 9


class EEGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: Literal["train", "test"],
        data_path: Path = Path("data/things-eeg2"),
        sub: int = 8,
        img_encoder: str = "clip_vith14",
    ):
        self.split = split
        self.data_path = data_path
        self.sub = sub
        self.img_encoder = img_encoder

        self._tensorcache = TensorCache(memory_cache_size=128000)
        self.data = torch.load(data_path / "prepared" / f"sub-{sub:02}" / f"{split}.pt")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        img_path = sample["img_path"]

        return {
            "img_path": img_path,
            "eeg_data": sample["eeg"],
            "img_embedding": self._tensorcache.get(
                img_path, self.img_encoder, f"{self.split}.pt"
            ),
            "idx": sample["idx"],
            "sub": sample["sub"],
        }


@torch.compile()
def get_logits(eeg_features: torch.Tensor, img_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    eeg_features = torch.nn.functional.normalize(eeg_features)
    img_features = torch.nn.functional.normalize(img_features)
    logit_scale = logit_scale.exp()

    logits = logit_scale * eeg_features @ img_features.T
    return logits


@torch.compile()
def get_clip_loss(logits: torch.Tensor, loss_func: torch.nn.Module) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)

    loss_eeg = loss_func(logits, labels)
    loss_img = loss_func(logits.T, labels)

    return (loss_eeg + loss_img) * 0.5


def run_step(
    model: atms.AtmsEEGEncoder, loss_func: torch.nn.Module, batch: dict, device: torch.device
) -> dict[str, torch.Tensor]:
    eeg_data = batch["eeg_data"].to(device)
    img_embedding = batch["img_embedding"].to(device)
    sub = batch["sub"].to(device)

    eeg_embeddings = model(eeg_data, sub)
    logits = get_logits(eeg_embeddings, img_embedding, model.logit_scale)

    align_loss = get_clip_loss(logits, loss_func) * ALIGN_LOSS_SCALE
    regress_loss = torch.nn.functional.mse_loss(eeg_embeddings, torch.nn.functional.normalize(img_embedding, dim=-1, p=2)) * REGRESS_LOSS_SCALE

    return {
        "align_loss": align_loss,
        "regress_loss": regress_loss,
        "loss": align_loss + regress_loss,
    }


def _add_or_create(d: dict, k: str, v: Any) -> None:
    if k not in d:
        d[k] = []
    d[k].append(v)


def _populate_dict_with_record(main_d: dict, sample_d: dict) -> None:
    for k, v in sample_d.items():
        _add_or_create(main_d, k, v)


def run_epoch(
    model: atms.AtmsEEGEncoder,
    loss_func: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    split: Literal["train", "test"],
    progress_bar: tqdm.tqdm | None = None,
):
    loss_values = {}
    if split == "train":
        model.train()
    else:
        model.eval()

    for batch in train_loader:
        if split == "train":
            optimizer.zero_grad()
            losses = run_step(model, loss_func, batch, device)
            losses["loss"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = run_step(model, loss_func, batch, device)

        _populate_dict_with_record(
            loss_values, {k: v.item() for k, v in losses.items()}
        )

        if progress_bar is not None:
            progress_bar.update(1)

    return {k: sum(v) / len(v) for k, v in loss_values.items()}


def eval_model(model: atms.AtmsEEGEncoder, test_loader, device):
    model.eval()

    metrics = {}
    for batch in test_loader:
        eeg_data = batch["eeg_data"].to(device)
        img_embedding = batch["img_embedding"].to(device)
        sub = batch["sub"].to(device)

        eeg_embeddings = model(eeg_data, sub)
        logits = get_logits(eeg_embeddings, img_embedding, model.logit_scale)

        top1_acc = (
            (logits.argmax(dim=-1) == torch.arange(len(logits), device=logits.device))
            .float()
            .mean()
        )
        _add_or_create(metrics, "top1_acc", top1_acc.item())

    return metrics


def train_eeg_encoder(
    model: atms.AtmsEEGEncoder,
    loss_func,
    train_loader,
    test_loader,
    optimizer,
    device,
    num_epochs,
    checkpoint_dir,
    checkpoint_interval,
    plot_dir,
    resume_checkpoint: Path | None = None,
):
    model.to(device)
    all_train_losses = {}
    all_test_losses = {}
    all_test_metrics = {}
    epoch = 0

    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        all_train_losses = checkpoint["train_losses"]
        all_test_losses = checkpoint["test_losses"]
        all_test_metrics = checkpoint["test_metrics"]
        epoch = checkpoint["epoch"] + 1

    with tqdm.tqdm(
        initial=epoch,
        total=num_epochs * len(train_loader), desc="Training model", unit="steps"
    ) as pbar:
        print(f"Starting model training for {num_epochs} epochs...")
        for epoch in range(epoch, num_epochs):
            train_losses = run_epoch(
                model,
                loss_func,
                train_loader,
                optimizer,
                device,
                "train",
                progress_bar=pbar,
            )
            test_losses = run_epoch(
                model, loss_func, test_loader, optimizer, device, "test"
            )
            test_metrics = eval_model(model, test_loader, device)

            _populate_dict_with_record(all_train_losses, train_losses)
            _populate_dict_with_record(all_test_losses, test_losses)
            _populate_dict_with_record(all_test_metrics, test_metrics)

            pbar.set_description(
                f"Training model | train_loss: {train_losses['loss']:.4f} | test_loss: {test_losses['loss']:.4f}"
            )
            pbar.update(1)


            for title, values in zip(
                ("train_losses", "test_losses", "metrics"),
                (all_train_losses, all_test_losses, all_test_metrics),
            ):
                plt.figure()
                plt.title(title)
                for label, value in values.items():
                    plt.plot(value, label=label)
                plt.legend()
                plt.savefig(plot_dir / f"{title}.png")


            if (epoch + 1) % checkpoint_interval == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_losses": all_train_losses,
                        "test_losses": all_test_losses,
                        "test_metrics": all_test_metrics,
                    },
                    checkpoint_dir
                    / f"epoch_{epoch}-val_loss_{test_losses['loss']:.4f}.pt",
                )


    return all_train_losses, all_test_losses, all_test_metrics


def main(args: dict):
    print(f"Training model with args:")
    for k, v in args.items():
        print(f"\t{k} - {v}")

    sub: int = args["sub"]
    batch_size: int = args["batch_size"]
    learning_rate: float = args["learning_rate"]
    weight_decay: float = args["weight_decay"]
    num_epochs: int = args["num_epochs"]
    checkpoint_interval: int = args["checkpoint_interval"]
    base_output_dir: Path = args["base_output_dir"]
    device: torch.device = torch.device(args["device"])
    resume_checkpoint: Path | None = args["resume_checkpoint"]

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = base_output_dir / "eeg_training" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")

    model = atms.AtmsEEGEncoder()
    model.compile()

    loss_func = torch.nn.CrossEntropyLoss()
    loss_func.compile()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_dataset = EEGDataset(split="train", sub=sub)
    test_dataset = EEGDataset(split="test", sub=sub)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=12,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
        persistent_workers=False,
        num_workers=0,
    )

    # preload cache
    for _ in tqdm.tqdm(
        it.chain(iter(train_dataset), iter(test_dataset)),
        total=len(train_dataset) + len(test_dataset),
        desc="Preloading cache",
    ):
        pass

    train_losses, test_losses, test_metrics = train_eeg_encoder(
        model,
        loss_func,
        train_loader,
        test_loader,
        optimizer,
        device,
        num_epochs,
        checkpoint_dir,
        checkpoint_interval,
        plot_dir,
        resume_checkpoint
    )

    print("Finished training model!")
    for names, values in zip(("train_losses", "test_losses", "test_metrics"), (train_losses, test_losses, test_metrics)):
        print(f"{names}:")
        for k, v in values.items():
            print(f"\t{k}: {v[-1]:.4f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sub", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument("--base_output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_checkpoint", type=Path, default=None)

    args = vars(parser.parse_args())
    main(args)
