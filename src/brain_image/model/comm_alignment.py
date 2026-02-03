import datetime
import json
import logging
from pathlib import Path
from typing import Literal, cast
from pytorch_lightning import LightningModule
import pytorch_lightning as pl


from torch.nn import functional as F
import torch
from torch import Tensor, nn
from torchvision.transforms import v2 as tv2
import itertools as it

import tqdm

from brain_image.augment import EEGAugmentationPipeline, ImageAugmentationPipeline
from brain_image.data.io import batch_load_images
from brain_image.data.datamodule import EEGDataModule
from brain_image.data.dataset.eeg_dataset import EEGDatasetConfig
from brain_image.data.dataset.union import resolve_dataset_config
from brain_image.metrics import get_retrieval_accuracy, get_top1_acc
from brain_image.model.comm.comm import CoMM
from brain_image.model.comm.comm_loss import CoMMLoss
from brain_image.model.comm.input_adapters import FeaturesInputAdapter
from brain_image.model.comm.mmfusion import MMFusion
from brain_image.model.comm.utils import (
    LinearWarmupCosineAnnealingLR,
    all_gather_batch_with_grad,
    set_weight_decay_per_param,
)
from brain_image.model.encoder.eeg_encoder.union import (
    EEGEncoderConfigType,
    create_eeg_encoder,
)
from brain_image.model.encoder.eeg_encoder.eeg_encoder import EEGEncoderConfig
from brain_image.model.encoder.img_encoder import (
    ImageEncoderName,
    IMAGE_ENCODER_DIM,
    load_image_encoder,
)
from brain_image.model.loss import CLIPLoss, CLIPSimLoss
from brain_image.model.model import TrainingModule, TrainingModuleConfig
from brain_image.optimizer import OptimizerConfig, get_optimizer_options
from brain_image.utils import gather_dataloader, gather_records, prep_batch_for_logs


class CommAlignmentConfig(TrainingModuleConfig):
    img_encoder: ImageEncoderName = "unaligned_synclr_vitb16"
    eeg_encoder: EEGEncoderConfigType

    embed_dim: int = 512
    proj_dim: int = 256

    img_size: int = 224
    models_path: Path = Path("models")
    eeg_encoder_path: Path | None = None

    encoder_lr: float = 3e-4
    encoder_min_lr: float = 1e-6
    encoder_warmup_epochs: int = 1
    encoder_delay_epochs: int = 0

    comm_lr: float = 3e-4
    comm_min_lr: float = 1e-6
    comm_warmup_epochs: int = 1
    comm_delay_epochs: int = 0

    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)

    train_img_encoder: bool = False
    eeg_scale: float = 1.0
    img_scale: float = 0.5

    modules_to_train: list[str] = ["eeg_encoder", "comm"]
    add_alignment: bool = False

    img_idx: int = 0
    eeg_idx: int = 1
    prototype_idx: int = 2


class CommAlignmentModel(TrainingModule):
    def __init__(
        self,
        config: CommAlignmentConfig,
        dataset_config: EEGDatasetConfig | dict,
        preload_images: bool = True,
        cache_images: bool = True,
        eeg_encoder_path: Path | None = None,
        compile: bool = True,
        model_id: str | None = None,
        **kwargs,
    ):

        if isinstance(config, dict):
            config = CommAlignmentConfig(**config)

        dataset_config = resolve_dataset_config(dataset_config)

        super().__init__(config, **kwargs)
        self.automatic_optimization = (
            False  # Disable automatic optimization, we will handle it manually
        )

        self.config = config
        self.model_id = model_id

        self.data_module = EEGDataModule(
            dataset_config,
        )

        logging.info(f"Seeding everything with seed: {self.config.seed}")
        pl.seed_everything(self.config.seed)

        device = self.device

        self.img_encoder = load_image_encoder(
            config.img_encoder,
            models_path=config.models_path,
            device=device,
            compile=False,
        )
        if not self.config.train_img_encoder:
            self.img_encoder.requires_grad_(False)

        eeg_encoder_path = eeg_encoder_path or self.config.eeg_encoder_path

        self.config.eeg_encoder.d_channels = dataset_config.num_channels
        self.config.eeg_encoder.d_time = dataset_config.time_length
        self.config.eeg_encoder.d_output = (
            self.config.eeg_encoder.d_output
            or IMAGE_ENCODER_DIM[self.config.img_encoder]
        )

        self.eeg_encoder = create_eeg_encoder(
            self.config.eeg_encoder,
            checkpoint_path=eeg_encoder_path,
        )

        self.config.eeg_encoder = self.eeg_encoder.config  # type: ignore

        self.metrics_on_pbar = "loss", "acc_eeg", "acc_img", "acc_proto"

        encoders = []
        encoders.insert(config.img_idx, self.img_encoder)
        encoders.insert(config.eeg_idx, self.eeg_encoder)
        input_adapters = []
        input_adapters.insert(
            config.img_idx,
            FeaturesInputAdapter(
                IMAGE_ENCODER_DIM[self.config.img_encoder], self.config.embed_dim
            ),
        )
        input_adapters.insert(
            config.eeg_idx,
            FeaturesInputAdapter(
                self.config.eeg_encoder.d_output, self.config.embed_dim
            ),
        )

        self.comm = CoMM(
            encoder=MMFusion(
                encoders=encoders,
                input_adapters=input_adapters,
                embed_dim=self.config.embed_dim,
            ),
            projection=CoMM._build_mlp(
                self.config.embed_dim, self.config.embed_dim, self.config.proj_dim
            ),
        )

        self.image_augmenter = ImageAugmentationPipeline()
        self.eeg_augmenter = EEGAugmentationPipeline(
            ampscale_prob=1,
            timeshift_prob=0.75,
            ampshift_prob=0.75,
            bandstop_prob=0,
            zeromask_prob=0,
            blur_prob=1,
        )
        self.image_augmenter.requires_grad_(False)
        self.eeg_augmenter.requires_grad_(False)

        self.image_pipe = tv2.Compose(
            [
                tv2.Resize((224), interpolation=tv2.InterpolationMode.BICUBIC),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )

        losses: dict[str, nn.Module] = {
            "comm_loss": CoMMLoss(),
        }
        if self.config.add_alignment:
            losses["align_loss"] = CLIPSimLoss()

        self.losses = nn.ModuleDict(
            losses
        )

        self.images = {}

        self.cache_images = cache_images
        self.preload_images = preload_images

        if preload_images:
            for split in ["train", "val", "test"]:
                dataset = self.data_module.get_dataset(
                    cast(Literal["train", "val", "test"], split)
                )
                img_paths = dataset.get_image_paths()

                for i in tqdm.tqdm(
                    range(0, len(img_paths), 32), desc=f"Preloading {split} images"
                ):
                    img_batch = img_paths[i : i + 32]
                    self.get_images(img_batch)

        if compile:
            self.comm.compile()

        self.save_hyperparameters(
            {
                "config": self.config.model_dump(mode="json"),
                "dataset_config": self.data_module.config.model_dump(mode="json"),
            },
        )

        self.compile: bool = compile
        self.atleast_one_training_step: bool = False

        logging.info(f"Finished initializing model")

    def get_name(self, timestamp: bool = False) -> str:
        name_components = []

        if timestamp:
            name_components.append(
                datetime.datetime.now().strftime("%y%m%d_%H%M%S"),
            )

        name_components.append(f"eeg_{self.config.eeg_encoder.eeg_encoder}")
        return "-".join(name_components)

    def get_images(self, image_paths: list[Path]) -> torch.Tensor:
        if self.cache_images and all(path in self.images for path in image_paths):
            return torch.stack([self.images[path] for path in image_paths])

        images = batch_load_images(
            image_paths, parallel=self.preload_images
        )  # Only parallel during preloading, otherwise dataloader throttles
        images = self.image_pipe(images)

        if self.cache_images:
            for path, image in zip(image_paths, images):
                self.images[path] = image

        return images

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    @torch.no_grad()
    def prepare_batch(self, batch):
        device = self.device

        eeg = batch["eeg_data"].to(device)
        img_paths = batch["img_path"]

        imgs = self.get_images(img_paths).to(device)

        img_aug1 = self.image_augmenter(imgs)
        img_aug2 = self.image_augmenter(imgs)
        eeg_aug1 = self.eeg_augmenter(eeg)
        eeg_aug2 = self.eeg_augmenter(eeg)

        return {
            "eeg": eeg,
            "img": imgs,
            "eeg_aug1": eeg_aug1,
            "eeg_aug2": eeg_aug2,
            "img_aug1": img_aug1,
            "img_aug2": img_aug2,
        }

    def comm_forward(self, eeg_aug1, eeg_aug2, img_aug1, img_aug2):
        """
        Outputs:
        - z1: <batch_size, n_modalities, embed_dim> Multimodal embedding of the first augmentation
        - z2: <batch_size, n_modalities, embed_dim> Multimodal embedding of the second augmentation
        """
        output_dict = self.comm.forward([img_aug1, eeg_aug1], [img_aug2, eeg_aug2])
        return output_dict["aug1_embed"], output_dict["aug2_embed"]

    def forward(self, batch):
        eeg_aug1 = batch["eeg_aug1"]
        eeg_aug2 = batch["eeg_aug2"]
        img_aug1 = batch["img_aug1"]
        img_aug2 = batch["img_aug2"]

        loss_dict: dict[str, Tensor] = {
        }

        z1, z2 = self.comm_forward(eeg_aug1, eeg_aug2, img_aug1, img_aug2)

        if "comm_loss" in self.losses:
            comm_loss = self.losses["comm_loss"]
            comm_loss_dict = comm_loss(z1, z2, self.config.prototype_idx)

            loss_dict.update({
                "loss_eeg_to_proto": comm_loss_dict[self.config.eeg_idx],
                "loss_img_to_proto": comm_loss_dict[self.config.img_idx],
                "loss_proto_to_proto": comm_loss_dict[self.config.prototype_idx],
            })


        if "align_loss" in self.losses:
            align_loss = self.losses["align_loss"]
            align_loss_dict = align_loss(
                z1[self.config.eeg_idx], z2[self.config.img_idx], norm=True
            )
            loss_dict.update({
                "loss_eeg_to_img": align_loss_dict["loss_e"],
                "loss_img_to_eeg": align_loss_dict["loss_i"],
            })

        
        loss_dict["loss"] = torch.stack(list(loss_dict.values())).sum()

        return loss_dict

    def training_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0, *args, **kwargs
    ):
        # Scaffolding
        self.atleast_one_training_step = True

        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        for opt in optimizers:
            opt.zero_grad()

        # Training Step
        batch = self.prepare_batch(batch)

        loss_dict = self.forward(batch)

        outputs = prep_batch_for_logs(loss_dict)
        for k, v in outputs.items():
            self.log(f"train/{k}", v, prog_bar=k in self.metrics_on_pbar)

        # Scaffolding
        self.manual_backward(loss_dict["loss"])

        for opt in optimizers:
            opt.step()

        for scheduler in schedulers:
            if scheduler is None:
                continue

            scheduler.step()  # type: ignore

        self.log("lr/epoch", self.trainer.current_epoch)
        self.log("lr/step", self.trainer.global_step)

        for opt_option in self.optimizer_options:
            name = "lr/" + opt_option["name"]
            lr = (
                opt_option["lr_scheduler"].get_last_lr()[0]
                if opt_option["lr_scheduler"] is not None
                else -1
            )

            self.log(name, lr)

        return loss_dict

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0, *args, **kwargs
    ):
        batch = self.prepare_batch(batch)

        with torch.no_grad():
            loss_dict = self.forward(batch)

        acc_dict = self.get_retrieval_accuracies(batch)
        set_dict = self.get_set_retrieval(batch)

        metrics = {
            **loss_dict,
            **acc_dict,
            **set_dict,
        }

        outputs = prep_batch_for_logs(metrics)

        for k, v in outputs.items():
            self.log(f"val/{k}", v, prog_bar=k in self.metrics_on_pbar)

        return loss_dict

    def test_step(
        self,
        batch,
        batch_idx: int = -1,
        dataloader_idx: int = 0,
        skip_log: bool = False,
        *args,
        **kwargs,
    ):
        batch = self.prepare_batch(batch)

        with torch.no_grad():
            loss_dict = self.forward(batch)

        acc_dict = self.get_retrieval_accuracies(batch)
        set_dict = self.get_set_retrieval(batch)

        metrics = {
            **loss_dict,
            **acc_dict,
            **set_dict,
        }

        outputs = prep_batch_for_logs(metrics)

        if not skip_log:
            for k, v in outputs.items():
                self.log(f"test/{k}", v)

        if self.atleast_one_training_step and (
            batch_idx == self.data_module.get_num_batches("test") - 1
        ):
            if self.log_dir is not None:
                with open(self.log_dir / "metrics.json", "w") as f:
                    json.dump(outputs, f)

        return outputs

    @torch.no_grad()
    def run_full_test(self, loader: torch.utils.data.DataLoader, **kwargs):
        self.eval()

        batch = gather_dataloader(loader)
        metrics = self.test_step(batch, skip_log=True, **kwargs)

        return metrics

    @torch.no_grad()
    def get_retrieval_accuracies(self, batch) -> dict[str, Tensor]:
        eeg = batch["eeg"]
        img = batch["img"]

        z_eeg = self.comm.encode_feature([eeg], [self.config.eeg_idx])
        z_img = self.comm.encode_feature([img], [self.config.img_idx])
        z_proto = self.comm.encode_feature([eeg, img], [self.config.eeg_idx, self.config.img_idx])

        z_eeg = F.normalize(z_eeg, p=2, dim=-1)
        z_img = F.normalize(z_img, p=2, dim=-1)
        z_proto = F.normalize(z_proto, p=2, dim=-1)

        acc_eeg_to_proto, acc_proto_to_eeg = get_retrieval_accuracy(z_eeg, z_proto, norm=False)
        acc_img_to_proto, acc_proto_to_img = get_retrieval_accuracy(z_img, z_proto, norm=False)
        acc_eeg_to_img, acc_img_to_eeg = get_retrieval_accuracy(z_eeg, z_img, norm=False)
        
        return {
            "acc_eeg_to_proto": acc_eeg_to_proto,
            "acc_img_to_proto": acc_img_to_proto,
            "acc_eeg_to_img": acc_eeg_to_img,
            "acc_proto_to_eeg": acc_proto_to_eeg,
            "acc_proto_to_img": acc_proto_to_img,
            "acc_img_to_eeg": acc_img_to_eeg,
        }
    
    @torch.no_grad()
    def get_set_retrieval(self, batch: dict) -> dict[str, Tensor]:
        eeg = batch["eeg"]
        img = batch["img"]

        n = eeg.size(0)

        z_eeg = self.comm.encoder.encode_single_mod(eeg, self.config.eeg_idx, project=True)
        z_img = self.comm.encoder.encode_single_mod(img, self.config.img_idx, project=True)
        
        z_eeg_fusion = self.comm.encoder.fusion_transformer([z_eeg])

        # For each eeg signal, we create a prototype with each image
        z_protos = torch.stack([
            self.comm.encoder.fusion_transformer(
                (z_eeg[i].unsqueeze(0).expand(n, -1, -1), z_img)
            ) 
            for i in range(n)
        ])  # <n_eeg, n_img, d>

        # z_proto <n_eeg, n_img, d>: 
        # eeg1 X img1, eeg1 X img2, ..., eeg1 X imgn
        # ...
        # eegn X img1, eegn X img2, ..., eegn X imgn
        # 
        # Of all the prototypes with eeg_i, the one with the highest dot product should be the one with correct image

        z_eeg_fusion = F.normalize(z_eeg_fusion, dim=-1) # <n, d>
        z_protos = F.normalize(z_protos, dim=-1) # <n, n, d>

        logits = torch.einsum("ed,eid->ei", z_eeg_fusion, z_protos) # <n_eeg, n_img>
        accuracy = get_top1_acc(logits, axis=1)
        
        return {
            "acc_set": accuracy
        }

    def configure_optimizers(self):
        logging.info("Configuring optimizers")

        optimizer_configs = [
            OptimizerConfig(
                name="eeg_encoder",
                modules=[self.eeg_encoder],
                lr=self.config.encoder_lr,
                min_lr=self.config.encoder_min_lr,
                warmup_epochs=self.config.encoder_warmup_epochs,
                delay_epochs=self.config.encoder_delay_epochs,
                enabled=self.eeg_encoder is not None,
                lr_scheduler="cosine_anneal",
            ),
            OptimizerConfig(
                name="comm",
                modules=[
                    self.comm.encoder.input_adapters,
                    self.comm.encoder.fusion_transformer,
                    self.comm.head,
                    *self.losses.values(),
                ],
                lr=self.config.comm_lr,
                min_lr=self.config.comm_min_lr,
                warmup_epochs=self.config.comm_warmup_epochs,
                delay_epochs=self.config.comm_delay_epochs,
                enabled=True,
                lr_scheduler="cosine_anneal",
            ),
        ]
        if self.config.train_img_encoder:
            raise NotImplementedError("Training the image encoder is not implemented yet")
            # If implemented, add this

        num_train_batches = self.data_module.get_num_batches("train")
        optimizer_options = get_optimizer_options(
            optimizer_configs,
            max_epochs=self.config.max_epochs,
            num_train_batches=num_train_batches,
            modules_to_optimize=self.config.modules_to_train,
            optimizer_params={
                "betas": self.config.betas,
                "weight_decay": self.config.weight_decay,
            },
        )

        self.optimizer_options = optimizer_options
        return optimizer_options
