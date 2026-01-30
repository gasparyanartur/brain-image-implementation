import datetime
import logging
from pathlib import Path
from typing import Literal, cast
from pytorch_lightning import LightningModule
import pytorch_lightning as pl


import torch
from torchvision.transforms import v2 as tv2
import itertools as it

import tqdm

from brain_image.augment import EEGAugmentationPipeline, ImageAugmentationPipeline
from brain_image.data.data import EEGDataModule, EEGDatasetConfig, batch_load_images
from brain_image.model.comm.comm import CoMM
from brain_image.model.comm.comm_loss import CoMMLoss
from brain_image.model.comm.input_adapters import FeaturesInputAdapter
from brain_image.model.comm.mmfusion import MMFusion
from brain_image.model.comm.utils import LinearWarmupCosineAnnealingLR, all_gather_batch_with_grad, set_weight_decay_per_param
from brain_image.model.eeg_encoder import create_eeg_encoder
from brain_image.model.eeg_encoder.eeg_encoder import EEGEncoderConfig
from brain_image.model.img_encoder import IMAGE_ENCODER, IMAGE_ENCODER_DIM, load_image_encoder
from brain_image.model.model import TrainingModule, TrainingModuleConfig
from brain_image.optimizer import OptimizerConfig, get_optimizer_options
from brain_image.utils import gather_records


class CommAlignmentConfig(TrainingModuleConfig):
    img_encoder: IMAGE_ENCODER = "unaligned_synclr_vitb16"
    eeg_encoder: EEGEncoderConfig
    
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

    modules_to_train: list[str] = ["eeg_encoder", "comm"]

    eeg_idx: int = 1
    img_idx: int = 0
    prototype_idx: int = 2



class CommAlignmentModel(TrainingModule):
    def __init__(
        self,
        config: CommAlignmentConfig,
        dataset_config: EEGDatasetConfig,
        preload_images: bool = True,
        cache_images: bool = True,
        eeg_encoder_path: Path | None = None,
        compile: bool = True,
        model_id: str | None = None,
        **kwargs,
    ):

        if isinstance(config, dict):
            config = CommAlignmentConfig(**config)

        super().__init__(config, **kwargs)
        self.automatic_optimization = (
            False  # Disable automatic optimization, we will handle it manually
        )


        self.config = config
        self.model_id = model_id


        logging.info(f"Seeding everything with seed: {self.config.seed}")
        pl.seed_everything(self.config.seed)

        device = self.device

        self.img_encoder = load_image_encoder(
            config.img_encoder,
            models_path=config.models_path,
            device=device,
            compile=False,
        )
        self.img_encoder.requires_grad_(False)

        eeg_encoder_path = eeg_encoder_path or self.config.eeg_encoder_path

        self.config.eeg_encoder.d_channels = dataset_config.num_channels
        self.config.eeg_encoder.d_time = dataset_config.time_length
        self.config.eeg_encoder.d_output = self.config.eeg_encoder.d_output or IMAGE_ENCODER_DIM[self.config.img_encoder]

        self.eeg_encoder = create_eeg_encoder(
            self.config.eeg_encoder,
            checkpoint_path=eeg_encoder_path,
        )

        encoders = []
        encoders.insert(config.img_idx, self.img_encoder)
        encoders.insert(config.eeg_idx, self.eeg_encoder)
        input_adapters = []
        input_adapters.insert(config.img_idx, FeaturesInputAdapter(IMAGE_ENCODER_DIM[self.config.img_encoder], self.config.embed_dim))
        input_adapters.insert(config.eeg_idx, FeaturesInputAdapter(self.config.eeg_encoder.d_output, self.config.embed_dim))

        self.comm = CoMM(
            encoder=MMFusion(
                encoders=encoders,
                input_adapters=input_adapters,
                embed_dim=self.config.embed_dim,
            ),
            projection=CoMM._build_mlp(self.config.embed_dim, self.config.embed_dim, self.config.proj_dim),
        )

        self.data_module = EEGDataModule(
            dataset_config, 
        )

        self.image_augmenter = ImageAugmentationPipeline()
        self.eeg_augmenter = EEGAugmentationPipeline()
        self.image_augmenter.requires_grad_(False)
        self.eeg_augmenter.requires_grad_(False)

        self.image_pipe = tv2.Compose(
            [
                tv2.Resize((224), interpolation=tv2.InterpolationMode.BICUBIC),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.comm_loss = CoMMLoss()
        self.images = {}

        self.cache_images = cache_images
        self.preload_images = preload_images

        if preload_images:
            for split in ["train", "val", "test"]:
                dataset = self.data_module.get_dataset(cast(Literal["train", "val", "test"], split))
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

        images = batch_load_images(image_paths, parallel=self.preload_images)   # Only parallel during preloading, otherwise dataloader throttles
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

        imgs = self.get_images(img_paths)

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
        - aug1_embed:
              <batch_size, n_modalities, embed_dim>
              Multimodal embedding of the first augmentation
        - aug2_embed:
              <batch_size, n_modalities, embed_dim>
              Multimodal embedding of the second augmentation
        - prototype:
              <int>
              Index of the multimodal representation within the batch.
              The modality at index `prototype` will be contrasted against all others.
        """
        output_dict = self.comm.forward([img_aug1, eeg_aug1], [img_aug2, eeg_aug2])
        return output_dict

    def forward(self, batch):
        eeg_aug1 = batch["eeg_aug1"]
        eeg_aug2 = batch["eeg_aug2"]
        img_aug1 = batch["img_aug1"]
        img_aug2 = batch["img_aug2"]

        comm_out = self.comm_forward(eeg_aug1, eeg_aug2, img_aug1, img_aug2)
        loss_dict = self.comm_loss(comm_out)

        return comm_out, loss_dict

    def training_step(self, batch, *args, **kwargs):
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

        _, loss_dict = self.forward(batch)

        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        

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
            lr = opt_option["lr_scheduler"].get_last_lr()[0] if opt_option["lr_scheduler"] is not None else -1

            self.log(name, lr)
        
        return loss_dict
    
    def validation_step(self, batch, *args, **kwargs):
        batch = self.prepare_batch(batch)

        with torch.no_grad():
            comm_out, loss_dict = self.forward(batch)
        
        z1 = comm_out["aug1_embed"]
        z2 = comm_out["aug2_embed"]
        ssl_acc = self.get_ssl_accuracy(z1, z2)
 
        self.log("val/loss", loss_dict["loss"], prog_bar=True)
        self.log(f"val/acc_eeg", ssl_acc[self.config.eeg_idx], prog_bar=True)
        self.log(f"val/acc_img", ssl_acc[self.config.img_idx], prog_bar=True)
        self.log(f"val/acc_proto", ssl_acc[self.config.prototype_idx], prog_bar=True)

        return loss_dict
    
    def test_step(self, batch, skip_log: bool = False, *args, **kwargs):
        batch = self.prepare_batch(batch)

        with torch.no_grad():
            comm_out, loss_dict = self.forward(batch)
        
        z1 = comm_out["aug1_embed"]
        z2 = comm_out["aug2_embed"]
        ssl_acc = self.get_ssl_accuracy(z1, z2)

        metrics = {
            "loss": loss_dict["loss"],
            "acc_eeg": ssl_acc[self.config.eeg_idx],
            "acc_img": ssl_acc[self.config.img_idx],
            "acc_proto": ssl_acc[self.config.prototype_idx],
        }

        if not skip_log:
            for k, v in metrics.items():
                self.log(f"test/{k}", v)

        return metrics
    
    @torch.no_grad()
    def run_full_test(self, loader: torch.utils.data.DataLoader, **kwargs):
        self.eval()

        metrics = []
        for batch in iter(loader):
            step_metrics = self.test_step(batch, skip_log=True, **kwargs)
            metrics.append(step_metrics)

        gathered_metrics = gather_records(metrics)
        mean = {k: torch.mean(v) for k, v in gathered_metrics.items()}
        std = {k: torch.std(v) for k, v in gathered_metrics.items()}

        return mean, std



        

    @torch.no_grad()
    def get_ssl_accuracy(self, z1, z2, prototype: int = -1, *args, **kwargs):
        from torch.nn import functional as F

        n = len(z1)
        device = z1[0].device

        z1 = [F.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [F.normalize(z, p=2, dim=-1) for z in z2]

        accuracies = []
        for i in range(n):
            sim = (z1[i] @ z2[prototype].T)  # dim [N, N] => the diag contains the correct pairs (i,j)
            pred = torch.argmax(sim, dim=1)
            accuracy = (pred == torch.arange(z1[i].size(0), device=device)).float().mean()
            accuracies.append(accuracy)

        return accuracies

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
                    self.comm_loss,
                    self.comm.head
                ],
                lr=self.config.comm_lr,
                min_lr=self.config.comm_min_lr,
                warmup_epochs=self.config.comm_warmup_epochs,
                delay_epochs=self.config.comm_delay_epochs,
                enabled=True,
                lr_scheduler="cosine_anneal",
            ),
        ]

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