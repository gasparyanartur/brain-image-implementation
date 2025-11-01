from __future__ import annotations

from collections.abc import Callable
import functools
import logging
from typing import Literal, cast

import scipy
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2 as tv2
import torch
from torch import nn





@torch.no_grad()
def get_metric_pixcorr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    processor = tv2.Compose(
        [
            tv2.Resize(425, tv2.InterpolationMode.BILINEAR),
            tv2.ToDtype(torch.float, scale=True),
        ]
    )

    assert len(pred) == len(gt)
    B = pred.size(0)

    pred = pred.cpu()
    pred = processor(pred)
    pred = pred.reshape(B, -1)

    gt = gt.cpu()
    gt = processor(gt)
    gt = gt.reshape(B, -1)

    all_corrs = np.stack(
        [
            np.corrcoef(torch.stack((pred[i].double(), gt[i].double())))[0, 1]
            for i in range(B)
        ]
    )
    return torch.from_numpy(all_corrs).mean().float()


@torch.no_grad()
def get_metric_ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim

    processor = tv2.Compose(
        [
            tv2.ToDtype(torch.float, scale=True),
            tv2.Resize(425, tv2.InterpolationMode.BILINEAR),
        ]
    )

    assert len(pred) == len(gt)
    B = gt.shape[0]

    pred = pred.cpu()
    pred = processor(pred)
    pred = pred.permute(0, 2, 3, 1)
    pred = rgb2gray(pred)  # type: ignore

    gt = gt.cpu()
    gt = processor(gt)
    gt = gt.permute(0, 2, 3, 1)
    gt = rgb2gray(gt)  # type: ignore

    score = np.stack(
        [
            ssim(
                gt[i],
                pred[i],
                multichannel=True,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0,
            )
            for i in range(B)
        ]
    ).mean()
    return torch.tensor(score).float()


@torch.no_grad()
def get_metric_alex_score(
    pred: torch.Tensor, gt: torch.Tensor, feature: Literal["alexnet2", "alexnet5"]
) -> torch.Tensor:
    from torchvision.models import alexnet, AlexNet_Weights

    device = pred.device
    gt = gt.to(device)

    weights = AlexNet_Weights.IMAGENET1K_V1

    model = create_feature_extractor(
        alexnet(weights=weights), return_nodes=["features.4", "features.11"]
    ).to(device)
    model.eval().requires_grad_(False)

    feature_to_feature_layer = {"alexnet2": "features.4", "alexnet5": "features.11"}

    feature_layer = feature_to_feature_layer[feature]

    processor = tv2.Compose(
        [
            tv2.ToDtype(torch.float, scale=True),
            tv2.Resize(256, interpolation=tv2.InterpolationMode.BILINEAR),
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pred = processor(pred)
    gt = processor(gt)

    score = _two_way_identification(pred, gt, model, feature_layer=feature_layer)
    return score


@torch.no_grad()
def get_metric_inceptionv3_score(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    from torchvision.models import inception_v3, Inception_V3_Weights

    device = pred.device
    gt = gt.to(device)

    weights = Inception_V3_Weights.DEFAULT
    model = create_feature_extractor(
        inception_v3(weights=weights), return_nodes=["avgpool"]
    ).to(device)
    model.eval().requires_grad_(False)

    feature_layer = "avgpool"

    processor = tv2.Compose(
        [
            tv2.ToDtype(torch.float, scale=True),
            tv2.Resize(342, interpolation=tv2.InterpolationMode.BILINEAR),
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pred = processor(pred)
    gt = processor(gt)

    score = _two_way_identification(pred, gt, model, feature_layer=feature_layer)
    return score


@torch.no_grad()
def get_metric_clip(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    import clip

    device = pred.device
    gt = gt.to(device)

    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval().requires_grad_(False)
    model = clip_model.encode_image

    processor = tv2.Compose(
        [
            tv2.ToDtype(torch.float, scale=True),
            tv2.Resize(224, interpolation=tv2.InterpolationMode.BILINEAR),
            tv2.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    pred = processor(pred)
    gt = processor(gt)

    score = _two_way_identification(
        pred, gt, cast(nn.Module, model), feature_layer=None
    )
    return score


@torch.no_grad()
def get_metric_efficientnet(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    import scipy

    device = pred.device
    gt = gt.to(device)

    weights = EfficientNet_B1_Weights.DEFAULT
    model = create_feature_extractor(
        efficientnet_b1(weights=weights), return_nodes=["avgpool"]
    ).to(device)
    model.eval().requires_grad_(False)

    processor = tv2.Compose(
        [
            tv2.ToDtype(torch.float, scale=True),
            tv2.Resize(255, interpolation=tv2.InterpolationMode.BILINEAR),
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pred = processor(pred)
    gt = processor(gt)

    score = _correlation_distance(pred, gt, model, feature_layer="avgpool")
    return score


@torch.no_grad()
def get_metric_swav(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    from torchvision.models.resnet import resnet50 as _resnet50

    def load_swav_resnet50():
        # Using torch.hub.load is broken on the swav repo, so we load the weights manually
        # From https://github.com/facebookresearch/swav/blob/main/hubconf.py
        swav_model = _resnet50(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
            map_location="cpu",
        )
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        swav_model.load_state_dict(state_dict, strict=False)
        return swav_model

    device = pred.device
    gt = gt.to(device)

    swav_model = load_swav_resnet50()
    model = create_feature_extractor(swav_model, return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)

    processor = tv2.Compose(
        [
            tv2.ToDtype(torch.float, scale=True),
            tv2.Resize(224, interpolation=tv2.InterpolationMode.BILINEAR),
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pred = processor(pred)
    gt = processor(gt)

    score = _correlation_distance(pred, gt, model, feature_layer="avgpool")
    return score


type MetricType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
type MetricName = Literal[
    "pixcorr",
    "ssim",
    "alex2",
    "alex5",
    "inceptionv3",
    "clip",
    "efficientnet",
    "swav",
]
METRIC_LOOKUP: dict[MetricName, MetricType] = {
    "pixcorr": get_metric_pixcorr,
    "ssim": get_metric_ssim,
    "alex2": functools.partial(get_metric_alex_score, feature="alexnet2"),
    "alex5": functools.partial(get_metric_alex_score, feature="alexnet5"),
    "inceptionv3": get_metric_inceptionv3_score,
    "clip": get_metric_clip,
    "efficientnet": get_metric_efficientnet,
    "swav": get_metric_swav,
}
METRIC_BIGGER_IS_BETTER: dict[MetricName, bool] = {
    "pixcorr": True,
    "ssim": True,
    "alex2": True,
    "alex5": True,
    "inceptionv3": True,
    "clip": True,
    "efficientnet": False,
    "swav": False,
}




def evaluate_metrics(preds: torch.Tensor, gts: torch.Tensor, metrics: list[MetricName]) -> dict[MetricName, torch.Tensor]:
    logging.info(f"Evaluating metrics: {metrics}")

    values = {}

    for metric_name in metrics:
        if metric_name not in METRIC_LOOKUP:
            raise ValueError(f"Unknown metric: {metric_name}")

        metric = METRIC_LOOKUP[metric_name]
        score = metric(preds, gts)

        values[metric_name] = score
        logging.info(f"\t{metric_name}: {score:.4f}")


    return values


@torch.no_grad()
def _two_way_identification(
    pred: torch.Tensor,
    gt: torch.Tensor,
    model: nn.Module,
    feature_layer: str | None = None,
) -> torch.Tensor:
    pred_f = model(pred)
    gt_f = model(gt)
    B = pred.size(0)

    if feature_layer is not None:
        pred_f = pred_f[feature_layer]
        gt_f = gt_f[feature_layer]

    pred_f = pred_f.flatten(1).cpu()
    gt_f = gt_f.flatten(1).cpu()

    r = np.corrcoef(pred_f, gt_f)
    r = r[:B, B:]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    perf = np.mean(success_cnt) / (B - 1)
    return torch.tensor(perf).float()


@torch.no_grad()
def _correlation_distance(
    pred: torch.Tensor,
    gt: torch.Tensor,
    model: nn.Module,
    feature_layer: str | None = None,
) -> torch.Tensor:
    pred_f = model(pred)
    gt_f = model(gt)
    B = pred.size(0)

    if feature_layer is not None:
        pred_f = pred_f[feature_layer]
        gt_f = gt_f[feature_layer]

    pred_f = pred_f.flatten(1).cpu().numpy()
    gt_f = gt_f.flatten(1).cpu().numpy()

    dist = np.array(
        [scipy.spatial.distance.correlation(gt_f[i], pred_f[i]) for i in range(B)]
    ).mean()
    return torch.tensor(dist).float()



@torch.compile()
@torch.no_grad()
def get_top1_acc(logits: torch.Tensor, axis=1) -> torch.Tensor:
    indexes = torch.arange(len(logits), device=logits.device)
    top1 = logits.topk(1, dim=axis).indices.flatten()
    top1_acc = (top1 == indexes).float().mean()
    return top1_acc
