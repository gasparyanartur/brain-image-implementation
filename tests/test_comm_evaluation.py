from pathlib import Path

import torch

from scripts.evaluation.test_comm import write_metrics_csv
from brain_image.model.comm_alignment import generate_prior_latents, get_target_retrieval_metrics


class FakePrior:
    def __init__(self):
        self.conditioning_batches = []

    def generate(self, conditioning, **kwargs):
        self.conditioning_batches.append(conditioning.clone())
        return conditioning[:, :2]


def test_write_metrics_csv_serializes_scalar_metrics(tmp_path: Path):
    output_path = tmp_path / "test_metrics.csv"

    write_metrics_csv(
        output_path,
        {
            "acc_eeg_to_img": torch.tensor(0.5),
            "loss": 1.25,
        },
    )

    assert output_path.read_text().splitlines() == [
        "metric,value",
        "acc_eeg_to_img,0.5",
        "loss,1.25",
    ]


def test_generate_prior_latents_uses_eeg_conditioning_in_chunks():
    prior = FakePrior()
    eeg_latent = torch.arange(15, dtype=torch.float32).reshape(5, 3)

    generated = generate_prior_latents(
        prior,
        eeg_latent,
        num_steps=2,
        guidance_scale=3.0,
        batch_size=2,
        seed=42,
    )

    assert torch.equal(generated, eeg_latent[:, :2])
    assert [batch.shape[0] for batch in prior.conditioning_batches] == [2, 2, 1]


def test_target_retrieval_metrics_compare_each_representation_to_target():
    embeddings = torch.eye(4)

    metrics = get_target_retrieval_metrics(embeddings, embeddings, embeddings, embeddings)

    assert set(metrics) == {
        "acc_eeg_to_target_img",
        "acc_target_img_to_eeg",
        "acc_generated_img_to_target_img",
        "acc_target_img_to_generated_img",
        "acc_proto_to_target_img",
        "acc_target_img_to_proto",
    }
    assert all(value.item() == 1.0 for value in metrics.values())
