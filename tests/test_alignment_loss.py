import torch

from brain_image.model.loss import InfoNCELoss


def test_infonce_is_symmetric_in_both_retrieval_directions():
    eeg = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    image = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = InfoNCELoss(init_temperature=1.0)

    actual, logits = loss(eeg, image)
    labels = torch.arange(2)
    expected = (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.T, labels)
    ) / 2

    assert torch.allclose(actual, expected)
