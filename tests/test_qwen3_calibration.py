#!/usr/bin/env python3
"""CPU tests for the pairwise scalar closed-form solution."""

from __future__ import annotations

import torch

from exfold.qwen3.calibration import PairwiseScalarStats


def test_closed_form() -> None:
    stats = PairwiseScalarStats(3, ridge=1e-3)
    expert_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    outputs = torch.tensor(
        [
            [[2.0, 0.0], [1.0, 0.0]],
            [[0.0, 3.0], [0.0, 1.0]],
        ]
    )
    stats.update(expert_ids, outputs)
    coefficient, loss = stats.finalize()

    source_weight = outputs[:, 0].norm(dim=-1)
    expected = (
        source_weight
        * (outputs[:, 0] * outputs[:, 1]).sum(dim=-1)
    ).sum() / (
        (
            source_weight
            * outputs[:, 1].square().sum(dim=-1)
        ).sum()
        + 1e-3
    )
    assert torch.allclose(coefficient[0, 1], expected.float())
    assert loss[0, 1] < 1.0
    assert coefficient[2, 0] == 0
    assert loss[2, 0] == stats.default_loss
    assert torch.equal(coefficient.diagonal(), torch.ones(3))


def test_clipping_and_norms() -> None:
    stats = PairwiseScalarStats(2, ridge=1e-6, clip_min=-4.0, clip_max=4.0)
    expert_ids = torch.tensor([[0, 1]], dtype=torch.long)
    outputs = torch.tensor([[[10.0, 0.0], [1.0, 0.0]]])
    stats.update(expert_ids, outputs)
    coefficient, _ = stats.finalize()
    assert coefficient[0, 1] == 4.0
    assert torch.allclose(stats.expert_norms(), torch.tensor([10.0, 1.0]))


if __name__ == "__main__":
    test_closed_form()
    test_clipping_and_norms()
    print("PASS")
