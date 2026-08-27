"""Regression tests for calibration artifact validation."""

from __future__ import annotations

import unittest

import torch

from exfold.artifacts import validate_deepseek_payload


def _payload(shape: tuple[int, int, int] = (2, 4, 4)) -> dict:
    loss = torch.full(shape, 0.2, dtype=torch.float32)
    loss.diagonal(dim1=-2, dim2=-1).zero_()
    return {
        "prefill_coeff": torch.ones(shape),
        "prefill_selection_loss": loss.clone(),
        "prefill_confidence_loss": loss.clone(),
        "decode_coeff": torch.ones(shape),
        "decode_loss": loss.clone(),
        "norms": torch.ones(shape[:2]),
    }


class DeepSeekArtifactValidationTest(unittest.TestCase):
    def test_accepts_scalar_frobenius_losses(self) -> None:
        shape = (2, 4, 4)
        health = validate_deepseek_payload(_payload(shape), shape=shape)
        self.assertAlmostEqual(
            health["prefill_selection_loss"]["median_best_loss"], 0.2
        )

    def test_rejects_near_unity_prefill_selection(self) -> None:
        shape = (2, 4, 4)
        payload = _payload(shape)
        loss = torch.full(shape, 0.9999)
        loss.diagonal(dim1=-2, dim2=-1).zero_()
        payload["prefill_selection_loss"] = loss
        with self.assertRaisesRegex(ValueError, "degenerate"):
            validate_deepseek_payload(payload, shape=shape)

    def test_allows_layers_without_foldable_pairs(self) -> None:
        shape = (2, 4, 4)
        payload = _payload(shape)
        for key in (
            "prefill_selection_loss",
            "prefill_confidence_loss",
            "decode_loss",
        ):
            payload[key][0].fill_(1.0e30)
        validate_deepseek_payload(payload, shape=shape)


if __name__ == "__main__":
    unittest.main()
