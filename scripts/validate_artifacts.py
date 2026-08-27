#!/usr/bin/env python3
"""Validate both published calibration artifacts and print their hashes."""

from pathlib import Path

from exfold.artifacts import sha256, validate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "artifacts").glob("*.pt")):
        loss_health = validate(path)
        print(f"{sha256(path)}  {path.relative_to(root)}")
        for name, stats in loss_health.items():
            print(
                f"  {name}: median_best={stats['median_best_loss']:.6f}, "
                "worst_layer_median_best="
                f"{stats['worst_layer_median_best_loss']:.6f}"
            )


if __name__ == "__main__":
    main()
