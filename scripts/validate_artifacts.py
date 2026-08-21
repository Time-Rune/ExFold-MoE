#!/usr/bin/env python3
"""Validate both published calibration artifacts and print their hashes."""

from pathlib import Path

from exfold.artifacts import sha256, validate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "artifacts").glob("*.pt")):
        validate(path)
        print(f"{sha256(path)}  {path.relative_to(root)}")


if __name__ == "__main__":
    main()
