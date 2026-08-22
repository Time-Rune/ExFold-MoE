from pathlib import Path

import pytest

from exfold_sglang.entrypoint import (
    build_parser,
    build_sglang_command,
    configure_environment,
)


def test_original_unsets_exfold_environment() -> None:
    environ = {
        "EXFOLD_ENABLE": "1",
        "EXFOLD_MODEL": "deepseek-v4",
        "EXFOLD_SGLANG_ENABLE": "1",
        "EXFOLD_CALIBRATION_PATH": "/stale.pt",
        "EXFOLD_PREFILL_TOPK": "2",
        "EXFOLD_DECODE_EXPERT_BUDGET": "32",
    }
    configure_environment(
        enable_exfold=False,
        prefill_k=4,
        decode_k=128,
        environ=environ,
    )
    assert "EXFOLD_SGLANG_ENABLE" not in environ
    assert "EXFOLD_ENABLE" not in environ
    assert "EXFOLD_MODEL" not in environ
    assert environ["EXFOLD_SGLANG_H20_COMPAT"] == "1"


def test_exfold_environment_uses_requested_operating_point(tmp_path: Path) -> None:
    artifact = tmp_path / "calibration.pt"
    artifact.touch()
    environ: dict[str, str] = {}
    configure_environment(
        enable_exfold=True,
        prefill_k=3,
        decode_k=64,
        environ=environ,
        calibration_path=artifact,
    )
    assert environ["EXFOLD_SGLANG_ENABLE"] == "1"
    assert environ["EXFOLD_PREFILL_TOPK"] == "3"
    assert environ["EXFOLD_DECODE_EXPERT_BUDGET"] == "64"
    assert environ["EXFOLD_CALIBRATION_PATH"] == str(artifact.resolve())


@pytest.mark.parametrize(
    "args",
    [
        ["--exfold-prefill-k", "0"],
        ["--exfold-prefill-k", "7"],
        ["--exfold-decode-k", "0"],
        ["--exfold-decode-k", "257"],
    ],
)
def test_operating_point_ranges_are_validated(args: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(args)


def test_sglang_defaults_and_passthrough_override() -> None:
    command = build_sglang_command(["--port", "9000", "--chunked-prefill-size", "4096"])
    assert command[:2] == ["sglang", "serve"]
    assert command.count("--port") == 1
    assert command[command.index("--port") + 1] == "9000"
    assert command[-2:] == ["--chunked-prefill-size", "4096"]
