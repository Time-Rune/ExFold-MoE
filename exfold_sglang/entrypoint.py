"""One-command DeepSeek-V4 SGLang launcher for Original and ExFold."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path
from typing import MutableMapping, Sequence


ROUTED_TOPK = 6
NUM_ROUTED_EXPERTS = 256


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected true/false for --enable-exfold, got {value!r}"
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else _parse_bool(value)


def _bounded_int(name: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be in [{minimum}, {maximum}], got {parsed}"
            )
        return parsed

    return parse


def _env_bounded_int(
    env_name: str, default: int, display_name: str, minimum: int, maximum: int
) -> int:
    return _bounded_int(display_name, minimum, maximum)(
        os.environ.get(env_name, str(default))
    )


def _default_calibration_path() -> Path:
    configured_home = Path(os.environ.get("EXFOLD_HOME", "/opt/exfold"))
    configured = configured_home / "artifacts/deepseek-v4-flash.pt"
    if configured.is_file():
        return configured
    return Path(__file__).resolve().parents[1] / "artifacts/deepseek-v4-flash.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exfold-sglang-serve",
        description="Launch DeepSeek-V4-Flash with Original or ExFold SGLang routing.",
    )
    parser.add_argument(
        "--enable-exfold",
        nargs="?",
        const=True,
        type=_parse_bool,
        default=_env_bool("EXFOLD_SGLANG_ENABLE", False),
        metavar="BOOL",
        help="enable ExFold (true/false; default: false)",
    )
    parser.add_argument(
        "--exfold-prefill-k",
        type=_bounded_int("prefill K", 1, ROUTED_TOPK),
        default=_env_bounded_int(
            "EXFOLD_PREFILL_TOPK", 4, "prefill K", 1, ROUTED_TOPK
        ),
        metavar="K",
        help="number of routed experts retained per prefill token (1-6; default: 4)",
    )
    parser.add_argument(
        "--exfold-decode-k",
        type=_bounded_int("decode K", 1, NUM_ROUTED_EXPERTS),
        default=_env_bounded_int(
            "EXFOLD_DECODE_EXPERT_BUDGET",
            128,
            "decode K",
            1,
            NUM_ROUTED_EXPERTS,
        ),
        metavar="K",
        help="maximum routed experts retained per decode batch (1-256; default: 128)",
    )
    parser.add_argument(
        "--exfold-print-command",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def configure_environment(
    *,
    enable_exfold: bool,
    prefill_k: int,
    decode_k: int,
    environ: MutableMapping[str, str],
    calibration_path: Path | None = None,
) -> None:
    environ["EXFOLD_SGLANG_H20_COMPAT"] = "1"
    environ["SGLANG_OPT_USE_TILELANG_MHC_PRE"] = "0"
    environ["SGLANG_OPT_USE_TILELANG_MHC_POST"] = "0"
    environ.pop("EXFOLD_ENABLE", None)
    environ.pop("EXFOLD_MODEL", None)

    if not enable_exfold:
        for name in (
            "EXFOLD_SGLANG_ENABLE",
            "EXFOLD_CALIBRATION_PATH",
            "EXFOLD_PREFILL_TOPK",
            "EXFOLD_DECODE_EXPERT_BUDGET",
        ):
            environ.pop(name, None)
        return

    artifact = calibration_path or _default_calibration_path()
    if not artifact.is_file():
        raise FileNotFoundError(f"ExFold calibration artifact not found: {artifact}")
    environ["EXFOLD_SGLANG_ENABLE"] = "1"
    environ["EXFOLD_CALIBRATION_PATH"] = str(artifact.resolve())
    environ["EXFOLD_PREFILL_TOPK"] = str(prefill_k)
    environ["EXFOLD_DECODE_EXPERT_BUDGET"] = str(decode_k)


def _has_option(args: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in args)


def build_sglang_command(passthrough: Sequence[str]) -> list[str]:
    args = list(passthrough)
    defaults: tuple[tuple[str, str | None], ...] = (
        ("--trust-remote-code", None),
        ("--model-path", "/workspace/models/DeepSeek-V4-Flash"),
        ("--served-model-name", "deepseek-v4-flash-0731"),
        ("--tp", "4"),
        ("--moe-runner-backend", "marlin"),
        ("--tool-call-parser", "deepseekv4"),
        ("--reasoning-parser", "deepseek-v4"),
        ("--host", "0.0.0.0"),
        ("--port", "8081"),
        ("--enable-metrics", None),
        ("--enable-cache-report", None),
        ("--mem-fraction-static", "0.80"),
    )
    command = ["sglang", "serve"]
    for option, value in defaults:
        if _has_option(args, option):
            continue
        command.append(option)
        if value is not None:
            command.append(value)
    command.extend(args)
    return command


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    options, passthrough = parser.parse_known_args(argv)
    configure_environment(
        enable_exfold=options.enable_exfold,
        prefill_k=options.exfold_prefill_k,
        decode_k=options.exfold_decode_k,
        environ=os.environ,
    )
    command = build_sglang_command(passthrough)
    mode = (
        f"ExFold P{options.exfold_prefill_k}+D{options.exfold_decode_k}"
        if options.enable_exfold
        else "Original"
    )
    print(f"[ExFold launcher] mode={mode}", flush=True)
    print(f"[ExFold launcher] exec={shlex.join(command)}", flush=True)
    if options.exfold_print_command:
        return
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
