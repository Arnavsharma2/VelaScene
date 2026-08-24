#!/usr/bin/env python3
"""Run lightweight checks before starting a VelaScene training or inference job."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)")


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    detail: str


def requirement_names(lines: Iterable[str]) -> list[str]:
    """Extract distribution names from a pip requirements file."""
    names: list[str] = []
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "git+", "http://", "https://")):
            continue
        line = line.split(";", 1)[0].strip()
        match = REQUIREMENT_NAME.match(line)
        if match:
            names.append(match.group(1))
    return names


def check_python() -> Check:
    version = sys.version_info
    rendered = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) == (3, 10):
        return Check("PASS", "Python", f"{rendered} (the tested version)")
    if (version.major, version.minor) >= (3, 10):
        return Check("WARN", "Python", f"{rendered}; VelaScene is tested with Python 3.10")
    return Check("FAIL", "Python", f"{rendered}; Python 3.10 or newer is required")


def check_command(command: str) -> Check:
    location = shutil.which(command)
    if location:
        return Check("PASS", command, location)
    return Check("FAIL", command, "command was not found on PATH")


def check_requirements(path: Path) -> Check:
    if not path.is_file():
        return Check("FAIL", "Dependencies", f"requirements file not found: {path}")

    names = requirement_names(path.read_text(encoding="utf-8").splitlines())
    missing: list[str] = []
    for name in names:
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)

    if missing:
        return Check(
            "FAIL",
            "Dependencies",
            f"{len(missing)} missing from {path.name}: {', '.join(missing)}",
        )
    return Check("PASS", "Dependencies", f"all {len(names)} packages in {path.name} are installed")


def check_path(label: str, path: Path, *, directory: bool) -> Check:
    expected = "directory" if directory else "file"
    valid = path.is_dir() if directory else path.is_file()
    if valid:
        return Check("PASS", label, str(path.resolve()))
    return Check("FAIL", label, f"{expected} not found: {path}")


def check_accelerator(requested: str) -> tuple[str, Check]:
    try:
        import torch
    except Exception as exc:  # PyTorch can fail while loading native libraries.
        return requested, Check("FAIL", "Accelerator", f"could not import torch: {exc}")

    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    npu_available = False
    try:
        import torch_npu  # noqa: F401

        npu = getattr(torch, "npu", None)
        npu_available = bool(npu and npu.is_available())
    except (ImportError, RuntimeError):
        pass

    selected = requested
    if requested == "AUTO":
        if cuda_available:
            selected = "GPU"
        elif npu_available:
            selected = "NPU"

    if selected == "GPU" and cuda_available:
        name = torch.cuda.get_device_name(0)
        return selected, Check("PASS", "Accelerator", f"GPU available: {name}")
    if selected == "NPU" and npu_available:
        return selected, Check("PASS", "Accelerator", "Ascend NPU is available")
    if selected == "AUTO":
        return selected, Check("FAIL", "Accelerator", "no CUDA GPU or Ascend NPU was detected")
    return selected, Check("FAIL", "Accelerator", f"requested {selected}, but it is not available")


def default_requirements(device: str) -> Path:
    return Path("requirements_npu.txt" if device == "NPU" else "requirements.txt")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the Python environment, accelerator, dependencies, "
            "and input paths for VelaScene."
        )
    )
    parser.add_argument(
        "--device",
        type=str.upper,
        choices=("AUTO", "GPU", "NPU"),
        default=os.environ.get("DEVICE_TYPE", "AUTO").upper(),
        help="accelerator to validate (default: DEVICE_TYPE or AUTO)",
    )
    parser.add_argument("--requirements", type=Path, help="requirements file to validate")
    parser.add_argument("--data-root", type=Path, help="dataset directory to validate")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint file to validate")
    parser.add_argument(
        "--skip-device-check",
        action="store_true",
        help="skip hardware detection (useful when preparing an environment off-cluster)",
    )
    parser.add_argument(
        "--skip-package-check",
        action="store_true",
        help="skip validation of packages from the selected requirements file",
    )
    return parser.parse_args(argv)


def run_checks(args: argparse.Namespace) -> list[Check]:
    checks = [check_python(), check_command("torchrun")]
    selected_device = args.device

    if args.skip_device_check:
        checks.append(Check("WARN", "Accelerator", "hardware detection was skipped"))
    else:
        selected_device, accelerator = check_accelerator(args.device)
        checks.append(accelerator)

    if not args.skip_package_check:
        requirements = args.requirements or default_requirements(selected_device)
        checks.append(check_requirements(requirements))
    if args.data_root:
        checks.append(check_path("Dataset", args.data_root, directory=True))
    if args.checkpoint:
        checks.append(check_path("Checkpoint", args.checkpoint, directory=False))
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    checks = run_checks(parse_args(argv))
    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"[{check.status:4}] {check.name:<{width}}  {check.detail}")

    failures = sum(check.status == "FAIL" for check in checks)
    warnings = sum(check.status == "WARN" for check in checks)
    print(f"\nSummary: {failures} failure(s), {warnings} warning(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
