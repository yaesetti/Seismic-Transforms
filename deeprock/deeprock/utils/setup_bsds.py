#!/usr/bin/env python3
"""Download and build An et al.'s py-bsds500 backend for Thebe BSDS evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_URL = "https://github.com/anyuzoey/CNNforFaultInterpretation.git"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = ROOT / ".cache" / "an_pybsds500" / "CNNforFaultInterpretation"


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=None if cwd is None else str(cwd), env=env, check=True)


def clone_sparse(repo_url: str, dest: Path, force: bool) -> None:
    if force and dest.exists():
        shutil.rmtree(dest)

    if (dest / ".git").exists():
        run(["git", "fetch", "--depth", "1", "origin", "master"], cwd=dest)
        run(["git", "checkout", "FETCH_HEAD"], cwd=dest)
        run(["git", "sparse-checkout", "set", "py-bsds500"], cwd=dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            repo_url,
            str(dest),
        ]
    )
    run(["git", "sparse-checkout", "set", "py-bsds500"], cwd=dest)


def build_pybsds(pybsds_path: Path) -> None:
    if importlib.util.find_spec("Cython") is None:
        raise SystemExit(
            "Cython is required to build py-bsds500. Install project requirements "
            "first, for example: `pip install -r requirements.txt`."
        )
    env = os.environ.copy()
    for key in ("CFLAGS", "CXXFLAGS"):
        current = env.get(key, "")
        flag = "-Wno-error=format-security"
        env[key] = f"{current} {flag}".strip() if flag not in current else current
    run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=pybsds_path, env=env)


def smoke_test(pybsds_path: Path) -> None:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(pybsds_path) if not old_pythonpath else f"{pybsds_path}:{old_pythonpath}"
    )
    code = (
        "import numpy as np\n"
        "from bsds import correspond_pixels\n"
        "pred=np.zeros((20,30), dtype=bool); pred[5:10,10]=True\n"
        "gt=np.zeros((20,30), dtype=bool); gt[6:11,10]=True\n"
        "m1,m2,_,_=correspond_pixels.correspond_pixels(pred, gt, max_dist=0.0075)\n"
        "print('correspond_pixels smoke:', int((m1>0).sum()), int((m2>0).sum()))\n"
    )
    run([sys.executable, "-c", code], env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-clone the destination first.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Only clone/update the source tree.",
    )
    args = parser.parse_args()

    clone_sparse(args.repo_url, args.dest, args.force)
    pybsds_path = args.dest / "py-bsds500"
    if not pybsds_path.exists():
        raise SystemExit(f"Expected py-bsds500 at {pybsds_path}, but it was not found.")

    if not args.skip_build:
        build_pybsds(pybsds_path)
        smoke_test(pybsds_path)

    print("\nAn et al. py-bsds500 backend is ready:")
    print(pybsds_path)
    print("\nThe training CLI will auto-detect this path when --bsds_enable is used.")
    print("Optional explicit override:")
    print(f"  --bsds_pybsds_path {pybsds_path}")


if __name__ == "__main__":
    main()
