"""Run inference on a random video clip."""

from __future__ import annotations

import argparse
import random
from importlib import import_module
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Demo random clip analysis")
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--model", type=Path, default=Path("models/logreg.pkl"))
    args = parser.parse_args(argv)

    vids = sorted(args.data.glob("*.mp4"))
    if not vids:
        raise SystemExit(f"no videos found in {args.data}")
    clip = random.choice(vids)
    print(f"Analyzing {clip.name}...")

    infer_mod = import_module("src.03_infer_single_video")
    infer_mod.infer(args.model, clip)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
