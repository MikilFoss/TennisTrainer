"""Download public datasets used by the project."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

DATASETS = {
    "tennis_shot": {
        "url": "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/75m8vz7jr2/1/files?download=1",
        "sha256": "dummysha256shot",  # placeholder
        "filename": "tennis_shot_dataset.zip",
    },
    "tracknet": {
        "url": "https://zenodo.org/record/5594450/files/data.zip?download=1",
        "sha256": "dummysha256tracknet",  # placeholder
        "filename": "tracknet_dataset.zip",
    },
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_dataset(name: str, dest: Path) -> None:
    info = DATASETS[name]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256sum(dest) == info["sha256"]:
        print(f"{name}: already downloaded")
        return
    print(f"Downloading {name}...")
    tmp = dest.with_suffix(".tmp")
    urlretrieve(info["url"], tmp)
    digest = sha256sum(tmp)
    if digest != info["sha256"]:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"SHA256 mismatch for {name}: {digest}")
    tmp.rename(dest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    for name, info in DATASETS.items():
        fetch_dataset(name, args.output / info["filename"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - simple cli
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
