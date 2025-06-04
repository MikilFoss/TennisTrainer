"""Download public datasets used by the project."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

DATASETS = {
    "tennis_shot": {
        # DOI 10.17632/75m8vz7jr2.4
        "url": (
            "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/75m8vz7jr2/4/files/Tennis_Shot_Dataset.zip?download=1"
        ),
        # Pre-computed digest of the zip file
        "sha256": "4a1ff9fe9b9d8e2b8dbb7626d3e4c59cf1b7aa90d108d234ca970d52f7e1a49b",
        "filename": "tennis_shot_dataset.zip",
    },
    "tracknet": {
        # URL provided in the TrackNet paper
        "url": "https://www.csie.ntu.edu.tw/~cyy/TrackNet/TrackNet-2017.zip",
        "sha256": "03b8da0f6659b6cdaefb9738db1f8fd12725832c3805c829c793b4d7ae9e1fd5",
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
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)

    for name, info in DATASETS.items():
        fetch_dataset(name, args.output / info["filename"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - simple cli
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
