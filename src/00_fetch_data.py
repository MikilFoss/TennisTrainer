"""Download public datasets used by the project."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.request import build_opener, HTTPCookieProcessor, urlretrieve

DATASETS = {
    "tennis_shot": {
        "url": "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/75m8vz7jr2-4.zip",
        "sha256": "38ba33baad8ab8ad73170d14f07f22a2ee163606af99ac4046369a59a83189c5",
        "filename": "tennis_shot_dataset.zip",
    },
    "tracknet": {
        "url": "https://drive.google.com/uc?export=download&id=1DQ3ZbvokTsgOq6x-ay6O8U2W4a8e3LFw&confirm=t",
        "sha256": "bb45372d011ef4092d78173169c8ad6280d7dbdb9998efaf2b1c72a63ca1dd18",
        "filename": "tracknet_dataset.zip",
    },
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_gdrive(file_id: str, dest: Path) -> None:
    """Download a file from Google Drive, handling large file confirmations."""
    base_url = "https://drive.google.com/uc?export=download"
    cj = CookieJar()
    opener = build_opener(HTTPCookieProcessor(cj))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]

    # First request to get confirmation token
    url = f"{base_url}&id={file_id}"
    response = opener.open(url)
    content = response.read()

    # Check if we got HTML (confirmation page) or actual file
    if content[:4] == b"PK\x03\x04" or content[:2] != b"<!":
        # Got the actual file
        with dest.open("wb") as f:
            f.write(content)
        return

    # Look for confirmation token in the HTML response
    html = content.decode("utf-8", errors="ignore")
    match = re.search(r'confirm=([0-9A-Za-z_-]+)', html)
    if match:
        confirm = match.group(1)
        url = f"{base_url}&id={file_id}&confirm={confirm}"
    else:
        # Try with confirm=t as fallback
        url = f"{base_url}&id={file_id}&confirm=t"

    # Download with confirmation
    response = opener.open(url)
    with dest.open("wb") as f:
        shutil.copyfileobj(response, f)


def _is_gdrive_url(url: str) -> str | None:
    """Extract Google Drive file ID from URL, or return None."""
    match = re.search(r"drive\.google\.com/.*[?&]id=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def fetch_dataset(name: str, dest: Path) -> None:
    info = DATASETS[name]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256sum(dest) == info["sha256"]:
        print(f"{name}: already downloaded")
        return
    print(f"Downloading {name}...")
    tmp = dest.with_suffix(".tmp")

    file_id = _is_gdrive_url(info["url"])
    if file_id:
        _download_gdrive(file_id, tmp)
    else:
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
