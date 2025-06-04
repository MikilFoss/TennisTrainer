"""Preprocess side-view clips into training features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

from .court_utils import grid_zone_id, solve_homography


def interpolate_track(track: list[tuple[int, float, float]]) -> np.ndarray:
    """Interpolate ball centre positions using cubic splines."""
    frames = np.array([f for f, *_ in track])
    xs = np.array([x for _, x, _ in track])
    ys = np.array([y for _, _, y in track])
    full = np.arange(frames[0], frames[-1] + 1)
    if len(frames) >= 2:
        xs = CubicSpline(frames, xs)(full)
        ys = CubicSpline(frames, ys)(full)
    else:  # pragma: no cover - degenerate
        xs = np.full_like(full, xs[0])
        ys = np.full_like(full, ys[0])
    return np.c_[full, xs, ys]


def feature_vector(
    p1: np.ndarray, p2: np.ndarray, ball: np.ndarray, idx: int
) -> np.ndarray:
    """Return 10-D feature vector at frame index ``idx``."""

    def vel(arr: np.ndarray) -> np.ndarray:
        if idx == 0:
            return np.zeros(2)
        return arr[idx] - arr[idx - 1]

    f = [
        *p1[idx],
        *vel(p1),
        *p2[idx],
        *vel(p2),
        *ball[idx],
    ]
    return np.array(f, dtype=np.float32)


def process_clip(video: Path, label_path: Path) -> tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(str(video))
    frames = []
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError("empty video")

    H, _ = solve_homography(frames[0])

    # Dummy detection/tracking using bright regions for speed
    ball_track: list[tuple[int, float, float]] = []
    p1_track: list[tuple[float, float]] = []
    p2_track: list[tuple[float, float]] = []
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        m = cv2.moments(gray)
        cx = m["m10"] / (m["m00"] + 1e-5)
        cy = m["m01"] / (m["m00"] + 1e-5)
        ball_track.append((i, cx, cy))
        p1_track.append([cx / 2, cy])
        p2_track.append([cx * 1.5, cy])

    ball = interpolate_track(ball_track)
    p1 = np.array(p1_track, dtype=np.float32)
    p2 = np.array(p2_track, dtype=np.float32)

    hit_frame = 0
    if label_path.exists():
        with label_path.open() as f:
            lbl = json.load(f)
        hit_frame = int(lbl.get("hit_frame", 0))

    feat = feature_vector(p1, p2, ball[:, 1:], hit_frame)

    pt = cv2.perspectiveTransform(ball[[hit_frame], 1:].reshape(-1, 1, 2), H)[0, 0]
    zone = grid_zone_id(float(pt[0]), float(pt[1]))
    return feat, zone


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)

    feats = []
    labels = []
    for clip in sorted(args.raw.glob("*.mp4")):
        label_path = clip.with_suffix(".json")
        f, z = process_clip(clip, label_path)
        feats.append(f)
        labels.append(z)

    if not feats:
        # generate dummy dataset for demonstration
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(100, 10)).astype(np.float32)
        labels = rng.integers(0, 6, size=100).astype(np.int64)
    else:
        feats = np.stack(feats)
        labels = np.array(labels, dtype=np.int64)

    args.out.mkdir(parents=True, exist_ok=True)
    n = int(0.7 * len(feats))
    np.save(args.out / "train.npy", feats[:n])
    np.save(args.out / "train_labels.npy", labels[:n])
    np.save(args.out / "val.npy", feats[n:])
    np.save(args.out / "val_labels.npy", labels[n:])


if __name__ == "__main__":
    main()
