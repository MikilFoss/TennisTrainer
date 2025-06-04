"""Run inference on a single rally clip."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from importlib import import_module

from .court_utils import grid_zone_id, solve_homography

pre_mod = import_module("src.01_preprocess")
feature_vector = pre_mod.feature_vector
interpolate_track = pre_mod.interpolate_track


def infer(model_path: Path, clip: Path) -> np.ndarray:
    with model_path.open("rb") as f:
        clf = pickle.load(f)

    cap = cv2.VideoCapture(str(clip))
    frames = []
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError("empty video")

    H, _ = solve_homography(frames[0])
    track = [
        (i, frame.shape[1] / 2, frame.shape[0] / 2) for i, frame in enumerate(frames)
    ]
    ball = interpolate_track(track)
    p1 = np.tile([[100.0, 50.0]], (len(ball), 1))
    p2 = np.tile([[200.0, 50.0]], (len(ball), 1))

    idx = len(ball) // 2
    feat = feature_vector(p1, p2, ball[:, 1:], idx).reshape(1, -1)
    probs = clf.predict_proba(feat)[0]

    pt = cv2.perspectiveTransform(ball[[idx], 1:].reshape(-1, 1, 2), H)[0, 0]
    zone = grid_zone_id(float(pt[0]), float(pt[1]))
    print("predicted zone:", zone)
    print("probs:", probs)

    heat = probs.reshape(2, 3)
    plt.imshow(heat, cmap="hot", origin="lower")
    plt.xticks(range(3))
    plt.yticks(range(2))
    plt.show()
    return probs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/logreg.pkl"))
    args = parser.parse_args(argv)
    infer(args.model, args.clip)


if __name__ == "__main__":
    main()
