"""Utility functions for court geometry and homography."""

from __future__ import annotations

import cv2
import numpy as np

# dimensions in meters (approx. ITF standard for singles)
COURT_WIDTH = 8.23
COURT_LENGTH = 23.77
HALF_LENGTH = COURT_LENGTH / 2


def grid_zone_id(x: float, y: float) -> int:
    """Return landing zone id for a 3x2 grid on the opponent half."""
    col = int(np.clip(x / (COURT_WIDTH / 3), 0, 2))
    row = int(np.clip(y / (HALF_LENGTH / 2), 0, 1))
    return row * 3 + col


def solve_homography(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Detect court lines using Hough transform and solve homography.

    The function assumes the court is visible as a white rectangle on a dark
    background. It returns the homography matrix mapping image points to a unit
    square and the mean reprojection error for the detected corners.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=5
    )
    if lines is None:
        raise RuntimeError("No lines detected")
    pts = []
    for x1, y1, x2, y2 in lines[:, 0]:
        pts.extend([(x1, y1), (x2, y2)])
    pts = np.array(pts, dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    img_corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float32)
    court_corners = np.array(
        [[0, 0], [COURT_WIDTH, 0], [0, HALF_LENGTH], [COURT_WIDTH, HALF_LENGTH]],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(img_corners, court_corners)
    proj = cv2.perspectiveTransform(img_corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.mean(np.linalg.norm(proj - court_corners, axis=1)))
    return H, err


__all__ = [
    "COURT_WIDTH",
    "COURT_LENGTH",
    "HALF_LENGTH",
    "grid_zone_id",
    "solve_homography",
]
