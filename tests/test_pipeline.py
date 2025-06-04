from pathlib import Path

import numpy as np
import cv2
import pickle
from importlib import import_module
from src import court_utils

train_mod = import_module("src.02_train")


def test_homography_accuracy():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (350, 150), (255, 255, 255), 2)
    H, err = court_utils.solve_homography(img)
    assert err < 3


def test_feature_length():
    f = np.zeros(10, dtype=np.float32)
    assert len(f) == 10


def test_model_forward(tmp_path: Path):
    data_dir = tmp_path / "data"
    model_path = tmp_path / "model.pkl"
    train_mod.main(["--data", str(data_dir), "--model", str(model_path)])
    assert model_path.exists()
    with model_path.open("rb") as f:
        clf = pickle.load(f)
    out = clf.predict_proba(np.zeros((1, 10), dtype=np.float32))
    assert out.shape == (1, 6)
