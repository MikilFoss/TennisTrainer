"""Train logistic regression baseline."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (path / "train.npy").exists():
        path.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 10)).astype(np.float32)
        y = rng.integers(0, 6, size=100)
        np.save(path / "train.npy", X[:70])
        np.save(path / "train_labels.npy", y[:70])
        np.save(path / "val.npy", X[70:])
        np.save(path / "val_labels.npy", y[70:])
    X_train = np.load(path / "train.npy")
    y_train = np.load(path / "train_labels.npy")
    X_val = np.load(path / "val.npy")
    y_val = np.load(path / "val_labels.npy")
    return X_train, y_train, X_val, y_val


def main(argv: list[str] | None = None) -> float:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", type=Path, default=Path("models/logreg.pkl"))
    args = parser.parse_args(argv)

    X_train, y_train, X_val, y_val = load_data(args.data)

    clf = LogisticRegression(multi_class="multinomial", C=1.0, max_iter=300)
    clf.fit(X_train, y_train)
    acc = float(clf.score(X_val, y_val))
    print(f"val accuracy: {acc:.3f}")

    args.model.parent.mkdir(parents=True, exist_ok=True)
    with args.model.open("wb") as f:
        pickle.dump(clf, f)
    return acc


if __name__ == "__main__":
    main()
