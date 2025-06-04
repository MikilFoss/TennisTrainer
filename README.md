# TennisTrainer

Prototype for predicting ground-stroke landing zones in tennis matches.

## Setup

Create the conda environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate tennistrainer
```

## Pipeline

1. **Download data** (optional - scripts generate small dummy data if missing):
   ```bash
   python src/00_fetch_data.py
   ```
   This retrieves the datasets listed in `data/README.md` into `data/raw/`.

2. **Preprocess** videos into numpy arrays:
   ```bash
   python src/01_preprocess.py
   ```
   Outputs `data/processed/train.npy`, `val.npy` and their label files.

3. **Train** the baseline model:
   ```bash
   python src/02_train.py
   ```
   The trained classifier is stored as `models/logreg.pkl`.

4. **Run the demo** on a random rally clip:
   ```bash
   python src/demo.py
   ```
   This chooses a random video from `data/raw/` and displays the predicted
   landing zone probabilities on a heat map.

## Testing

Unit tests can be executed with

```bash
pytest -q
```
