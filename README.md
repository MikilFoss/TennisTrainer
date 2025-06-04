# TennisTrainer

Prototype for predicting ground-stroke landing zones in tennis matches.

## Setup

Create the conda environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate tennistrainer
```

Large datasets are referenced in `data/README.md` and are downloaded via
`python src/00_fetch_data.py`.
