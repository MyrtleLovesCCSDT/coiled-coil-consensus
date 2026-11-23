# Coiled-Coil 1D Consensus & YES-Gate

Simple tool for predicting coiled-coil dimer Kd from sequences and screening 3-peptide YES gates (`I + Q:S <-> I:Q + S`).

## What is inside

- **1D Kd predictor (`src/consensus.py`)**: combines `iCipa_nter`, `iCipa_vert`, `bCIPA`, and `Potapov` from `pyccscore` (scanning shifts `-7..+7`, parallel and antiparallel) using `StandardScaler + RidgeCV`, calibrated on 14 experimental Kd pairs (Thomas et al. 2013, Aronsson et al. 2015).
- **YES-gate simulator (`src/yes_gate.py`)**: solves the 6-dimer equilibrium using `eqtk` and runs Monte Carlo stability checks.

## Setup

```bash
pip install -r requirements.txt
```

You also need `pyccscore` (from [NGB2H](https://github.com/ajasja/NGB2H)) installed in your Python environment.

## Usage

1. Generate `configs/consensus_calibration.json`:
```bash
python scripts/calibrate.py
```

2. Predict Kd for one pair:
```bash
python scripts/predict.py --seq1 EIAALEQEIAALEKENAALEWEIAALEQ --reg1 g --seq2 KIAALKQKIAALKYKNAALKKKIAALKQ --reg2 g
```

Or run on a CSV file (`seq1,seq2` columns):
```bash
python scripts/predict.py --csv pairs.csv --out scored.csv
```

3. Check a YES-gate triplet `(I, Q, S)`:
```bash
python scripts/predict.py --yes-gate --I <SEQ_I> --Q <SEQ_Q> --S <SEQ_S>
```

4. Search for de novo YES-gate designs (saves results to `results/`):
```bash
python scripts/design_yes_gate.py
```
