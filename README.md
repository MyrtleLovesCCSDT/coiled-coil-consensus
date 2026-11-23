# Coiled-Coil 1D Consensus & YES-Gate

Pipeline for predicting coiled-coil dimer Kd from sequences and screening 3-peptide YES gates (`I + Q:S <-> I:Q + S`).

## Method

- **1D Kd Consensus (`src/consensus.py`)**: combines `iCipa_nter`, `iCipa_vert`, `bCIPA`, and `Potapov` from `pyccscore` (scanning register shifts `-7..+7`, parallel and antiparallel) using `StandardScaler + RidgeCV`.
- **Calibration (`scripts/calibrate.py`)**: fitted on 14 experimental Kd pairs (Thomas et al. 2013, Aronsson et al. 2015) and zero-shot evaluated on 158 Tm pairs (Crooks et al. 2017):
  - LOO-CV Kd (`n=14`): `r = +0.917`, `R2 = 0.841`, `RMSE = 0.643` log10(M)
  - Zero-shot Crooks Tm (`n=158`): `r = +0.813`, `R2 = 0.660`
- **YES-Gate Screening (`src/yes_gate.py`, `scripts/design_yes_gate.py`)**: designs soluble 21-28 aa peptides, solves the 6-dimer equilibrium via `eqtk`, and filters by slippage margin and Monte Carlo perturbation (`sigma = LOO RMSE`).

## Setup

```bash
pip install -r requirements.txt
```

Requires `pyccscore` (from [NGB2H](https://github.com/ajasja/NGB2H)) installed in your Python environment.

## Usage

1. Calibrate the 1D consensus (`configs/consensus_calibration.json`):
```bash
python scripts/calibrate.py
```

2. Run de novo YES-gate search (saves tables and titration curves to `results/`):
```bash
python scripts/design_yes_gate.py
```
