# Coiled-Coil Consensus: YES-Gate Design via Molecular Commutation

Computational pipeline for de novo design and biophysical validation of peptide YES-gates (`I + Q:S <-> I:Q + S`) via reversible coiled-coil strand commutation, translating the concept of non-complementary information processing (Nikitin, *Nature Chemistry*, 2023) to synthetic protein systems.

---

## Architecture Overview

```text
                      +--- [I] (Input, acidic, e.g. -5)
                      |
                      v
  [Q] (Quencher/Anchor, basic, +3) : [S] (Signal/Reporter, acidic, -3/-4)
                      |
                      +---> Reversible displacement
                      |
                      v
            [I:Q] (tight dimer)  +  [S] (free, active output)
```

1. **1D Multi-Model Consensus**: Combines 4 orthogonal scoring functions (`iCipa_nter`, `iCipa_vert`, `bCIPA`, `Potapov`) via sign-normalized Ridge regression trained on experimentally measured Kd values.
2. **Law of Mass Action Equilibrium (`eqtk`)**: Simulates 9-species equilibrium (6 dimers + 3 monomers) with Monte Carlo stress-testing (sigma = RMSE_LOO) over all affinity constants.
3. **Sequence Design Rules**: Enforces designed heptad register (`d=L`, `b/c=A/S`), electrostatic asymmetry (E/K -> Q/A), and buried polar specificity (`Asn` at core position `a`).
4. **Full-Atom 3D Validation**: Evaluates target heterodimers, homodimers, and potential trimer traps (`IQQ`, `IIQ`, `QQQ`) using Boltz-2 structure prediction and PyRosetta energy relaxation.

---

## Repository Structure

```
coiled-coil-consensus/
|-- configs/
|   `-- consensus_calibration.json   # Fitted Ridge coefficients & normalization scalers
|-- src/
|   |-- anchors.py                   # Reference experimental datasets (Thomas, Aronsson, Crooks)
|   |-- consensus.py                 # Multi-model scoring & register inference
|   |-- yes_gate.py                  # EQTK equilibrium & Monte Carlo robustness
|   `-- structure_3d.py              # CIF coordinate parsing & geometric auditing
|-- scripts/
|   |-- calibrate.py                 # Fit 1D consensus on experimental anchors
|   |-- design_yes_gate.py           # Rational design & candidate screening
|   |-- export_3d_jobs.py            # Generate Boltz-2 YAML & ColabFold FASTA jobs
|   |-- analyze_boltz.py             # Extract 3D metrics from Boltz-2 CIF predictions
|   |-- score_rosetta.py             # PyRosetta FastRelax & InterfaceAnalyzer scoring
|   `-- rank_final.py                # Multi-stage ranking with trimer and homodimer veto
|-- requirements.txt
`-- requirements-3d.txt              # Optional dependencies for 3D validation
```

---

## Installation

### 1. Python Environment

```bash
git clone https://github.com/MyrtleLovesCCSDT/coiled-coil-consensus.git
cd coiled-coil-consensus
pip install -r requirements.txt
```

### 2. Building `pyccscore`

The consensus predictor relies on `pyccscore` compiled from [dancsi/OrthoCCTools](https://github.com/dancsi/OrthoCCTools) or [ajasja/NGB2H](https://github.com/ajasja/NGB2H) (`01_ortoCC_score`):

```bash
git clone https://github.com/ajasja/NGB2H.git
cd NGB2H/01_ortoCC_score
mkdir build && cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

Ensure the compiled `.so` (Linux/macOS) or `.pyd` (Windows) is accessible on your `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/01_ortoCC_score/build/src/pyccscore:$PYTHONPATH"
```

*Note for modern Clang / C++20*: If compiling against newer standard libraries, ensure `FMT_USE_CONSTEVAL = 0` is set in the vendored `spdlog/fmt` headers if consteval errors occur.

---

## Quickstart Pipeline

### Step 1: Consensus Calibration

Fits 1D Ridge regression on combined Thomas et al. (2013) and Aronsson et al. (2015) anchor data:

```bash
python scripts/calibrate.py --out configs/consensus_calibration.json --save-csv
```

- **LOO-CV on Kd anchors (n=14)**: r = +0.931, R2 = 0.868, RMSE = 0.62 log10(M)
- **Zero-shot Crooks et al. (2017) Tm (n=158)**: r = +0.813, R2 = 0.661

### Step 2: YES-Gate Screening & Audit

Screens de novo soluble candidates and audits switching behavior at target operating concentrations:

```bash
python scripts/design_yes_gate.py --S0 0.5 --Q0 5.0 --Ion 12.5 --n-mc 200
```

Outputs:
- `results/designed_yes_gates.csv`: Full screened library.
- `results/shortlist_yes_gates.csv`: Pareto-optimal candidates by architectural class and regime.
- `results/yes_gate_titration.png`: Equilibrium titration curves for representative designs.

---

## 3D Validation Pipeline (Optional)

Requires `boltz`, `gemmi`, and `pyrosetta` (see `requirements-3d.txt`).

### 1. Export Target Complexes & Controls

Generates inputs for all dimers (`I:Q`, `Q:S`, `I:S`), homodimers (`I:I`, `Q:Q`, `S:S`), trimer traps (`IQQ`, `IIQ`, `QQQ`), and experimental controls:

```bash
python scripts/export_3d_jobs.py
```

### 2. Predict Structures via Boltz-2

```bash
boltz predict results/3d_inputs/boltz_yaml --out_dir results/boltz_out
```

### 3. Extract Geometry, Relax with PyRosetta, and Rank

```bash
python scripts/analyze_boltz.py
python scripts/score_rosetta.py
python scripts/rank_final.py
```

Outputs `results/final_master_scorecard.csv` and `results/final_3d_master_summary.png`.

A candidate achieves `Passed_3D = True` only if it satisfies all of the following:
- Thermodynamic hierarchy: `dG(IQ) < dG(QS) << dG(IS)`
- Homodimer suppression: `dG(QQ)` and `dG(II)` are strictly suppressed relative to reference homodimers
- Trimer veto (`Trimer_Safe = True`): Rejects designs where any competitor trimer forms a well-packed hydrophobic core without unsatisfied buried hydrogen bonds.

---

## Reference Benchmarks

| Benchmark Dataset | Metric | Consensus Value | Description |
| :--- | :--- | :--- | :--- |
| **Thomas + Aronsson (n=14)** | LOO r / RMSE | +0.931 / 0.62 | Cross-validated log10(Kd) accuracy |
| **Crooks et al. (n=158)** | Zero-shot r / R2 | +0.813 / 0.661 | Zero-shot Tm linear regression |
| **NGB2H CCmax (n=17,977)** | ROC-AUC | 0.884 | Distinguishing high-affinity from non-binding pairs |
| **Rosetta Controls (n=6)** | Pearson r / Spearman rho | +0.858 / +0.928 | Interface dG_separated vs experimental log10(Kd) |

---

## Top Validated Designs

From the final multi-stage selection, representative winning candidates include:

- **D13** (Class: `Truncated_24aa`, Regime: `HighContrast_<3nM`):
  - Kd(IQ) ~ 0.82 nM, Kd(QS) ~ 0.15 uM, MC Pass = 76%
  - Rosetta: dG(IQ) = -63.32 REU, dG(QS) = -55.01 REU, Gap = +8.30 REU
  - Trimer audit: Unsatisfied H-bonds jump from 0 in dimer to 4.67 in trimer; trimer safe.
- **Shared Input Pair for Synthesis**:
  - `I` (shared, 28 aa): `EIASLEKEISALEKENASLEWEISALEK`
  - `Q_D13` (28 aa): `KVASLKEQISALKEKNASLKWQISALKE`
  - `S_D13` (24 aa, C-terminal aligned): `LEKEISALESENASLQWEISALEK`

---

## License

MIT License. See `LICENSE` for details.
