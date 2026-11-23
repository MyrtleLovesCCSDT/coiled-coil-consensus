# coiled-coil-consensus

Design and screening of peptide YES-gates built from coiled coils.

A gate is three peptides. `Q` holds the signal peptide `S` in a moderate-affinity heterodimer. Adding the input peptide `I`, which binds `Q` more tightly, displaces `S` and releases it. The output is free `S`:

```
I + Q:S  <->  I:Q + S
```

Selection happens in two stages. A sequence-level model predicts Kd for every pair, and the gate is solved as a mass-action equilibrium. Surviving candidates are then checked structurally with Boltz-2 and PyRosetta.

## Install

```bash
git clone https://github.com/MyrtleLovesCCSDT/coiled-coil-consensus.git
cd coiled-coil-consensus
pip install -r requirements.txt
```

### pyccscore

The scoring functions come from `pyccscore`, a C++20/pybind11 extension. It is not on PyPI and must be built. Requires a C++20 compiler, Python development headers, OpenMP and CMake.

```bash
git clone --recurse-submodules https://github.com/ajasja/NGB2H.git
cd NGB2H/01_ortoCC_score
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target pyccscore -j
export PYTHONPATH="$PWD/build/src/pyccscore:$PYTHONPATH"
```

The same code is also available directly from [dancsi/OrthoCCTools](https://github.com/dancsi/OrthoCCTools).

On macOS: Apple Clang ships without OpenMP, so install `libomp` and point `-DOpenMP_*` at it. If the vendored `spdlog/fmt` headers fail on `consteval`, set `FMT_USE_CONSTEVAL` to `0` in `src/external/spdlog/include/spdlog/fmt/bundled/core.h`.

### 3D validation (optional)

```bash
pip install -r requirements-3d.txt
```

PyRosetta is licence-gated and is not installed from PyPI — obtain it from [rosettacommons.org](https://www.rosettacommons.org/software/license-and-download).

## Usage

### 1. Calibrate

```bash
python scripts/calibrate.py
```

Fits the consensus on 14 measured Kd pairs and writes `configs/consensus_calibration.json`. Prints both validation metrics.

### 2. Screen

```bash
python scripts/design_yes_gate.py --S0 0.5 --Q0 5.0 --Ion 12.5
```

Builds sequence pools, scores all pairs, solves the equilibrium and ranks the survivors.

| Output | Contents |
| --- | --- |
| `results/designed_yes_gates.csv` | Every triplet that passed the equilibrium filters |
| `results/shortlist_yes_gates.csv` | Best candidates per architectural class and affinity regime |
| `results/yes_gate_titration.png` | `[S]free` against `[I]0` for representative designs |

Concentrations are arguments because the thresholds depend on them: the fraction of a peptide lost to a parasitic complex scales as `~2*C0/Kd`, so the homodimer cutoffs are derived from `--Q0` rather than hardcoded.

### 3. Validate in 3D

```bash
python scripts/export_3d_jobs.py
boltz predict results/3d_inputs/boltz_yaml --out_dir results/boltz_out
python scripts/analyze_boltz.py
python scripts/score_rosetta.py
python scripts/rank_final.py
```

Produces `results/final_master_scorecard.csv` and `results/final_3d_master_summary.png`.

Exported jobs cover the three target dimers, the `Q:Q` and `S:S` homodimers, competing trimers (`IQQ`, `IIQ`, `QQQ`), and the experimental control pairs. Controls carry their measured Kd in the manifest, which is what maps Rosetta REU onto a molar scale — without them interface energies are dimensionless.

Boltz-2 jobs are written with `msa: empty`. These peptides are short and de novo, and an MSA pulls in fragments of natural coiled coils that distort the register. Run ColabFold with `--msa-mode single_sequence` for the same reason.

`Passed_3D` currently requires:

- `dG_separated(Q:S)` below `--max-dg-qs`
- `dG(Q:S) - dG(I:Q) >= --min-gap-iq-qs`, so the input outcompetes the signal peptide
- `dG(I:S) - dG(Q:S) >= --min-gap-qs-is`, so the parasitic pair stays weak
- `MC_Pass_rate >= --min-mc-pass` from the 1D stage

Homodimer and trimer metrics are computed and written to the scorecard, and they enter `Total_Score`, but they are not yet part of the pass/fail gate. `homo_II` is not exported either. Treat the trimer columns as diagnostics to read, not as a filter that has already been applied.

## Method

**Kd prediction.** Four scoring functions (`iCipa_nter`, `iCipa_vert`, `bCIPA`, `Potapov`) are sign-normalised so that higher means stronger, scanned over register shifts `-7..+7` in both orientations, and combined by ridge regression against measured log10(Kd).
 


## Validation

| Benchmark | Metric | Value |
| --- | --- | --- |
| Thomas 2013 + Aronsson 2015, n=14 | LOO-CV on log10(Kd) | r = +0.917, R² = 0.841, RMSE = 0.643 |
| Crooks 2017, n=158 | zero-shot Tm | r = +0.813, R² = 0.660 |

The second row is the stronger of the two. It is a different peptide family, never seen during calibration, and the model is fitted on molar Kd yet transfers to melting temperatures it was never trained on.

The first row overstates generalisation: it is leave-one-*pair*-out, and peptides recur across pairs — `BN4` appears in 4 of the 14 — so a held-out pair is scored by a model that has already seen both of its peptides. A leave-one-*peptide*-out split is the honest number.

Two further limits worth stating. The calibration set contains exactly one homodimer (`BN4:BN4`, Kd 48 µM), so homodimer predictions below that are extrapolation. And `score_rosetta.py` prints control correlations on each run — read them from the run rather than from here, since they depend on which control jobs completed.

## Reference designs

Three peptides from the current shortlist, sharing `I` and `Q`:

| Role | Length | Sequence | Net charge |
| --- | --- | --- | --- |
| `I` | 28 | `EIASLEKEISALEKENASLEWEISALEK` | −5 |
| `Q` | 28 | `KVASLKEQISALKEKNASLKWQISALKE` | +3 |
| `S` | 24 | `LEKEISALESENASLQWEISALEK` | −4 |

Predicted affinities and all per-design metrics live in `results/shortlist_yes_gates.csv` and `results/final_master_scorecard.csv`; they change with every rerun and are deliberately not duplicated here.

