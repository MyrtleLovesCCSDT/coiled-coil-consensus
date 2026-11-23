# Coiled-Coil 1D Consensus & YES-Gate

Simple tool for predicting coiled-coil dimer Kd from sequences and screening 3-peptide YES gates (`I + Q:S <-> I:Q + S`).

## What is inside

- **1D Kd predictor (`src/consensus.py`)**: combines `iCipa_nter`, `iCipa_vert`, `bCIPA`, and `Potapov` from `pyccscore` (scanning shifts `-7..+7`, parallel and antiparallel) using `StandardScaler + RidgeCV`, calibrated on 14 experimental Kd pairs (Thomas et al. 2013, Aronsson et al. 2015).
- **YES-gate simulator (`src/yes_gate.py`)**: solves the 6-dimer equilibrium using `eqtk` and runs Monte Carlo stability checks.
- **De novo gate designer (`scripts/design_yes_gate.py`)**: screens coiled-coil sequence pools and outputs ranked YES-gate triplets to `results/`.

## Validation

- **LOO-CV on the 14 Kd pairs**: `r = +0.917`, `R2 = 0.841`, `RMSE = 0.643` log10(M).
  Note this is leave-one-*pair*-out, and peptides recur across pairs (e.g. `BN4` appears in
  4 of 14), so it overstates generalisation. A leave-one-*peptide*-out split is the honest number.
- **Zero-shot on Crooks et al. 2017** (`n = 158` Tm pairs, a different peptide family, never seen
  during calibration): `r = +0.813`, `R2 = 0.660`. The model is fitted on molar Kd and transfers
  to melting temperatures it was never trained on — this is the stronger evidence of the two.

## Setup

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Build and install `pyccscore`

`pyccscore` is a C++20/pybind11 extension from [OrthoCCTools](https://github.com/dancsi/OrthoCCTools) (also included as the `01_ortoCC_score` submodule in [NGB2H](https://github.com/ajasja/NGB2H)). It is not on PyPI and requires a C++20 compiler, Python 3 development headers, OpenMP, and CMake.

Clone **either** `OrthoCCTools` directly **or** `NGB2H` with submodules:

```bash
# Option A: OrthoCCTools directly
git clone https://github.com/dancsi/OrthoCCTools.git
cd OrthoCCTools

# Option B: via NGB2H (requires --recurse-submodules)
# git clone --recurse-submodules https://github.com/ajasja/NGB2H.git
# cd NGB2H/01_ortoCC_score
```

Build only the `pyccscore` Python module:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target pyccscore -j
```

Then copy `build/src/pyccscore/pyccscore*.so` (or `.pyd` on Windows) into your Python environment's `site-packages` (check via `python -c "import sysconfig; print(sysconfig.get_path('platlib'))"`), or add `build/src/pyccscore` to `PYTHONPATH`.

> **macOS / modern Clang notes:**
> - Apple Clang does not include OpenMP by default. Install `llvm-openmp` (e.g. via conda-forge or Homebrew `libomp`) and pass `-DPython3_EXECUTABLE="$(which python)"` plus `-DOpenMP_C_*` / `-DOpenMP_CXX_*` flags pointing to `libomp`.
> - Modern Clang in C++20 mode may reject `consteval` in the bundled `spdlog/fmt` header (`FMT_STRING`); setting `FMT_USE_CONSTEVAL` to `0` in `src/external/spdlog/include/spdlog/fmt/bundled/core.h` resolves this.

## Usage

1. Generate `configs/consensus_calibration.json`:
```bash
python scripts/calibrate.py
```

2. Search for de novo YES-gate designs (saves results to `results/`):
```bash
python scripts/design_yes_gate.py --S0 0.5 --Q0 5.0 --Ion 12.5
```

Default concentrations use `Q0 = 5.0 uM` and `I_on = 12.5 uM` (`S0 = 0.5 uM`) so that `Q0` exceeds `Kd_QS` across the `0.14-2.51 uM` (`pKd 5.60-6.85`) design window and avoids biasing selection toward kinetically trapped pairs.
