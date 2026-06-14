# Ramadan Basal Policy Simulation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20688965.svg)](https://doi.org/10.5281/zenodo.20688965)

Reproducibility package for paired virtual-patient evaluation of basal insulin
policies during simulated Ramadan fasting.

This public package intentionally contains **no manuscript source or manuscript
PDF**. It includes simulation code, analysis code, compact derived results, and
three generated figures. Large raw simulation traces are excluded from Git and
can be regenerated locally.

## Contents

- `simulation/ramadan_simulator.py`: UVA/Padova virtual-patient simulation.
- `scripts/run_primary.py`: pre-Ramadan reference plus five full-day policies.
- `scripts/run_sensor_robustness.py`: two additional CGM sensor-seed replicates.
- `scripts/run_scenario_extension.py`: stochastic meal/timing scenarios and an
  80% fasting-window-only policy.
- `scripts/analyze_*.py`: analyses starting from regenerated raw outputs.
- `scripts/verify_derived_results.py`: quick verification of bundled results.
- `data/derived/`: compact analysis tables.
- `data/input/pre_ramadan_tbr.csv`: minimal phenotype input for the scenario
  extension.
- `figures/`: generated result figures, without manuscript text.

Virtual patient identifiers are simulator model identifiers, not human
participant identifiers. No clinical or personally identifiable data are
included.

## Installation

Python 3.11 was used for the archived analysis.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick verification

This command checks the principal bundled numerical results without generating
the large raw trajectories:

```bash
python scripts/verify_derived_results.py
```

## Full reproduction

The complete simulation is computationally intensive and generates roughly
2 GB of raw CSV output.

```bash
python scripts/run_primary.py --workers 8
python scripts/analyze_primary.py

python scripts/run_sensor_robustness.py --replicates 1 2 --workers 8
python scripts/analyze_sensor_robustness.py

python scripts/run_scenario_extension.py --scenarios 1 2 3 --workers 8
python scripts/analyze_scenario_extension.py
```

Generated raw files are written under `raw/`, and regenerated tables/figures
under `outputs/`. Both directories are ignored by Git.

## Analysis design

The primary experiment compares 100%, 90%, 80%, 70%, and 60% full-day basal
delivery in paired 30-day simulations. Robustness checks add independent CGM
sensor seeds and independently seeded meal/timing scenarios. The scenario
extension also compares an 80% fasting-window policy with full-day 80% and
tests a policy rule in a held-out scenario.

The work is mechanistic and in silico. It does not establish a clinical dosing
recommendation and has not been clinically validated.

## Citation

Banjar A. *Ramadan Basal Policy Simulation*. Version v1.0.0. Zenodo; 2026.
https://doi.org/10.5281/zenodo.20688965
