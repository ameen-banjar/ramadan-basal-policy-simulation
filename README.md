# Ramadan Basal Policy Simulation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20688965.svg)](https://doi.org/10.5281/zenodo.20688965)

Reproducibility package for paired virtual-patient evaluation of basal insulin
policies during simulated Ramadan fasting — including cross-model validation
with the Hovorka 2004 Cambridge simulator.

This public package intentionally contains **no manuscript source or manuscript
PDF**. It includes simulation code, analysis code, compact derived results, and
generated figures. Large raw simulation traces are excluded from Git and
can be regenerated locally.

## Contents

### Simulation scripts
- `simulation/ramadan_simulator.py`: UVA/Padova virtual-patient simulation (original).
- `simulation/simglucose_ramadan.py`: Standalone UVA/Padova 5-policy study (30 patients).
- `simulation/hovorka_ramadan.py`: Hovorka 2004 Cambridge model simulation (10 virtual adults).

### Analysis scripts (UVA/Padova primary study)
- `scripts/run_primary.py`: pre-Ramadan reference plus five full-day policies.
- `scripts/run_sensor_robustness.py`: two additional CGM sensor-seed replicates.
- `scripts/run_scenario_extension.py`: stochastic meal/timing scenarios and an
  80% fasting-window-only policy.
- `scripts/analyze_*.py`: analyses starting from regenerated raw outputs.
- `scripts/verify_derived_results.py`: quick verification of bundled results.

### Data
- `data/derived/`: compact analysis tables (UVA/Padova primary study).
- `data/derived/simglucose_dose_response.csv`: 5-policy dose-response (30 patients).
- `data/derived/hovorka_dose_response.csv`: 5-policy dose-response (10 Hovorka adults).
- `data/input/pre_ramadan_tbr.csv`: minimal phenotype input for the scenario extension.

### Figures
- `figures/figure_dose_response.png`: primary UVA/Padova dose-response.
- `figures/figure_phenotype_policy.png`: phenotype-response analysis.
- `figures/figure_scenario_extension.png`: stochastic scenario extension.
- `figures/fig1_cross_model_dose_response.png`: cross-model comparison (UVA/Padova vs Hovorka).

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

### UVA/Padova primary study

```bash
python scripts/run_primary.py --workers 8
python scripts/analyze_primary.py

python scripts/run_sensor_robustness.py --replicates 1 2 --workers 8
python scripts/analyze_sensor_robustness.py

python scripts/run_scenario_extension.py --scenarios 1 2 3 --workers 8
python scripts/analyze_scenario_extension.py
```

### Hovorka cross-model replication

```bash
python simulation/hovorka_ramadan.py
```

Results are written to `hovorka_simulation/results/`.

### Simglucose 5-policy standalone

```bash
python simulation/simglucose_ramadan.py
```

Results are written to `simglucose_simulation/results/`.

Generated raw files are written under `raw/`, and regenerated tables/figures
under `outputs/`. Both directories are ignored by Git.

## Analysis design

The primary experiment compares 100%, 90%, 80%, 70%, and 60% full-day basal
delivery in paired 30-day simulations using the UVA/Padova model (30 virtual
patients). Robustness checks add independent CGM sensor seeds and independently
seeded meal/timing scenarios. The scenario extension also compares an 80%
fasting-window policy with full-day 80% and tests a policy rule in a held-out
scenario.

Cross-model validation replicates the five-policy experiment in the structurally
distinct Hovorka 2004 Cambridge model (10 virtual adults), confirming that the
graded hypoglycaemia–hyperglycaemia trade-off is not specific to any single
simulator family.

The work is mechanistic and in silico. It does not establish a clinical dosing
recommendation and has not been clinically validated.

## Citation

Banjar A. *Ramadan Basal Policy Simulation*. Version v1.0.0. Zenodo; 2026.
https://doi.org/10.5281/zenodo.20688965
