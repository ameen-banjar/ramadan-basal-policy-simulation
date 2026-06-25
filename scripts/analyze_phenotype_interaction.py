#!/usr/bin/env python3
"""
Formal policy x pre-Ramadan TBR interaction (effect modification).

Tests whether the TIR response to basal reduction is modified by each
virtual patient's pre-Ramadan TBR. Uses the within-patient TIR change from
the 100% policy:

    dTIR_ip = TIR_ip - TIR_i,100      for p in {90,80,70,60}

Model (policy as categorical, patient as block):
    dTIR_ip = alpha_p + beta_p * (PreTBR_i - mean PreTBR) + b_i + eps_ip

Primary inference is a PATIENT-BLOCKED permutation omnibus test of
    H0: beta_90 = beta_80 = beta_70 = beta_60 = 0
by permuting the pre-Ramadan TBR vector ACROSS patients while keeping each
patient's four policy rows intact (repeated-measures structure preserved).
The omnibus statistic is the sum of squared per-policy slopes. Per-policy
slopes receive patient-level bootstrap 95% CIs.

Outputs:
  data/derived/phenotype_interaction_slopes.csv
  prints omnibus P and per-policy slopes
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
RNG = np.random.default_rng(20260218)
N_PERM = 20000
N_BOOT = 20000
REDUCED = [90, 80, 70, 60]


def slope(x, y):
    """OLS slope of y on x."""
    x = x - x.mean()
    denom = (x * x).sum()
    return (x * y).sum() / denom if denom > 0 else 0.0


def main():
    sl = pd.read_csv(DERIVED / "five_policy_subject_level.csv")
    pre = pd.read_csv(ROOT / "data" / "input" / "pre_ramadan_tbr.csv")

    tir = sl.pivot_table(index="Patient", columns="PolicyPercent", values="TIR")
    pre = pre.set_index("Patient")["PreTBR"]
    patients = [p for p in tir.index if p in pre.index]
    tir = tir.loc[patients]
    pre = pre.loc[patients]
    pre_c = (pre - pre.mean()).values   # centred pre-TBR
    n = len(patients)
    print(f"Patients: {n}")

    # within-patient TIR change from 100%
    dT = {p: (tir[p] - tir[100]).values for p in REDUCED}

    # observed per-policy slopes
    obs_slopes = {p: slope(pre_c, dT[p]) for p in REDUCED}
    obs_stat = sum(s * s for s in obs_slopes.values())

    # patient-blocked permutation: permute pre-TBR across patients,
    # each patient's four policy rows move together (block intact)
    perm_stats = np.empty(N_PERM)
    for i in range(N_PERM):
        pc = pre_c[RNG.permutation(n)]
        perm_stats[i] = sum(slope(pc, dT[p]) ** 2 for p in REDUCED)
    p_omni = (np.sum(perm_stats >= obs_stat) + 1) / (N_PERM + 1)

    # per-policy bootstrap CIs
    rows = []
    for p in REDUCED:
        d = dT[p]
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = RNG.integers(0, n, n)
            boot[b] = slope(pre_c[idx], d[idx])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"PolicyPct": p, "slope": obs_slopes[p],
                     "CI_low": lo, "CI_high": hi})
    res = pd.DataFrame(rows)
    res.to_csv(DERIVED / "phenotype_interaction_slopes.csv", index=False)

    print(f"\nOmnibus patient-blocked permutation test of effect modification:")
    print(f"  H0: all per-policy pre-TBR slopes = 0")
    print(f"  statistic = sum of squared slopes = {obs_stat:.3f}")
    print(f"  P = {p_omni:.4f}\n")
    print("Per-policy slope of dTIR on centred pre-Ramadan TBR (points TIR per 1% TBR):")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
