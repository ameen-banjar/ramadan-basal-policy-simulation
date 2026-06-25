#!/usr/bin/env python3
"""
Meal-size sensitivity analysis of the basal dose-response.

Uses the SAME primary-engine output as the main analysis (run_primary.py),
restricted to the complete-case virtual patients (those completing all five
30-day Ramadan policies). For each policy, daily TIR is stratified by the
day's carbohydrate level (normalised to each patient's age-group target),
and the within-patient difference between low-CHO days (<=85% of target) and
high-CHO days (>=110%) is computed with a patient-level bootstrap 95% CI.
A basal-policy x carbohydrate-category interaction is tested by comparing the
low-vs-high difference across policies (permutation test on the policy slope).

Outputs:
  data/derived/meal_size_by_policy.csv         (per-policy low/high means + CI)
  data/derived/meal_size_patient_level.csv     (per-patient low/high TIR)
  figures/figure_meal_size.png
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "raw" / "primary"
DERIVED = ROOT / "data" / "derived"
FIGS = ROOT / "figures"
DERIVED.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

POLICIES = {
    "Ramadan_100_NoIntervention": 100,
    "Ramadan_090_Intervention":   90,
    "Ramadan_080_Intervention":   80,
    "Ramadan_070_Intervention":   70,
    "Ramadan_060_Intervention":   60,
}
# Age-group daily carbohydrate targets (g) used in the protocol
CHO_TARGET = {"child": 150.0, "adolescent": 220.0, "adult": 250.0}
LOW_THR, HIGH_THR = 0.85, 1.10   # fractions of target
RNG = np.random.default_rng(20260218)
N_BOOT = 20000


def load_daily() -> pd.DataFrame:
    """Collect all per-day summaries across patients and policies."""
    rows = []
    for f in RAW.rglob("*_DailySummary.csv"):
        period = f.parent.name
        if period not in POLICIES:
            continue
        df = pd.read_csv(f)
        df["PolicyPct"] = POLICIES[period]
        rows.append(df)
    daily = pd.concat(rows, ignore_index=True)
    # CHO as fraction of the patient's age-group target
    daily["CHO_frac"] = daily.apply(
        lambda r: r["CHO_g"] / CHO_TARGET[r["Role"]], axis=1
    )
    daily["CHO_cat"] = np.where(daily["CHO_frac"] <= LOW_THR, "low",
                        np.where(daily["CHO_frac"] >= HIGH_THR, "high", "mid"))
    return daily


def complete_case_patients(daily: pd.DataFrame) -> list[str]:
    """Patients completing all five policies (30 observed days each)."""
    counts = (daily[daily["CHO_cat"] != "x"]
              .groupby(["Patient", "PolicyPct"])["Date"].nunique()
              .unstack("PolicyPct"))
    full = counts[(counts >= 30).all(axis=1)].index.tolist()
    return sorted(full)


def bootstrap_ci(values: np.ndarray, n=N_BOOT):
    idx = RNG.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    daily = load_daily()
    patients = complete_case_patients(daily)
    print(f"Complete-case patients: {len(patients)}")
    daily = daily[daily["Patient"].isin(patients)]

    # Patient-level mean TIR on low- and high-CHO days, per policy
    pl_rows = []
    for pct in POLICIES.values():
        sub = daily[daily["PolicyPct"] == pct]
        for pat in patients:
            ps = sub[sub["Patient"] == pat]
            lo = ps[ps["CHO_cat"] == "low"]["TIR"].mean()
            hi = ps[ps["CHO_cat"] == "high"]["TIR"].mean()
            pl_rows.append({"Patient": pat, "PolicyPct": pct,
                            "TIR_low": lo, "TIR_high": hi,
                            "diff": lo - hi})
    pl = pd.DataFrame(pl_rows).dropna()
    pl.to_csv(DERIVED / "meal_size_patient_level.csv", index=False)

    # Per-policy summary with patient-level bootstrap CI on the difference
    summ = []
    for pct in POLICIES.values():
        d = pl[pl["PolicyPct"] == pct]
        diffs = d["diff"].values
        lo_ci, hi_ci = bootstrap_ci(diffs)
        summ.append({
            "PolicyPct": pct,
            "n": len(d),
            "TIR_low_mean": d["TIR_low"].mean(),
            "TIR_high_mean": d["TIR_high"].mean(),
            "diff_mean": diffs.mean(),
            "diff_CI_low": lo_ci,
            "diff_CI_high": hi_ci,
        })
    summ = pd.DataFrame(summ)
    summ.to_csv(DERIVED / "meal_size_by_policy.csv", index=False)
    print(summ.round(2).to_string(index=False))

    # Interaction test: does the low-vs-high difference depend on policy?
    # Permutation on the Spearman correlation between policy and within-patient diff.
    from scipy.stats import spearmanr
    obs_rho, _ = spearmanr(pl["PolicyPct"], pl["diff"])
    perm = np.empty(N_BOOT)
    pol = pl["PolicyPct"].values.copy()
    dif = pl["diff"].values
    for i in range(N_BOOT):
        perm[i] = spearmanr(RNG.permutation(pol), dif)[0]
    p_int = (np.sum(np.abs(perm) >= abs(obs_rho)) + 1) / (N_BOOT + 1)
    print(f"\nInteraction (policy x CHO category): "
          f"rho={obs_rho:.3f}, permutation P={p_int:.4f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pcts = list(POLICIES.values())
    ax = axes[0]
    for pct, color in zip(pcts, ['#1a4369','#2e7d32','#e65100','#6a1b9a','#b71c1c']):
        sub = daily[daily["PolicyPct"] == pct]
        bc = sub.groupby("CHO_frac")["TIR"].mean()
        ax.plot(bc.index*100, bc.values, 'o-', color=color, lw=2, ms=4,
                label=f"{pct}%")
    ax.set_xlabel("Daily carbohydrate (% of target)")
    ax.set_ylabel("Time in range (%)")
    ax.set_title(f"TIR by meal size and policy\n({len(patients)} complete-case patients)",
                 fontweight='bold', fontsize=11)
    ax.legend(title="Basal policy", fontsize=9)
    ax.grid(alpha=0.3, ls='--')

    ax = axes[1]
    x = np.arange(len(pcts))
    diffs = [summ[summ.PolicyPct==p]["diff_mean"].values[0] for p in pcts]
    los   = [summ[summ.PolicyPct==p]["diff_CI_low"].values[0] for p in pcts]
    his   = [summ[summ.PolicyPct==p]["diff_CI_high"].values[0] for p in pcts]
    err = [np.array(diffs)-np.array(los), np.array(his)-np.array(diffs)]
    ax.bar(x, diffs, color='#1976d2', alpha=0.85)
    ax.errorbar(x, diffs, yerr=err, fmt='none', ecolor='k', capsize=4)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{p}%" for p in pcts])
    ax.set_xlabel("Basal policy")
    ax.set_ylabel("Low-minus-high CHO TIR difference (pp)")
    ax.set_title("Meal-size effect with patient-level 95% CI",
                 fontweight='bold', fontsize=11)
    ax.grid(alpha=0.3, ls='--', axis='y')

    fig.suptitle("Meal-size sensitivity of the basal dose-response",
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGS / "figure_meal_size.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {FIGS/'figure_meal_size.png'}")


if __name__ == "__main__":
    main()
