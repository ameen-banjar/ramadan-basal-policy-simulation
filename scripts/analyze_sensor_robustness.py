#!/usr/bin/env python3
"""Sensor-seed robustness analysis for the five-policy dose-response study.

Combines the primary results (seed/replicate 0) with two additional
independent-CGM-sensor-seed replicates (1 and 2) for the same 29 complete-case
virtual patients and five basal policies. Reports seed-averaged dose-response,
a variance decomposition (between-patient vs. within-patient/seed), and the
phenotype correlation recomputed on seed-averaged data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RUNS = ROOT / "raw" / "primary"
ROBUST_RUNS = ROOT / "raw" / "sensor_robustness"
TABLES = ROOT / "outputs" / "tables"
RNG = np.random.default_rng(20260613)
PERMUTATION_RNG = np.random.default_rng(20260614)

POLICIES = {
    "Ramadan_100_NoIntervention": 100,
    "Ramadan_090_Intervention": 90,
    "Ramadan_080_Intervention": 80,
    "Ramadan_070_Intervention": 70,
    "Ramadan_060_Intervention": 60,
}
METRICS = ["TIR", "TBR", "TBR_<54", "TAR"]


def bootstrap_mean_ci(values: np.ndarray, replicates: int = 20000) -> tuple[float, float]:
    samples = RNG.choice(values, size=(replicates, len(values)), replace=True)
    return tuple(np.percentile(samples.mean(axis=1), [2.5, 97.5]))


def paired_mean_permutation_p(values: np.ndarray, replicates: int = 200000) -> float:
    """Two-sided sign-flip test for a paired mean difference."""
    observed = abs(values.mean())
    exceedances = 0
    completed = 0
    chunk_size = 10000
    while completed < replicates:
        size = min(chunk_size, replicates - completed)
        signs = PERMUTATION_RNG.choice((-1.0, 1.0), size=(size, len(values)))
        permuted = np.abs((signs * values).mean(axis=1))
        exceedances += int((permuted >= observed).sum())
        completed += size
    return (exceedances + 1) / (replicates + 1)


def load_replicate0_daily() -> pd.DataFrame:
    frames = []
    for path in PRIMARY_RUNS.glob("*/*/*/*_DailySummary.csv"):
        frame = pd.read_csv(path)
        period = str(frame["Period"].iloc[0])
        if period in POLICIES:
            frame["StudyDay"] = np.arange(1, len(frame) + 1)
            frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    daily["PolicyPercent"] = daily["Period"].map(POLICIES)
    daily["Replicate"] = 0
    return daily[daily["Patient"].ne("child#008")]


def load_replicate_daily(replicate: int) -> pd.DataFrame:
    frames = []
    for path in (ROBUST_RUNS / f"replicate{replicate}").glob("*/*/*/*_DailySummary.csv"):
        frame = pd.read_csv(path)
        period = str(frame["Period"].iloc[0])
        if period in POLICIES:
            frame["StudyDay"] = np.arange(1, len(frame) + 1)
            frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    daily["PolicyPercent"] = daily["Period"].map(POLICIES)
    daily["Replicate"] = replicate
    return daily


def load_completion0() -> pd.DataFrame:
    completion = pd.read_csv(PRIMARY_RUNS / "completion.csv")
    completion = completion[completion["Period"].isin(POLICIES)].copy()
    completion["PolicyPercent"] = completion["Period"].map(POLICIES)
    completion["Replicate"] = 0
    return completion[completion["Patient"].ne("child#008")]


def load_completion_replicate(replicate: int) -> pd.DataFrame:
    completion = pd.read_csv(ROBUST_RUNS / "completion_parts" / f"replicate{replicate}.csv")
    completion["PolicyPercent"] = completion["Period"].map(POLICIES)
    return completion


def subject_level(daily: pd.DataFrame, completion: pd.DataFrame) -> pd.DataFrame:
    cohort = daily[daily["StudyDay"].le(30)]
    subject = (
        cohort.groupby(["Patient", "Role", "Replicate", "PolicyPercent"], as_index=False)[METRICS]
        .mean()
    )
    rescue = completion[["Patient", "Replicate", "PolicyPercent", "SafetyCorrections"]].rename(
        columns={"SafetyCorrections": "RescueCorrections"}
    )
    subject = subject.merge(rescue, on=["Patient", "Replicate", "PolicyPercent"], validate="one_to_one")
    return subject


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    daily0 = load_replicate0_daily()
    daily1 = load_replicate_daily(1)
    daily2 = load_replicate_daily(2)
    completion0 = load_completion0()
    completion1 = load_completion_replicate(1)
    completion2 = load_completion_replicate(2)

    daily = pd.concat([daily0, daily1, daily2], ignore_index=True)
    completion = pd.concat([completion0, completion1, completion2], ignore_index=True)

    subject = subject_level(daily, completion)
    n_patients = subject["Patient"].nunique()
    assert n_patients == 29
    assert len(subject) == 29 * 5 * 3

    all_metrics = METRICS + ["RescueCorrections"]

    # --- Seed-averaged subject-level table ---
    seedavg = (
        subject.groupby(["Patient", "Role", "PolicyPercent"], as_index=False)[all_metrics]
        .mean()
    )

    # --- Dose-response summary on seed-averaged data ---
    summary_rows = []
    for policy, group in seedavg.groupby("PolicyPercent"):
        for metric in all_metrics:
            values = group[metric].to_numpy()
            low, high = bootstrap_mean_ci(values)
            summary_rows.append(
                {
                    "PolicyPercent": int(policy),
                    "Metric": metric,
                    "Mean": values.mean(),
                    "SD": values.std(ddof=1),
                    "CI95Low": low,
                    "CI95High": high,
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values(["Metric", "PolicyPercent"], ascending=[True, False])

    # --- Paired differences vs 100% on seed-averaged data ---
    diff_rows = []
    for metric in all_metrics:
        wide = seedavg.pivot(index="Patient", columns="PolicyPercent", values=metric)
        raw_p = []
        rows_for_metric = []
        for policy in [90, 80, 70, 60]:
            difference = (wide[policy] - wide[100]).to_numpy()
            low, high = bootstrap_mean_ci(difference)
            p_value = paired_mean_permutation_p(difference)
            raw_p.append(p_value)
            rows_for_metric.append(
                {
                    "Metric": metric,
                    "Comparison": f"{policy}% minus 100%",
                    "MeanDifference": difference.mean(),
                    "CI95Low": low,
                    "CI95High": high,
                    "RawP": p_value,
                }
            )
        adjusted = multipletests(raw_p, method="holm")[1]
        for row, adjusted_p in zip(rows_for_metric, adjusted):
            row["AdjustedP"] = adjusted_p
            diff_rows.append(row)
    diffs = pd.DataFrame(diff_rows)

    # --- Variance decomposition: between-patient vs within-patient(seed) ---
    decomp_rows = []
    for policy in [100, 90, 80, 70, 60]:
        for metric in all_metrics:
            sub = subject[subject["PolicyPercent"].eq(policy)]
            patient_means = sub.groupby("Patient")[metric].mean()
            between_var = patient_means.var(ddof=1)
            within_var = sub.groupby("Patient")[metric].var(ddof=1).mean()
            total_var = between_var + within_var
            decomp_rows.append(
                {
                    "PolicyPercent": policy,
                    "Metric": metric,
                    "BetweenPatientVar": between_var,
                    "WithinPatientSeedVar": within_var,
                    "WithinShare": within_var / total_var if total_var > 0 else np.nan,
                }
            )
    decomposition = pd.DataFrame(decomp_rows)

    # --- Phenotype correlation on seed-averaged data ---
    # Pre-Ramadan TBR comes from the pre-Ramadan control arm (replicate 0 only, as in primary analysis)
    pre_frames = []
    for path in PRIMARY_RUNS.glob("*/*/*/*_DailySummary.csv"):
        frame = pd.read_csv(path)
        if str(frame["Period"].iloc[0]) == "PreRamadan_100_Control":
            frame["StudyDay"] = np.arange(1, len(frame) + 1)
            pre_frames.append(frame)
    pre_daily = pd.concat(pre_frames, ignore_index=True)
    pre_daily = pre_daily[pre_daily["Patient"].ne("child#008") & pre_daily["StudyDay"].le(30)]
    pre_tbr = pre_daily.groupby("Patient", as_index=False)["TBR"].mean().rename(columns={"TBR": "Pre_TBR"})

    wide_tir = seedavg.pivot(index="Patient", columns="PolicyPercent", values="TIR")
    pheno = pre_tbr.set_index("Patient")
    correlation_rows = []
    for policy in [90, 80, 70, 60]:
        delta = (wide_tir[policy] - wide_tir[100])
        rho, p_value = stats.spearmanr(pheno["Pre_TBR"], delta.reindex(pheno.index))
        correlation_rows.append({"PolicyPercent": policy, "SpearmanRho": rho, "P": p_value})
    correlations = pd.DataFrame(correlation_rows)

    # --- Save outputs ---
    subject.to_csv(TABLES / "robustness_subject_level_by_seed.csv", index=False)
    seedavg.to_csv(TABLES / "robustness_seed_averaged_subject_level.csv", index=False)
    summary.to_csv(TABLES / "robustness_dose_response_seedavg.csv", index=False)
    diffs.to_csv(TABLES / "robustness_paired_diffs_seedavg.csv", index=False)
    decomposition.to_csv(TABLES / "robustness_variance_decomposition.csv", index=False)
    correlations.to_csv(TABLES / "robustness_phenotype_correlations_seedavg.csv", index=False)

    print("=== Seed-averaged dose-response (TIR/TBR/TAR/Rescue) ===")
    print(summary[summary["Metric"].isin(["TIR", "TBR", "TAR", "RescueCorrections"])].to_string(index=False))
    print("\n=== Paired differences vs 100% (seed-averaged) ===")
    print(diffs.to_string(index=False))
    print("\n=== Variance decomposition (within-patient/seed share of total variance) ===")
    print(decomposition[decomposition["Metric"].isin(["TIR", "TBR", "TAR", "RescueCorrections"])].to_string(index=False))
    print("\n=== Phenotype correlation (Pre-Ramadan TBR vs Delta TIR), seed-averaged ===")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
