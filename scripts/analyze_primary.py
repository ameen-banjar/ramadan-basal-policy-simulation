#!/usr/bin/env python3
"""Five-policy dose-response analysis for the Ramadan virtual-patient study."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "raw" / "primary"
FIGURES = ROOT / "outputs" / "figures"
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
METRICS = ["Mean", "CV", "GMI", "TIR", "TBR", "TBR_<54", "TAR", "LBGI", "HBGI"]


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


def load_daily_files() -> pd.DataFrame:
    frames = []
    files = list(RUNS.glob("*/*/*/*_DailySummary.csv"))
    for path in files:
        frame = pd.read_csv(path)
        period = str(frame["Period"].iloc[0])
        if period in POLICIES or period == "PreRamadan_100_Control":
            frame["StudyDay"] = np.arange(1, len(frame) + 1)
            frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    daily["PolicyPercent"] = daily["Period"].map(POLICIES)
    return daily


def load_completion() -> pd.DataFrame:
    completion = pd.read_csv(RUNS / "completion.csv")
    completion = completion[
        completion["Period"].isin(POLICIES) | completion["Period"].eq("PreRamadan_100_Control")
    ].copy()
    completion["PolicyPercent"] = completion["Period"].map(POLICIES)
    return completion


def subject_level(daily: pd.DataFrame, completion: pd.DataFrame) -> pd.DataFrame:
    cohort = daily[
        daily["Period"].isin(POLICIES)
        & daily["Patient"].ne("child#008")
        & daily["StudyDay"].le(30)
    ]
    subject = (
        cohort.groupby(["Patient", "Role", "Period", "PolicyPercent"], as_index=False)[METRICS]
        .mean()
    )
    rescue = completion[
        completion["Period"].isin(POLICIES) & completion["Patient"].ne("child#008")
    ][["Patient", "Period", "SafetyCorrections"]].rename(
        columns={"SafetyCorrections": "RescueCorrections"}
    )
    subject = subject.merge(rescue, on=["Patient", "Period"], validate="one_to_one")
    counts = subject.groupby("Patient")["PolicyPercent"].nunique()
    assert len(subject) == 29 * 5 and counts.eq(5).all()
    return subject


def dose_response_summary(subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in subject.groupby("PolicyPercent", sort=False):
        for metric in METRICS + ["RescueCorrections"]:
            values = group[metric].to_numpy()
            low, high = bootstrap_mean_ci(values)
            rows.append(
                {
                    "PolicyPercent": int(policy),
                    "Metric": metric,
                    "Mean": values.mean(),
                    "SD": values.std(ddof=1),
                    "CI95Low": low,
                    "CI95High": high,
                }
            )
    return pd.DataFrame(rows).sort_values(["Metric", "PolicyPercent"], ascending=[True, False])


def repeated_measures_tests(subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS + ["RescueCorrections"]:
        wide = subject.pivot(index="Patient", columns="PolicyPercent", values=metric)
        friedman = stats.friedmanchisquare(*(wide[p].to_numpy() for p in [100, 90, 80, 70, 60]))
        rows.append(
            {
                "Metric": metric,
                "Test": "Friedman across five policies",
                "Comparison": "100, 90, 80, 70, 60",
                "MeanDifference": np.nan,
                "CI95Low": np.nan,
                "CI95High": np.nan,
                "RawP": friedman.pvalue,
                "AdjustedP": friedman.pvalue,
            }
        )
        pair_rows = []
        for policy in [90, 80, 70, 60]:
            difference = (wide[policy] - wide[100]).to_numpy()
            low, high = bootstrap_mean_ci(difference)
            pair_rows.append(
                {
                    "Metric": metric,
                    "Test": "Paired mean permutation vs 100%",
                    "Comparison": f"{policy}% minus 100%",
                    "MeanDifference": difference.mean(),
                    "CI95Low": low,
                    "CI95High": high,
                    "RawP": paired_mean_permutation_p(difference),
                }
            )
        adjusted = multipletests([row["RawP"] for row in pair_rows], method="holm")[1]
        for row, adjusted_p in zip(pair_rows, adjusted):
            row["AdjustedP"] = adjusted_p
            rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_all_30(daily: pd.DataFrame) -> pd.DataFrame:
    subset = daily[daily["Period"].isin(POLICIES) & daily["StudyDay"].le(9)]
    subject = (
        subset.groupby(["Patient", "Role", "Period", "PolicyPercent"], as_index=False)[METRICS]
        .mean()
    )
    assert len(subject) == 30 * 5
    rows = []
    for metric in ["TIR", "TBR", "TBR_<54", "TAR"]:
        wide = subject.pivot(index="Patient", columns="PolicyPercent", values=metric)
        for policy in [90, 80, 70, 60]:
            difference = (wide[policy] - wide[100]).to_numpy()
            low, high = bootstrap_mean_ci(difference)
            rows.append(
                {
                    "Metric": metric,
                    "Comparison": f"{policy}% minus 100%",
                    "MeanDifference": difference.mean(),
                    "CI95Low": low,
                    "CI95High": high,
                }
            )
    return pd.DataFrame(rows)


def phenotype_analysis(
    daily: pd.DataFrame, subject: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre = (
        daily[
            daily["Period"].eq("PreRamadan_100_Control")
            & daily["Patient"].ne("child#008")
            & daily["StudyDay"].le(30)
        ]
        .groupby(["Patient", "Role"], as_index=False)[METRICS]
        .mean()
        .rename(columns={metric: f"Pre_{metric}" for metric in METRICS})
    )
    wide = subject.pivot(index=["Patient", "Role"], columns="PolicyPercent")
    patient = pre.copy()
    for policy in [90, 80, 70, 60]:
        for metric in ["TIR", "TBR", "TAR"]:
            patient[f"Delta_{metric}_{policy}"] = (
                wide[(metric, policy)] - wide[(metric, 100)]
            ).to_numpy()
    patient["PreTBRGroup"] = np.where(patient["Pre_TBR"] >= 4, "TBR >=4%", "TBR <4%")

    correlation_rows = []
    for policy in [90, 80, 70, 60]:
        for feature in ["Pre_TBR", "Pre_TIR", "Pre_LBGI", "Pre_CV"]:
            rho, p_value = stats.spearmanr(patient[feature], patient[f"Delta_TIR_{policy}"])
            correlation_rows.append(
                {
                    "PolicyPercent": policy,
                    "Feature": feature,
                    "SpearmanRho": rho,
                    "P": p_value,
                }
            )
    correlations = pd.DataFrame(correlation_rows)

    eligibility = subject.assign(
        Eligible=lambda x: (x["TBR"] <= 4.0) & (x["TBR_<54"] <= 1.0)
    )
    selections = []
    for patient_id, group in eligibility.groupby("Patient"):
        eligible = group[group["Eligible"]]
        pool = eligible if not eligible.empty else group
        selected = pool.sort_values(["TIR", "PolicyPercent"], ascending=[False, False]).iloc[0]
        selections.append(
            {
                "Patient": patient_id,
                "Role": selected["Role"],
                "PreTBR": patient.loc[patient["Patient"].eq(patient_id), "Pre_TBR"].iloc[0],
                "SelectedPolicyPercent": int(selected["PolicyPercent"]),
                "SelectedTIR": selected["TIR"],
                "SelectedTBR": selected["TBR"],
                "AnyPolicyMetHypoglycemiaTargets": not eligible.empty,
            }
        )
    selection = pd.DataFrame(selections)
    return patient, correlations, selection


def plot_dose_response(summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    specifications = [
        ("TIR", "TIR, %", "#0878b9"),
        ("TBR", "TBR <70 mg/dL, %", "#d55e00"),
        ("TAR", "TAR >180 mg/dL, %", "#cc79a7"),
        ("RescueCorrections", "Rescue corrections per 30 days", "#009e73"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for axis, (metric, ylabel, color) in zip(axes.flat, specifications):
        frame = summary[summary["Metric"].eq(metric)].sort_values("PolicyPercent")
        yerr = np.vstack(
            [frame["Mean"] - frame["CI95Low"], frame["CI95High"] - frame["Mean"]]
        )
        axis.errorbar(
            frame["PolicyPercent"],
            frame["Mean"],
            yerr=yerr,
            marker="o",
            markersize=8,
            linewidth=2.5,
            capsize=4,
            color=color,
        )
        axis.set(xlabel="Basal insulin delivered, %", ylabel=ylabel)
        axis.set_xticks([60, 70, 80, 90, 100])
    figure.suptitle("Thirty-day dose-response in 29 complete virtual patients", y=1.01)
    figure.tight_layout()
    figure.savefig(FIGURES / "figure_dose_response.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_phenotype_policy(patient: pd.DataFrame, selection: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), constrained_layout=True)
    colors = {90: "#56b4e9", 80: "#009e73", 70: "#e69f00", 60: "#d55e00"}
    for policy in [90, 80, 70, 60]:
        axes[0].scatter(
            patient["Pre_TBR"],
            patient[f"Delta_TIR_{policy}"],
            label=f"{policy}%",
            color=colors[policy],
            alpha=0.78,
            s=45,
        )
        slope, intercept = np.polyfit(patient["Pre_TBR"], patient[f"Delta_TIR_{policy}"], 1)
        x_values = np.linspace(patient["Pre_TBR"].min(), patient["Pre_TBR"].max(), 100)
        axes[0].plot(x_values, slope * x_values + intercept, color=colors[policy], linewidth=1.5)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].axvline(4, color="#666666", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Pre-Ramadan TBR, %",
        ylabel="TIR difference vs 100%, points",
        title="Phenotype and policy response",
    )
    axes[0].legend(title="Basal policy", frameon=False, ncol=2)

    counts = (
        selection["SelectedPolicyPercent"]
        .value_counts()
        .reindex([100, 90, 80, 70, 60], fill_value=0)
    )
    axes[1].bar(counts.index.astype(str) + "%", counts.values, color="#276f9b")
    axes[1].set(
        xlabel="Selected basal policy",
        ylabel="Number of virtual patients",
        title="Exploratory hypoglycemia-target selection",
    )
    for index, value in enumerate(counts.values):
        axes[1].text(index, value + 0.25, str(value), ha="center", fontsize=11)
    figure.savefig(FIGURES / "figure_phenotype_policy.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    completion: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    sensitivity: pd.DataFrame,
    correlations: pd.DataFrame,
    selection: pd.DataFrame,
) -> None:
    key_metrics = summary[summary["Metric"].isin(["TIR", "TBR", "TAR", "RescueCorrections"])]
    pairwise = tests[tests["Test"].eq("Paired mean permutation vs 100%")]
    terminal = completion[completion["Terminal"] & completion["Period"].isin(POLICIES)]
    lines = [
        "PRIMARY_COHORT_N=29",
        "PRIMARY_HORIZON_DAYS=30",
        "SENSITIVITY_COHORT_N=30",
        "SENSITIVITY_HORIZON_DAYS=9",
        f"TERMINAL_POLICY_RUNS={len(terminal)}",
        f"SELECTED_POLICY_COUNTS={selection['SelectedPolicyPercent'].value_counts().sort_index().to_dict()}",
    ]
    for policy in [100, 90, 80, 70, 60]:
        for metric in ["TIR", "TBR", "TAR", "RescueCorrections"]:
            row = key_metrics[
                key_metrics["PolicyPercent"].eq(policy) & key_metrics["Metric"].eq(metric)
            ].iloc[0]
            lines.append(f"{metric}_{policy}_MEAN={row['Mean']:.3f}")
    for policy in [90, 80, 70, 60]:
        for metric in ["TIR", "TBR", "TAR", "RescueCorrections"]:
            row = pairwise[
                pairwise["Metric"].eq(metric)
                & pairwise["Comparison"].eq(f"{policy}% minus 100%")
            ].iloc[0]
            lines.append(
                f"DELTA_{metric}_{policy}={row['MeanDifference']:.3f} "
                f"[{row['CI95Low']:.3f},{row['CI95High']:.3f}] "
                f"P_HOLM={row['AdjustedP']:.6g}"
            )
    for policy in [90, 80, 70, 60]:
        row = correlations[
            correlations["PolicyPercent"].eq(policy)
            & correlations["Feature"].eq("Pre_TBR")
        ].iloc[0]
        lines.append(
            f"PRE_TBR_VS_DELTA_TIR_{policy}_RHO={row['SpearmanRho']:.3f} P={row['P']:.6g}"
        )
    lines.append("TERMINAL_EVENTS:")
    lines.extend(terminal.to_string(index=False).splitlines())
    lines.append("NINE_DAY_SENSITIVITY:")
    lines.extend(sensitivity.to_string(index=False).splitlines())
    (TABLES / "multi_policy_summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    daily = load_daily_files()
    completion = load_completion()
    subject = subject_level(daily, completion)
    summary = dose_response_summary(subject)
    tests = repeated_measures_tests(subject)
    sensitivity = sensitivity_all_30(daily)
    patient, correlations, selection = phenotype_analysis(daily, subject)

    subject.to_csv(TABLES / "five_policy_subject_level.csv", index=False)
    summary.to_csv(TABLES / "five_policy_dose_response.csv", index=False)
    tests.to_csv(TABLES / "five_policy_inference.csv", index=False)
    sensitivity.to_csv(TABLES / "five_policy_sensitivity_9day.csv", index=False)
    patient.to_csv(TABLES / "five_policy_phenotype_response.csv", index=False)
    correlations.to_csv(TABLES / "five_policy_phenotype_correlations.csv", index=False)
    selection.to_csv(TABLES / "five_policy_selection.csv", index=False)

    plot_dose_response(summary)
    plot_phenotype_policy(patient, selection)
    write_summary(completion, summary, tests, sensitivity, correlations, selection)
    print((TABLES / "multi_policy_summary.txt").read_text())


if __name__ == "__main__":
    main()
