#!/usr/bin/env python3
"""Analyze stochastic meal/timing scenarios and fasting-window policy."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "raw" / "scenario_extension"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
PRE_SOURCE = ROOT / "data" / "input" / "pre_ramadan_tbr.csv"
RNG = np.random.default_rng(20260614)
PERM_RNG = np.random.default_rng(20260615)

POLICIES = {
    "Ramadan_100_Stochastic": "100%",
    "Ramadan_090_Stochastic": "90%",
    "Ramadan_080_Stochastic": "80%",
    "Ramadan_070_Stochastic": "70%",
    "Ramadan_060_Stochastic": "60%",
    "Ramadan_FastingWindow_080_Stochastic": "Fasting-window 80%",
}
FULL_POLICIES = ["100%", "90%", "80%", "70%", "60%"]
METRICS = ["TIR", "TBR", "TBR_<54", "TAR"]


def bootstrap_ci(values: np.ndarray, replicates: int = 20000) -> tuple[float, float]:
    draws = RNG.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def permutation_p(values: np.ndarray, replicates: int = 200000) -> float:
    observed = abs(values.mean())
    exceedances = 0
    for _ in range(0, replicates, 10000):
        signs = PERM_RNG.choice((-1.0, 1.0), size=(10000, len(values)))
        exceedances += int((np.abs((signs * values).mean(axis=1)) >= observed).sum())
    return (exceedances + 1) / (replicates + 1)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in RUNS.glob("scenario*/*/*/*/*_DailySummary.csv"):
        frame = pd.read_csv(path)
        frame["Scenario"] = int(path.parts[-5].replace("scenario", ""))
        frame["Policy"] = frame["Period"].map(POLICIES)
        frame["StudyDay"] = np.arange(1, len(frame) + 1)
        frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    completions = pd.concat(
        [pd.read_csv(path) for path in (RUNS / "completion_parts").glob("scenario*.csv")],
        ignore_index=True,
    )
    completions["Policy"] = completions["Period"].map(POLICIES)
    return daily, completions


def subject_level(daily: pd.DataFrame, completions: pd.DataFrame) -> pd.DataFrame:
    complete_keys = completions[
        (~completions["Terminal"]) & completions["DaysObserved"].eq(30)
    ][["Patient", "Scenario", "Policy"]]
    cohort = daily.merge(complete_keys, on=["Patient", "Scenario", "Policy"], how="inner")
    subject = (
        cohort.groupby(["Patient", "Role", "Scenario", "Policy"], as_index=False)[METRICS]
        .mean()
    )
    rescue = completions.rename(columns={"SafetyCorrections": "RescueCorrections"})[
        ["Patient", "Scenario", "Policy", "RescueCorrections"]
    ]
    subject = subject.merge(rescue, on=["Patient", "Scenario", "Policy"], validate="one_to_one")
    return subject


def paired_complete_scenarios(subject: pd.DataFrame) -> pd.DataFrame:
    complete_blocks = (
        subject.groupby(["Patient", "Scenario"])
        .size()
        .loc[lambda values: values.eq(len(POLICIES))]
        .index
    )
    index = pd.MultiIndex.from_frame(subject[["Patient", "Scenario"]])
    return subject[index.isin(complete_blocks)].copy()


def scenario_dose_response(subject: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    averaged = (
        subject.groupby(["Patient", "Role", "Policy"], as_index=False)[
            METRICS + ["RescueCorrections"]
        ]
        .mean()
    )
    rows = []
    for policy, group in averaged.groupby("Policy"):
        for metric in METRICS + ["RescueCorrections"]:
            values = group[metric].to_numpy()
            low, high = bootstrap_ci(values)
            rows.append(
                {
                    "Policy": policy,
                    "Metric": metric,
                    "Mean": values.mean(),
                    "SD": values.std(ddof=1),
                    "CI95Low": low,
                    "CI95High": high,
                    "N": len(values),
                }
            )
    summary = pd.DataFrame(rows)

    contrasts = []
    for metric in METRICS + ["RescueCorrections"]:
        wide = averaged.pivot(index="Patient", columns="Policy", values=metric)
        comparisons = [
            ("90% minus 100%", "90%", "100%"),
            ("80% minus 100%", "80%", "100%"),
            ("70% minus 100%", "70%", "100%"),
            ("60% minus 100%", "60%", "100%"),
            ("Fasting-window 80% minus 100%", "Fasting-window 80%", "100%"),
            ("Fasting-window 80% minus full-day 80%", "Fasting-window 80%", "80%"),
        ]
        metric_rows = []
        for label, left, right in comparisons:
            valid = wide[[left, right]].dropna()
            difference = (valid[left] - valid[right]).to_numpy()
            low, high = bootstrap_ci(difference)
            metric_rows.append(
                {
                    "Metric": metric,
                    "Comparison": label,
                    "N": len(difference),
                    "MeanDifference": difference.mean(),
                    "CI95Low": low,
                    "CI95High": high,
                    "RawP": permutation_p(difference),
                }
            )
        adjusted = multipletests([row["RawP"] for row in metric_rows], method="holm")[1]
        for row, p_value in zip(metric_rows, adjusted):
            row["AdjustedP"] = p_value
            contrasts.append(row)
    return summary, pd.DataFrame(contrasts)


def phenotype_by_scenario(subject: pd.DataFrame) -> pd.DataFrame:
    pre = pd.read_csv(PRE_SOURCE)[["Patient", "PreTBR"]]
    rows = []
    for scenario, group in subject.groupby("Scenario"):
        wide = group.pivot(index="Patient", columns="Policy", values="TIR")
        joined = pre.set_index("Patient").join(wide)
        for policy in ["90%", "80%", "70%", "60%", "Fasting-window 80%"]:
            delta = joined[policy] - joined["100%"]
            valid = joined["PreTBR"].notna() & delta.notna()
            rho, p_value = stats.spearmanr(joined.loc[valid, "PreTBR"], delta[valid])
            rows.append(
                {
                    "Scenario": scenario,
                    "Policy": policy,
                    "N": int(valid.sum()),
                    "SpearmanRho": rho,
                    "P": p_value,
                }
            )
    return pd.DataFrame(rows)


def heldout_policy_validation(
    subject: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training = subject[subject["Scenario"].isin([1, 2])]
    training = (
        training[training["Policy"].isin(FULL_POLICIES)]
        .groupby(["Patient", "Role", "Policy"], as_index=False)[
            METRICS + ["RescueCorrections"]
        ]
        .mean()
    )
    selections = []
    for patient, group in training.groupby("Patient"):
        eligible = group[
            (group["TBR"] <= 4)
            & (group["TBR_<54"] <= 1)
            & (group["TAR"] <= 25)
        ]
        pool = eligible if not eligible.empty else group
        selected = pool.sort_values(["TIR", "Policy"], ascending=[False, False]).iloc[0]
        selections.append(
            {
                "Patient": patient,
                "Role": selected["Role"],
                "SelectedPolicy": selected["Policy"],
                "TrainingMetAllTargets": not eligible.empty,
            }
        )
    selections = pd.DataFrame(selections)

    heldout = subject[subject["Scenario"].eq(3)].merge(
        selections, on=["Patient", "Role"], how="inner"
    )
    chosen = heldout[heldout["Policy"].eq(heldout["SelectedPolicy"])].copy()
    baseline = heldout[heldout["Policy"].eq("100%")][
        ["Patient", "TIR", "TBR", "TBR_<54", "TAR", "RescueCorrections"]
    ].rename(columns={metric: f"Baseline_{metric}" for metric in METRICS + ["RescueCorrections"]})
    validation = chosen.merge(baseline, on="Patient", validate="one_to_one")
    for metric in METRICS + ["RescueCorrections"]:
        validation[f"Delta_{metric}"] = validation[metric] - validation[f"Baseline_{metric}"]

    summary_rows = []
    for metric in METRICS + ["RescueCorrections"]:
        difference = validation[f"Delta_{metric}"].to_numpy()
        low, high = bootstrap_ci(difference)
        summary_rows.append(
            {
                "Metric": metric,
                "N": len(difference),
                "MeanDifference": difference.mean(),
                "CI95Low": low,
                "CI95High": high,
                "RawP": permutation_p(difference),
                "SelectedTargetAttainment": (
                    validation["TBR"].le(4)
                    & validation["TBR_<54"].le(1)
                    & validation["TAR"].le(25)
                ).mean(),
                "BaselineTargetAttainment": (
                    validation["Baseline_TBR"].le(4)
                    & validation["Baseline_TBR_<54"].le(1)
                    & validation["Baseline_TAR"].le(25)
                ).mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["AdjustedP"] = multipletests(summary["RawP"], method="holm")[1]
    return selections, validation, summary


def plot_extension(summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    tir = summary[summary["Metric"].eq("TIR")].set_index("Policy")
    order = ["100%", "90%", "80%", "70%", "60%"]
    axes[0].errorbar(
        [100, 90, 80, 70, 60],
        tir.loc[order, "Mean"],
        yerr=np.vstack(
            [
                tir.loc[order, "Mean"] - tir.loc[order, "CI95Low"],
                tir.loc[order, "CI95High"] - tir.loc[order, "Mean"],
            ]
        ),
        marker="o",
        capsize=4,
        linewidth=2.5,
        color="#0878b9",
    )
    axes[0].invert_xaxis()
    axes[0].set(
        xlabel="Full-day basal insulin delivered, %",
        ylabel="TIR, %",
        title="Independent meal/timing scenarios",
    )

    metrics = ["TIR", "TBR", "TAR"]
    labels = ["TIR", "TBR", "TAR"]
    frame = contrasts[
        contrasts["Comparison"].eq("Fasting-window 80% minus full-day 80%")
        & contrasts["Metric"].isin(metrics)
    ].set_index("Metric").loc[metrics]
    axes[1].bar(labels, frame["MeanDifference"], color="#5b8ff9")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(
        ylabel="Mean difference, percentage points",
        title="Fasting-window 80% minus full-day 80%",
    )
    figure.savefig(FIGURES / "figure_scenario_extension.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    daily, completions = load_data()
    subject = paired_complete_scenarios(subject_level(daily, completions))
    summary, contrasts = scenario_dose_response(subject)
    phenotype = phenotype_by_scenario(subject)
    selections, validation, validation_summary = heldout_policy_validation(subject)

    completions.to_csv(TABLES / "scenario_extension_completion.csv", index=False)
    subject.to_csv(TABLES / "scenario_extension_subject_level.csv", index=False)
    summary.to_csv(TABLES / "scenario_extension_summary.csv", index=False)
    contrasts.to_csv(TABLES / "scenario_extension_contrasts.csv", index=False)
    phenotype.to_csv(TABLES / "scenario_extension_phenotype.csv", index=False)
    selections.to_csv(TABLES / "scenario_extension_training_selections.csv", index=False)
    validation.to_csv(TABLES / "scenario_extension_heldout_validation.csv", index=False)
    validation_summary.to_csv(
        TABLES / "scenario_extension_heldout_summary.csv", index=False
    )
    plot_extension(summary, contrasts)

    print("Completion:", completions.groupby(["Scenario", "Terminal"]).size().to_dict())
    print("\nKey contrasts:")
    print(
        contrasts[
            contrasts["Comparison"].isin(
                [
                    "Fasting-window 80% minus 100%",
                    "Fasting-window 80% minus full-day 80%",
                ]
            )
        ].to_string(index=False)
    )
    print("\nPhenotype by scenario:")
    print(phenotype.to_string(index=False))
    print("\nHeld-out mean differences:")
    print(validation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
