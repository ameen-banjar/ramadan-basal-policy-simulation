#!/usr/bin/env python3
"""Check the bundled derived results without regenerating large raw outputs."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"


def close(actual: float, expected: float, tolerance: float = 0.06) -> None:
    if not np.isclose(actual, expected, atol=tolerance):
        raise AssertionError(f"Expected {expected}, found {actual}")


def main() -> None:
    primary = pd.read_csv(DATA / "five_policy_dose_response.csv")
    inference = pd.read_csv(DATA / "five_policy_inference.csv")
    scenario = pd.read_csv(DATA / "scenario_extension_contrasts.csv")
    heldout = pd.read_csv(DATA / "scenario_extension_heldout_summary.csv")

    tir = primary[primary["Metric"].eq("TIR")].set_index("PolicyPercent")["Mean"]
    for policy, expected in {100: 81.7, 90: 79.9, 80: 73.9, 70: 65.3, 60: 56.0}.items():
        close(tir.loc[policy], expected)

    primary_90 = inference[
        inference["Metric"].eq("TIR")
        & inference["Comparison"].eq("90% minus 100%")
    ].iloc[0]
    close(primary_90["MeanDifference"], -1.8)
    close(primary_90["AdjustedP"], 0.031, tolerance=0.002)

    fasting = scenario[
        scenario["Comparison"].eq("Fasting-window 80% minus full-day 80%")
    ].set_index("Metric")
    close(fasting.loc["TIR", "MeanDifference"], 0.9)
    close(fasting.loc["TBR", "MeanDifference"], 1.1)
    close(fasting.loc["TAR", "MeanDifference"], -2.0)

    heldout = heldout.set_index("Metric")
    close(heldout.loc["TIR", "MeanDifference"], 0.6)
    close(heldout.loc["TBR", "MeanDifference"], -1.6)
    print("Derived-result checks passed.")


if __name__ == "__main__":
    main()
