#!/usr/bin/env python3
"""Run meal/timing stochastic scenarios and a fasting-window-only policy."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SIMULATOR = ROOT / "simulation"
sys.path.insert(0, str(LOCAL_SIMULATOR))

from ramadan_simulator import cohort_parameters, run_period  # noqa: E402


POLICIES = {
    "Ramadan_100_Stochastic": (1.00, False),
    "Ramadan_090_Stochastic": (0.90, False),
    "Ramadan_080_Stochastic": (0.80, False),
    "Ramadan_070_Stochastic": (0.70, False),
    "Ramadan_060_Stochastic": (0.60, False),
    "Ramadan_FastingWindow_080_Stochastic": (0.80, True),
}
BASE_SENSOR_SEED = 20260218
BASE_MEAL_SEED = 20260614


def run_patient_scenario(
    patient: str,
    role: str,
    params_row: pd.Series,
    scenario: int,
    output_root: Path,
    days: int,
) -> list[dict]:
    patient_number = int(patient.split("#")[1])
    sensor_seed = BASE_SENSOR_SEED + patient_number * 10 + scenario * 700_000
    meal_seed = BASE_MEAL_SEED + patient_number * 100 + scenario * 10_000
    records = []
    for period, (factor, fasting_only) in POLICIES.items():
        output_dir = output_root / f"scenario{scenario}" / role / patient / period
        result = run_period(
            patient_id=patient,
            role=role,
            period=period,
            kind="ramadan",
            basal_factor=factor,
            days=days,
            seed=sensor_seed,
            params=params_row,
            output_dir=output_dir,
            correction_strategy="none",
            meal_seed=meal_seed,
            stochastic_meals=True,
            fasting_window_only=fasting_only,
        )
        result["BasalFactor"] = factor
        result["Scenario"] = scenario
        records.append(result)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "raw" / "scenario_extension"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--patients", nargs="*")
    args = parser.parse_args()

    params = cohort_parameters()
    params = params[params["Patient"].ne("child#008")]
    if args.patients:
        params = params[params["Patient"].isin(args.patients)]

    completion_records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for scenario in args.scenarios:
            for _, row in params.iterrows():
                future = executor.submit(
                    run_patient_scenario,
                    row["Patient"],
                    row["Role"],
                    row,
                    scenario,
                    args.output,
                    args.days,
                )
                futures[future] = (row["Patient"], scenario)
        for future in as_completed(futures):
            patient, scenario = futures[future]
            try:
                results = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {patient} scenario {scenario}: {exc}", flush=True)
                continue
            completion_records.extend(results)
            print(f"done {patient} scenario {scenario}", flush=True)

    completion = pd.DataFrame(completion_records)
    completion_dir = args.output / "completion_parts"
    completion_dir.mkdir(parents=True, exist_ok=True)
    for scenario, group in completion.groupby("Scenario"):
        group.to_csv(completion_dir / f"scenario{scenario}.csv", index=False)
    print(json.dumps({"n_results": len(completion_records)}))


if __name__ == "__main__":
    main()
