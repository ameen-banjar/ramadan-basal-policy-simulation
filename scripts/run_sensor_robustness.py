#!/usr/bin/env python3
"""Run additional sensor-seed replicates for the five-policy dose-response study.

Replicate 0 corresponds to the existing primary results (already computed).
This script produces replicates 1 and 2 using independent CGM sensor seeds,
for all 29 complete-case virtual patients (child#008 excluded, matching the
primary analysis), across all five basal policies.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from ramadan_simulator import cohort_parameters, run_period  # noqa: E402


POLICIES = {
    "Ramadan_100_NoIntervention": 1.00,
    "Ramadan_090_Intervention": 0.90,
    "Ramadan_080_Intervention": 0.80,
    "Ramadan_070_Intervention": 0.70,
    "Ramadan_060_Intervention": 0.60,
}

REPLICATE_SEED_OFFSET = 500_000
BASE_SEED = 20260218
DAYS = 30


def run_patient_replicate(patient: str, role: str, params_row: pd.Series, replicate: int, output_root: Path) -> list[dict]:
    patient_number = int(patient.split("#")[1])
    seed = BASE_SEED + patient_number * 10 + replicate * REPLICATE_SEED_OFFSET
    results = []
    for period, factor in POLICIES.items():
        output_dir = output_root / f"replicate{replicate}" / role / patient / period
        result = run_period(
            patient_id=patient,
            role=role,
            period=period,
            kind="ramadan",
            basal_factor=factor,
            days=DAYS,
            seed=seed,
            params=params_row,
            output_dir=output_dir,
            correction_strategy="none",
        )
        result["BasalFactor"] = factor
        result["Replicate"] = replicate
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "raw" / "sensor_robustness"
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    params = cohort_parameters()
    params = params[params["Patient"] != "child#008"]

    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for replicate in args.replicates:
            for _, row in params.iterrows():
                fut = executor.submit(run_patient_replicate, row["Patient"], row["Role"], row, replicate, args.output)
                futures[fut] = (row["Patient"], replicate)
        completion_records = []
        for fut in as_completed(futures):
            patient, replicate = futures[fut]
            try:
                results = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {patient} replicate {replicate}: {exc}", flush=True)
                continue
            completion_records.extend(results)
            print(f"done {patient} replicate {replicate}", flush=True)

    completion = pd.DataFrame(completion_records)
    completion_dir = args.output / "completion_parts"
    completion_dir.mkdir(parents=True, exist_ok=True)
    for replicate, group in completion.groupby("Replicate"):
        group.to_csv(completion_dir / f"replicate{replicate}.csv", index=False)
    print(json.dumps({"n_results": len(completion_records)}))


if __name__ == "__main__":
    main()
