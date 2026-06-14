#!/usr/bin/env python3
"""Run the paired pre-Ramadan reference and five Ramadan basal policies."""

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
    "PreRamadan_100_Control": ("pre", 1.00),
    "Ramadan_100_NoIntervention": ("ramadan", 1.00),
    "Ramadan_090_Intervention": ("ramadan", 0.90),
    "Ramadan_080_Intervention": ("ramadan", 0.80),
    "Ramadan_070_Intervention": ("ramadan", 0.70),
    "Ramadan_060_Intervention": ("ramadan", 0.60),
}
BASE_SEED = 20260218


def run_patient(
    patient: str,
    role: str,
    params_row: pd.Series,
    output_root: Path,
    days: int,
) -> list[dict]:
    patient_number = int(patient.split("#")[1])
    seed = BASE_SEED + patient_number * 10
    records = []
    for period, (kind, factor) in POLICIES.items():
        output_dir = output_root / role / patient / period
        result = run_period(
            patient_id=patient,
            role=role,
            period=period,
            kind=kind,
            basal_factor=factor,
            days=days,
            seed=seed,
            params=params_row,
            output_dir=output_dir,
            correction_strategy="none",
        )
        result["BasalFactor"] = factor
        records.append(result)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "raw" / "primary")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--patients", nargs="*")
    args = parser.parse_args()

    params = cohort_parameters()
    if args.patients:
        params = params[params["Patient"].isin(args.patients)]

    completion_records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_patient,
                row["Patient"],
                row["Role"],
                row,
                args.output,
                args.days,
            ): row["Patient"]
            for _, row in params.iterrows()
        }
        for future in as_completed(futures):
            patient = futures[future]
            try:
                completion_records.extend(future.result())
                print(f"done {patient}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {patient}: {exc}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(completion_records).to_csv(args.output / "completion.csv", index=False)
    print(json.dumps({"n_results": len(completion_records)}))


if __name__ == "__main__":
    main()
