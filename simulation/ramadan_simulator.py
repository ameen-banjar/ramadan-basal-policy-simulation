#!/usr/bin/env python3
"""Reproducible three-arm Ramadan simulation for virtual T1D cohorts.

The script uses the 30 UVA/Padova virtual patients distributed with
``simglucose``. It fixes a controller error in the exploratory notebooks:
correction insulin is delivered once at meal onset rather than repeatedly
throughout every meal sample.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import CONTROL_QUEST
from simglucose.controller.base import Action as ControllerAction
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv


BASELINE_CHO = {"child": 150.0, "adolescent": 220.0, "adult": 250.0}
RAMADAN_CHO_FACTORS = (0.80, 1.10, 0.90, 1.25, 0.70, 1.00, 1.15, 0.85, 1.30, 0.95)
RAMADAN_SPLIT = {"suhoor": 0.35, "iftar": 0.50, "snack": 0.15}
PRE_RAMADAN_SPLIT = {"breakfast": 0.30, "lunch": 0.35, "dinner": 0.30, "snack": 0.05}
PERIODS = (
    ("PreRamadan_100_Control", "pre", 1.00),
    ("Ramadan_100_NoIntervention", "ramadan", 1.00),
    ("Ramadan_070_Intervention", "ramadan", 0.70),
)


@dataclass(frozen=True)
class Meal:
    start: dt.datetime
    end: dt.datetime
    grams: float
    name: str


class MealScenario:
    def __init__(self, meals: list[Meal]):
        self.meals = meals
        self.start_time = min(meal.start for meal in meals).replace(hour=0, minute=0)

    def reset(self):
        return None

    def meal_at(self, when: dt.datetime) -> Meal | None:
        for meal in self.meals:
            if meal.start <= when < meal.end:
                return meal
        return None

    def get_action(self, when: dt.datetime):
        meal = self.meal_at(when)
        if meal is None:
            return SimpleNamespace(meal=0.0)
        duration_minutes = (meal.end - meal.start).total_seconds() / 60.0
        return SimpleNamespace(meal=meal.grams / duration_minutes)


def patient_ids() -> dict[str, list[str]]:
    return {
        role: [f"{role}#{index:03d}" for index in range(1, 11)]
        for role in ("child", "adolescent", "adult")
    }


def build_meals(
    kind: str,
    role: str,
    start: dt.datetime,
    days: int,
    *,
    meal_seed: int | None = None,
    stochastic_meals: bool = False,
) -> list[Meal]:
    rng = np.random.default_rng(meal_seed)
    meals: list[Meal] = []
    for day_index in range(days):
        date = (start + dt.timedelta(days=day_index)).date()
        if kind == "pre":
            total = BASELINE_CHO[role]
            specs = (
                ("breakfast", dt.time(8, 0), 30, PRE_RAMADAN_SPLIT["breakfast"]),
                ("lunch", dt.time(13, 0), 30, PRE_RAMADAN_SPLIT["lunch"]),
                ("snack", dt.time(15, 30), 20, PRE_RAMADAN_SPLIT["snack"]),
                ("dinner", dt.time(19, 0), 30, PRE_RAMADAN_SPLIT["dinner"]),
            )
        else:
            total = BASELINE_CHO[role] * RAMADAN_CHO_FACTORS[(day_index // 3) % 10]
            suhoor_offset = (60, 120, 180)[(day_index // 10) % 3]
            suhoor_start = dt.datetime.combine(date, dt.time(5, 30)) - dt.timedelta(
                minutes=suhoor_offset + 60
            )
            iftar_start = dt.datetime.combine(date, dt.time(18, 30))
            snack_start = dt.datetime.combine(date, dt.time(22, 0))
            if stochastic_meals:
                # Independent daily scenario uncertainty, paired across policies.
                total *= float(np.clip(rng.lognormal(mean=-0.5 * 0.18**2, sigma=0.18), 0.65, 1.45))
                suhoor_start += dt.timedelta(minutes=int(rng.integers(-45, 46)))
                iftar_start += dt.timedelta(minutes=int(rng.integers(-20, 21)))
                snack_start += dt.timedelta(minutes=int(rng.integers(-45, 46)))
                split = rng.dirichlet(np.asarray([10.5, 15.0, 4.5]))
            else:
                split = np.asarray(
                    [RAMADAN_SPLIT["suhoor"], RAMADAN_SPLIT["iftar"], RAMADAN_SPLIT["snack"]]
                )
            specs = (
                ("suhoor", suhoor_start.time(), 60, split[0]),
                ("iftar_fast", iftar_start.time(), 15, split[1] * 0.20),
                (
                    "iftar_main",
                    (iftar_start + dt.timedelta(minutes=15)).time(),
                    45,
                    split[1] * 0.80,
                ),
                ("snack", snack_start.time(), 40, split[2]),
            )
        for name, start_time, duration, fraction in specs:
            meal_start = dt.datetime.combine(date, start_time)
            meals.append(
                Meal(
                    start=meal_start,
                    end=meal_start + dt.timedelta(minutes=duration),
                    grams=total * fraction,
                    name=name,
                )
            )
    return meals


def cohort_parameters() -> pd.DataFrame:
    quest = pd.read_csv(CONTROL_QUEST).set_index("Name")
    records = []
    for role, ids in patient_ids().items():
        for patient_id in ids:
            patient = T1DPatient.withName(patient_id)
            basal_u_per_min = float(patient._params.u2ss * patient._params.BW / 6000.0)
            records.append(
                {
                    "Patient": patient_id,
                    "Role": role,
                    "Basal_U_per_min": basal_u_per_min,
                    "ICR_g_per_U": float(quest.loc[patient_id, "CR"]),
                    "ISF_mg_dL_per_U": float(quest.loc[patient_id, "CF"]),
                    "Age_years": float(quest.loc[patient_id, "Age"]),
                    "TDI_U_per_day": float(quest.loc[patient_id, "TDI"]),
                }
            )
    return pd.DataFrame(records)


def safe_pump() -> InsulinPump:
    pump = InsulinPump.withName("InsulinPump")
    defaults = {
        "inc_basal": 0.01,
        "inc_bolus": 0.05,
        "U2PMOL": 6000.0,
        "max_basal": 2.0,
        "min_basal": 0.0,
        "max_bolus": 10.0,
        "min_bolus": 0.0,
    }
    clean = {}
    for key, value in getattr(pump, "_params", {}).items():
        try:
            clean[key] = float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            clean[key] = defaults.get(key, np.nan)
    for key, value in defaults.items():
        if key not in clean or not np.isfinite(clean[key]):
            clean[key] = value
    pump._params = clean
    return pump


def controller_action(
    *,
    cgm: float,
    meal_rate: float,
    meal_started: bool,
    sample_time: float,
    basal_rate: float,
    basal_factor: float,
    icr: float,
    isf: float,
    correction_strategy: str,
    safety_correction_due: bool,
    target: float = 140.0,
) -> tuple[ControllerAction, float, float, bool]:
    basal = basal_rate * basal_factor
    carb_bolus_u = meal_rate * sample_time / icr if meal_rate > 0 else 0.0
    correction_u = (
        max(0.0, (cgm - target) / isf)
        if correction_strategy == "meal_onset" and meal_started
        else 0.0
    )
    safety_correction = False
    if cgm >= 300.0 and safety_correction_due:
        correction_u = min(1.0, max(0.0, (cgm - 250.0) / isf))
        safety_correction = True
    total_bolus_u = carb_bolus_u + correction_u

    # Basal suspension is a prespecified safety feature, applied identically
    # in both Ramadan arms and the pre-Ramadan control.
    if cgm < 70.0:
        basal = 0.0
        total_bolus_u = 0.0

    bolus_rate = total_bolus_u / sample_time
    return (
        ControllerAction(basal=basal, bolus=bolus_rate),
        carb_bolus_u,
        correction_u,
        safety_correction,
    )


def summarize_day(group: pd.DataFrame) -> pd.Series:
    glucose = group["CGM_mg_dL"].dropna().to_numpy(dtype=float)
    mean = float(np.mean(glucose))
    sd = float(np.std(glucose, ddof=0))
    f = 1.509 * ((np.log(np.clip(glucose, 1, None)) ** 1.084) - 5.381)
    risk = 10 * f**2
    lbgi = float(risk[f < 0].mean()) if np.any(f < 0) else 0.0
    hbgi = float(risk[f > 0].mean()) if np.any(f > 0) else 0.0
    return pd.Series(
        {
            "Mean": mean,
            "SD": sd,
            "CV": sd / mean * 100,
            "GMI": 3.31 + 0.02392 * mean,
            "TIR": np.mean((glucose >= 70) & (glucose <= 180)) * 100,
            "TBR": np.mean(glucose < 70) * 100,
            "TBR_<54": np.mean(glucose < 54) * 100,
            "TAR": np.mean(glucose > 180) * 100,
            "LBGI": lbgi,
            "HBGI": hbgi,
            "OverallRisk": lbgi + hbgi,
            "MinCGM": float(np.min(glucose)),
            "CHO_g": float(group["CHO_g_per_min"].sum() * 3),
            "Insulin_U": float(group["Delivered_Insulin_U"].sum()),
            "BasalSuspension_min": float((group["Basal_U_per_min"] == 0).sum() * 3),
        }
    )


def run_period(
    *,
    patient_id: str,
    role: str,
    period: str,
    kind: str,
    basal_factor: float,
    days: int,
    seed: int,
    params: pd.Series,
    output_dir: Path,
    correction_strategy: str,
    meal_seed: int | None = None,
    stochastic_meals: bool = False,
    fasting_window_only: bool = False,
) -> dict:
    start = dt.datetime(2026, 1, 18) if kind == "pre" else dt.datetime(2026, 2, 18)
    end = start + dt.timedelta(days=days)
    scenario = MealScenario(
        build_meals(
            kind,
            role,
            start,
            days,
            meal_seed=meal_seed,
            stochastic_meals=stochastic_meals,
        )
    )
    patient = T1DPatient.withName(patient_id)
    sensor = CGMSensor.withName("Dexcom", seed=seed)
    pump = safe_pump()
    env = T1DSimEnv(patient, sensor, pump, scenario)
    reset = env.reset()
    observation = reset.observation
    records = []
    previous_meal: Meal | None = None
    terminal = False
    last_safety_correction: dt.datetime | None = None
    fasting_windows: dict[dt.date, tuple[dt.datetime, dt.datetime]] = {}
    if fasting_window_only:
        for meal in scenario.meals:
            if meal.name == "suhoor":
                fasting_windows.setdefault(meal.start.date(), [meal.end, meal.end])
            elif meal.name == "iftar_fast" and meal.start.date() in fasting_windows:
                fasting_windows[meal.start.date()][1] = meal.start

    while env.time < end:
        record_time = env.time
        meal = scenario.meal_at(env.time)
        meal_rate = scenario.get_action(env.time).meal
        meal_started = meal is not None and (
            previous_meal is None or previous_meal.start != meal.start
        )
        cgm = float(np.asarray(observation.CGM).reshape(-1)[0])
        safety_correction_due = (
            last_safety_correction is None
            or env.time - last_safety_correction >= dt.timedelta(hours=4)
        )
        effective_basal_factor = basal_factor
        if fasting_window_only:
            window = fasting_windows.get(env.time.date())
            effective_basal_factor = (
                basal_factor if window and window[0] <= env.time < window[1] else 1.0
            )
        action, carb_bolus_u, correction_u, safety_correction = controller_action(
            cgm=cgm,
            meal_rate=meal_rate,
            meal_started=meal_started,
            sample_time=float(env.sample_time),
            basal_rate=float(params["Basal_U_per_min"]),
            basal_factor=effective_basal_factor,
            icr=float(params["ICR_g_per_U"]),
            isf=float(params["ISF_mg_dL_per_U"]),
            correction_strategy=correction_strategy,
            safety_correction_due=safety_correction_due,
        )
        if safety_correction:
            last_safety_correction = env.time
        bg = float(env.patient.observation.Gsub)
        step = env.step(action)
        delivered_u = float(env.insulin_hist[-1]) * float(env.sample_time)
        records.append(
            {
                "Time": record_time,
                "Patient": patient_id,
                "Role": role,
                "Period": period,
                "CGM_mg_dL": cgm,
                "BG_mg_dL": bg,
                "Basal_U_per_min": float(action.basal),
                "Bolus_U_per_min": float(action.bolus),
                "CarbBolus_U": carb_bolus_u,
                "CorrectionBolus_U": correction_u,
                "SafetyCorrection": safety_correction,
                "Delivered_Insulin_U": delivered_u,
                "CHO_g_per_min": float(step.info["meal"]),
                "Meal": meal.name if meal else "",
                "MealStarted": meal_started,
            }
        )
        observation = step.observation
        previous_meal = meal
        if step.done:
            terminal = True
            break

    frame = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{patient_id}_{period}_AllDays.csv", index=False)
    frame["Date"] = pd.to_datetime(frame["Time"]).dt.date
    daily = frame.groupby("Date", sort=True).apply(summarize_day, include_groups=False).reset_index()
    daily.insert(1, "Patient", patient_id)
    daily.insert(2, "Role", role)
    daily.insert(3, "Period", period)
    daily.to_csv(output_dir / f"{patient_id}_{period}_DailySummary.csv", index=False)
    return {
        "Patient": patient_id,
        "Role": role,
        "Period": period,
        "DaysRequested": days,
        "DaysObserved": int(daily.shape[0]),
        "SamplesObserved": int(frame.shape[0]),
        "Terminal": terminal,
        "MinimumBG": float(frame["BG_mg_dL"].min()),
        "MinimumCGM": float(frame["CGM_mg_dL"].min()),
        "MaximumBG": float(frame["BG_mg_dL"].max()),
        "SafetyCorrections": int(frame["SafetyCorrection"].sum()),
        "MealSeed": meal_seed,
        "StochasticMeals": stochastic_meals,
        "FastingWindowOnly": fasting_window_only,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("raw/simulator_output"))
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--patients", nargs="*", help="Optional patient IDs for a test run")
    parser.add_argument("--periods", nargs="*", help="Optional period names for a test run")
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument(
        "--correction-strategy",
        choices=("none", "meal_onset"),
        default="none",
        help="Primary analysis uses none; meal_onset is reserved for sensitivity analysis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    params = cohort_parameters()
    params.to_csv(args.output / "virtual_cohort_parameters.csv", index=False)
    selected_patients = set(args.patients or params["Patient"])
    selected_periods = set(args.periods or [period[0] for period in PERIODS])
    completion = []
    daily_frames = []

    for _, row in params.iterrows():
        patient_id = row["Patient"]
        if patient_id not in selected_patients:
            continue
        role = row["Role"]
        patient_number = int(patient_id.split("#")[1])
        for period_index, (period, kind, basal_factor) in enumerate(PERIODS):
            if period not in selected_periods:
                continue
            print(f"Running {patient_id}: {period}", flush=True)
            period_dir = args.output / role / patient_id / period
            result = run_period(
                patient_id=patient_id,
                role=role,
                period=period,
                kind=kind,
                basal_factor=basal_factor,
                days=args.days,
                seed=args.seed + patient_number * 10,
                params=row,
                output_dir=period_dir,
                correction_strategy=args.correction_strategy,
            )
            completion.append(result)
            daily_frames.append(
                pd.read_csv(period_dir / f"{patient_id}_{period}_DailySummary.csv")
            )
            print(json.dumps(result), flush=True)

    completion_frame = pd.DataFrame(completion)
    completion_frame.to_csv(args.output / "simulation_completion.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(
        args.output / "Combined_AllPatients_DailySummary_3Arm.csv", index=False
    )
    with open(args.output / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "days": args.days,
                "seed": args.seed,
                "patients": sorted(selected_patients),
                "periods": sorted(selected_periods),
                "controller": {
                    "meal_bolus": "CHO/ICR distributed over meal samples",
                    "correction": args.correction_strategy,
                    "basal_suspension": "CGM < 70 mg/dL",
                    "hyperglycemia_rescue": (
                        "CGM >= 300 mg/dL; correct toward 250 mg/dL, maximum 1 U, "
                        "minimum 4-hour interval"
                    ),
                },
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
