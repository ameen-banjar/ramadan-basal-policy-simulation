#!/usr/bin/env python3
"""
UVA/Padova (simglucose) Ramadan Basal Policy Dose-Response Study
=================================================================
Parallel design to the Hovorka simulation for cross-model comparison.

Study design:
  - All 30 UVA/Padova virtual patients (10 child, 10 adolescent, 10 adult)
  - Pre-Ramadan: 30 days, 4 meals (breakfast/lunch/snack/dinner)
  - Ramadan: 5 policies × 30 days (100%, 90%, 80%, 70%, 60% basal)
  - CGM seed: 42 per patient (fixed, reproducible)
  - Correction strategy: none (only safety bolus ≥300 mg/dL, suspend <70 mg/dL)
  - Meal design: identical to Hovorka simulation for fair comparison

Outputs:
  - Per-patient CSV files (CGM traces)
  - summary_all_patients_policies.csv
  - run.log
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path("/Users/abanjar/Ramadan/publication_study")))

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import CONTROL_QUEST
from simglucose.controller.base import Action as ControllerAction
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from types import SimpleNamespace

# ============================================================
# PATHS
# ============================================================
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / "run.log"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

# ============================================================
# STUDY PARAMETERS
# ============================================================
POLICIES      = [1.00, 0.90, 0.80, 0.70, 0.60]
N_DAYS        = 30
CGM_SEED_BASE = 42   # seed for patient i = CGM_SEED_BASE + i

BASELINE_CHO = {"child": 150.0, "adolescent": 220.0, "adult": 250.0}

# 10-level carb factor sequence (cycles every 10 days, same as Hovorka)
RAMADAN_CHO_FACTORS = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30]

# Suhoor offset from Fajr (05:30), cycles every 10 days (same as Hovorka)
SUHOOR_OFFSETS_MIN = [60, 120, 180]

ROLES = ["child", "adolescent", "adult"]


# ============================================================
# MEAL SCHEDULE BUILDER
# ============================================================
def build_meals(kind: str, role: str, start: dt.datetime, days: int) -> list:
    """Build deterministic meal list. kind='pre' or 'ramadan'."""
    meals = []
    cho_base = BASELINE_CHO[role]

    for d in range(days):
        date = (start + dt.timedelta(days=d)).date()

        if kind == "pre":
            total = cho_base
            specs = [
                ("breakfast",  dt.time(8, 0),       30, 0.30),
                ("lunch",      dt.time(13, 0),       30, 0.35),
                ("snack",      dt.time(15, 30),      20, 0.05),
                ("dinner",     dt.time(19, 0),       30, 0.30),
            ]
        else:
            total = cho_base * RAMADAN_CHO_FACTORS[d % 10]
            offset = SUHOOR_OFFSETS_MIN[(d // 10) % 3]
            fajr       = dt.datetime.combine(date, dt.time(5, 30))
            suhoor_s   = fajr - dt.timedelta(minutes=offset + 60)
            iftar_s    = dt.datetime.combine(date, dt.time(18, 30))
            snack_s    = dt.datetime.combine(date, dt.time(22, 0))
            specs = [
                ("suhoor",      suhoor_s.time(),                           60, 0.35),
                ("iftar_fast",  iftar_s.time(),                            15, 0.10),
                ("iftar_main",  (iftar_s + dt.timedelta(minutes=15)).time(), 45, 0.40),
                ("snack",       snack_s.time(),                            40, 0.15),
            ]

        for name, t, dur, frac in specs:
            ms = dt.datetime.combine(date, t)
            me = ms + dt.timedelta(minutes=dur)
            meals.append((ms, me, total * frac, name))

    return meals


class MealScenario:
    def __init__(self, meals):
        self.meals = meals
        self.start_time = min(m[0] for m in meals).replace(hour=0, minute=0)

    def reset(self):
        return None

    def meal_at(self, when):
        for (ms, me, grams, name) in self.meals:
            if ms <= when < me:
                return (ms, me, grams, name)
        return None

    def get_action(self, when):
        m = self.meal_at(when)
        if m is None:
            return SimpleNamespace(meal=0.0)
        ms, me, grams, _ = m
        dur_min = (me - ms).total_seconds() / 60.0
        return SimpleNamespace(meal=grams / dur_min)


# ============================================================
# PUMP HELPER
# ============================================================
def safe_pump():
    pump = InsulinPump.withName("InsulinPump")
    defaults = dict(inc_basal=0.01, inc_bolus=0.05, U2PMOL=6000.0,
                    max_basal=2.0, min_basal=0.0, max_bolus=10.0, min_bolus=0.0)
    clean = {}
    for k, v in getattr(pump, "_params", {}).items():
        try:
            clean[k] = float(np.asarray(v).reshape(-1)[0])
        except Exception:
            clean[k] = defaults.get(k, np.nan)
    for k, v in defaults.items():
        if k not in clean or not np.isfinite(clean.get(k, float("nan"))):
            clean[k] = v
    pump._params = clean
    return pump


# ============================================================
# COHORT PARAMETERS
# ============================================================
def cohort_parameters() -> pd.DataFrame:
    quest = pd.read_csv(CONTROL_QUEST).set_index("Name")
    records = []
    idx = 0
    for role in ROLES:
        for n in range(1, 11):
            pid = f"{role}#{n:03d}"
            patient = T1DPatient.withName(pid)
            basal_u_per_min = float(patient._params.u2ss * patient._params.BW / 6000.0)
            records.append({
                "Patient":         pid,
                "Role":            role,
                "Basal_U_per_min": basal_u_per_min,
                "ICR_g_per_U":     float(quest.loc[pid, "CR"]),
                "ISF_mg_dL_per_U": float(quest.loc[pid, "CF"]),
                "CGM_seed":        CGM_SEED_BASE + idx,
                "Index":           idx,
            })
            idx += 1
    return pd.DataFrame(records)


# ============================================================
# SIMULATION ENGINE
# ============================================================
def run_period(*, pid, role, kind, basal_factor, days, seed, params) -> pd.DataFrame:
    start = dt.datetime(2026, 1, 18) if kind == "pre" else dt.datetime(2026, 2, 18)
    end   = start + dt.timedelta(days=days)

    meals    = build_meals(kind, role, start, days)
    scenario = MealScenario(meals)

    patient = T1DPatient.withName(pid)
    sensor  = CGMSensor.withName("Dexcom", seed=seed)
    pump    = safe_pump()
    env     = T1DSimEnv(patient, sensor, pump, scenario)
    obs     = env.reset().observation

    basal_u_per_min = float(params["Basal_U_per_min"])
    icr             = float(params["ICR_g_per_U"])
    isf             = float(params["ISF_mg_dL_per_U"])

    records              = []
    prev_meal_start      = None
    last_safety_t        = None
    terminal             = False

    while env.time < end:
        t       = env.time
        cgm     = float(np.asarray(obs.CGM).reshape(-1)[0])
        m       = scenario.meal_at(t)
        meal_rate = scenario.get_action(t).meal
        meal_started = m is not None and (prev_meal_start != m[0])
        if meal_started:
            prev_meal_start = m[0]

        # Effective basal (with policy factor)
        basal = basal_u_per_min * basal_factor

        # Carb bolus at meal onset — cover TOTAL meal grams (not just one sample)
        carb_bolus_u = 0.0
        if meal_started and m is not None:
            ms, me, total_grams, _ = m
            carb_bolus_u = total_grams / icr

        # Safety correction ≥300 mg/dL (≤1 U, ≥4 h apart)
        safety_bolus_u = 0.0
        safety_due = last_safety_t is None or (t - last_safety_t) >= dt.timedelta(hours=4)
        if cgm >= 300.0 and safety_due:
            safety_bolus_u = min(1.0, max(0.0, (cgm - 250.0) / isf))
            last_safety_t  = t

        # Suspension <70 mg/dL
        if cgm < 70.0:
            basal          = 0.0
            carb_bolus_u   = 0.0
            safety_bolus_u = 0.0

        total_bolus_u = carb_bolus_u + safety_bolus_u
        action = ControllerAction(basal=basal, bolus=total_bolus_u / float(env.sample_time))

        step = env.step(action)
        obs  = step.observation

        records.append({
            "time":     t,
            "day":      (t - start).days,
            "CGM_mgdL": cgm,
            "basal":    basal,
            "bolus_U":  total_bolus_u,
        })

        if cgm > 600.0:
            terminal = True
            break

    df = pd.DataFrame(records)
    df["terminal"] = terminal
    return df


# ============================================================
# OUTCOME METRICS (matching Hovorka output format)
# ============================================================
def outcomes(df: pd.DataFrame) -> dict:
    g = df["CGM_mgdL"].values
    m = float(np.mean(g))
    return dict(
        TIR     = float(np.mean((g >= 70) & (g <= 180)) * 100),
        TBR     = float(np.mean(g < 70)  * 100),
        TBR54   = float(np.mean(g < 54)  * 100),
        TAR     = float(np.mean(g > 180) * 100),
        MeanCGM = m,
        GMI     = 3.31 + 0.02392 * m,
        CV      = float(np.std(g) / m * 100) if m > 0 else float("nan"),
        N_obs   = len(g),
    )


# ============================================================
# MAIN
# ============================================================
log("=" * 60)
log("UVA/Padova (simglucose) Ramadan Basal Policy Study")
log(f"Patients: 30 (10 child + 10 adolescent + 10 adult)")
log(f"Policies: {POLICIES} | Days: {N_DAYS}")
log(f"CGM seed: base={CGM_SEED_BASE} (patient i → seed {CGM_SEED_BASE}+i)")
log(f"Output  : {OUT_DIR}")
log("=" * 60)

params_df = cohort_parameters()
summary   = []

for _, row in params_df.iterrows():
    pid    = row["Patient"]
    role   = row["Role"]
    seed   = int(row["CGM_seed"])

    log(f"\n{'─'*50}")
    log(f"Patient: {pid}  Role={role}  Basal={row['Basal_U_per_min']:.4f} U/min  "
        f"ICR={row['ICR_g_per_U']:.1f} g/U  ISF={row['ISF_mg_dL_per_U']:.1f} mg/dL/U")

    # Pre-Ramadan
    df_pre  = run_period(pid=pid, role=role, kind="pre", basal_factor=1.0,
                         days=N_DAYS, seed=seed, params=row)
    pre_out = outcomes(df_pre)
    df_pre.to_csv(OUT_DIR / f"{pid}_PreRamadan_AllDays.csv", index=False)
    log(f"  Pre-Ramadan → TIR={pre_out['TIR']:.1f}%  TBR={pre_out['TBR']:.1f}%  TAR={pre_out['TAR']:.1f}%")

    row_summary = {"Patient": pid, "Role": role, "Period": "PreRamadan", "Policy": 1.0,
                   **{k: pre_out[k] for k in ["TIR","TBR","TBR54","TAR","MeanCGM","GMI","CV"]}}
    summary.append(row_summary)

    # Ramadan — 5 policies
    for policy in POLICIES:
        pct  = int(policy * 100)
        df_r = run_period(pid=pid, role=role, kind="ramadan", basal_factor=policy,
                          days=N_DAYS, seed=seed, params=row)
        out  = outcomes(df_r)
        df_r.to_csv(OUT_DIR / f"{pid}_Ramadan{pct:03d}_AllDays.csv", index=False)
        log(f"  Policy {pct:3d}% → TIR={out['TIR']:.1f}%  TBR={out['TBR']:.1f}%  TAR={out['TAR']:.1f}%")
        summary.append({"Patient": pid, "Role": role, "Period": "Ramadan", "Policy": policy,
                         **{k: out[k] for k in ["TIR","TBR","TBR54","TAR","MeanCGM","GMI","CV"]}})

# ============================================================
# SUMMARY TABLE
# ============================================================
summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_DIR / "summary_all_patients_policies.csv", index=False)

log("\n" + "=" * 60)
log("DOSE-RESPONSE SUMMARY — Ramadan periods (all 30 patients)")
log("=" * 60)
log(f"  {'Policy':>8}  {'TIR mean±SD':>18}  {'TBR mean±SD':>18}  {'TAR mean±SD':>18}")

ram = summary_df[summary_df["Period"] == "Ramadan"]
for pol in POLICIES:
    sub = ram[ram["Policy"] == pol]
    log(f"  {int(pol*100):>7}%  "
        f"  {sub['TIR'].mean():5.1f}±{sub['TIR'].std():4.1f}%  "
        f"  {sub['TBR'].mean():5.1f}±{sub['TBR'].std():4.1f}%  "
        f"  {sub['TAR'].mean():5.1f}±{sub['TAR'].std():4.1f}%")

log("\nPre-Ramadan baseline (all 30 patients):")
pre = summary_df[summary_df["Period"] == "PreRamadan"]
log(f"  TIR={pre['TIR'].mean():.1f}±{pre['TIR'].std():.1f}%  "
    f"TBR={pre['TBR'].mean():.1f}±{pre['TBR'].std():.1f}%  "
    f"TAR={pre['TAR'].mean():.1f}±{pre['TAR'].std():.1f}%")

log(f"\n✅  Done — results in {OUT_DIR}")
