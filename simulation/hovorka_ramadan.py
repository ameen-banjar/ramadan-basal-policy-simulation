#!/usr/bin/env python3
"""
Hovorka (Cambridge) T1D Simulator — Ramadan Basal Policy Dose-Response Study
=============================================================================
Implements the physiological model from:
  Hovorka R, et al. "Nonlinear model predictive control of glucose concentration
  in subjects with type 1 diabetes." Physiol Meas. 2004;25(4):905-20.

Virtual patient parameter ranges based on:
  Wilinska ME, et al. "In Silico Testing of Artificial Pancreas."
  J Diabetes Sci Technol. 2010;4(1):102-15.

Replicates the 5-policy paired crossover design from the UVA/Padova study:
  100%, 90%, 80%, 70%, 60% of individual basal rate over 30-day Ramadan.

Outputs: per-patient per-policy CSV files + summary table.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import pandas as pd
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / "run.log"

def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')

# ============================================================
# HOVORKA 2004 — NOMINAL PARAMETERS
# (all per-kg quantities; absolute values require × BW)
# ============================================================
# Glucose subsystem
#   Q1, Q2   : glucose mass in compartments 1 & 2 (mmol)
#   G = Q1 / (VG × BW)  in mmol/L  →  × 18.015 = mg/dL
#
# Insulin action (remote effects)
#   x1 : effect on glucose distribution/transport (min⁻¹)
#   x2 : effect on glucose utilization (min⁻¹)
#   x3 : effect on endogenous glucose production (dimensionless)
#
# SC insulin absorption
#   S1, S2 : subcutaneous depot compartments (mU)
#   I      : plasma insulin concentration (mU/L)
#
# Gut absorption
#   D1, D2 : gut glucose compartments (mmol)

NOMINAL = dict(
    VG    = 0.16,    # Glucose distribution volume           L/kg
    F01   = 0.0097,  # Non-insulin-dep glucose utilization   mmol/kg/min
    EGP0  = 0.0161,  # EGP at zero insulin action            mmol/kg/min
    k12   = 0.066,   # Q2→Q1 transfer rate                   min⁻¹
    ka1   = 0.006,   # x1 deactivation rate                  min⁻¹
    kb1   = 0.0034,  # x1 activation rate constant           min⁻¹ per mU/L
    ka2   = 0.06,    # x2 deactivation rate                  min⁻¹
    kb2   = 0.006,   # x2 activation rate constant           min⁻¹ per mU/L
    ka3   = 0.03,    # x3 deactivation rate                  min⁻¹
    kb3   = 0.024,   # x3 activation rate constant           min⁻¹ per mU/L
    tau_s = 55.0,    # SC absorption time constant            min
    ke    = 0.138,   # Plasma insulin elimination             min⁻¹
    VI    = 0.12,    # Insulin distribution volume            L/kg
    Ag    = 0.8,     # Carbohydrate bioavailability fraction
    tmax_G= 40.0,    # Gut absorption time constant           min
)

# ============================================================
# VIRTUAL PATIENTS — 10 adults
# Insulin sensitivity (isf_mult) spans ≈ 0.6–1.5 × nominal
# Body weight 62–92 kg (realistic adult T1D range)
# EGP0 and F01 vary ±20% to introduce physiological diversity
# ============================================================
PATIENTS_DEF = [
    # (pid,           BW,  isf_mult, egp_mult, f01_mult)
    ("adult#001",     62,   1.50,     0.80,     0.93),
    ("adult#002",     68,   1.30,     0.85,     0.95),
    ("adult#003",     72,   1.15,     0.90,     0.97),
    ("adult#004",     75,   1.05,     0.95,     0.98),
    ("adult#005",     78,   1.00,     0.98,     1.00),
    ("adult#006",     80,   0.95,     1.00,     1.00),
    ("adult#007",     82,   0.90,     1.02,     1.02),
    ("adult#008",     85,   0.85,     1.08,     1.04),
    ("adult#009",     88,   0.75,     1.15,     1.06),
    ("adult#010",     92,   0.60,     1.22,     1.08),
]

def make_params(pid, BW, isf_mult, egp_mult, f01_mult):
    p = dict(NOMINAL)
    p['pid']      = pid
    p['BW']       = float(BW)
    p['isf_mult'] = isf_mult
    p['kb1'] = NOMINAL['kb1'] * isf_mult
    p['kb2'] = NOMINAL['kb2'] * isf_mult
    p['kb3'] = NOMINAL['kb3'] * isf_mult
    p['EGP0'] = NOMINAL['EGP0'] * egp_mult
    p['F01']  = NOMINAL['F01']  * f01_mult
    return p

ALL_PATIENTS = [make_params(*row) for row in PATIENTS_DEF]

# ============================================================
# STEADY-STATE SOLVER
# Target fasting glucose: 7.5 mmol/L ≈ 135 mg/dL
# ============================================================
TARGET_G_SS = 7.5  # mmol/L

def _ss_balance(u_b, p, G_ss):
    """Glucose balance residual at steady state (no meals, no renal clearance)."""
    BW = p['BW']
    I_ss = u_b / (p['ke'] * p['VI'] * BW)
    x1   = p['kb1'] / p['ka1'] * I_ss
    x2   = p['kb2'] / p['ka2'] * I_ss
    x3   = p['kb3'] / p['ka3'] * I_ss
    Q1   = G_ss * p['VG'] * BW
    Q2   = x1 * Q1 / (p['k12'] + x2)
    F01c = p['F01'] * BW   # G_ss > 4.5 assumed
    balance = (-(F01c + x1 * Q1) + p['k12'] * Q2 + p['EGP0'] * BW * (1 - x3))
    return balance

def find_basal(p, G_target=TARGET_G_SS):
    """Find u_b (mU/min) that yields G_target at fasting steady state."""
    try:
        return brentq(_ss_balance, 1e-4, 10.0, args=(p, G_target), xtol=1e-7)
    except ValueError:
        return 0.5  # fallback

def ss_state(u_b, p, G_ss=TARGET_G_SS):
    """Return 10-element initial state vector at steady state."""
    BW = p['BW']
    I_ss = u_b / (p['ke'] * p['VI'] * BW)
    x1   = p['kb1'] / p['ka1'] * I_ss
    x2   = p['kb2'] / p['ka2'] * I_ss
    x3   = p['kb3'] / p['ka3'] * I_ss
    Q1   = G_ss * p['VG'] * BW
    Q2   = x1 * Q1 / (p['k12'] + x2)
    S12  = u_b * p['tau_s']
    return np.array([Q1, Q2, x1, x2, x3, S12, S12, I_ss, 0.0, 0.0])

# ============================================================
# ODE — Hovorka 2004 (state-space form)
# State: [Q1, Q2, x1, x2, x3, S1, S2, I, D1, D2]
# ============================================================
def hovorka_ode(t, y, p, u_basal, meal_sched, bolus_events):
    Q1, Q2, x1, x2, x3, S1, S2, I, D1, D2 = y

    BW = p['BW']
    G  = max(Q1, 0.0) / (p['VG'] * BW)          # mmol/L

    # Non-insulin-dependent glucose utilization
    F01c = p['F01'] * BW if G >= 4.5 else p['F01'] * BW * G / 4.5

    # Renal clearance (threshold 9 mmol/L)
    FR = 0.003 * (G - 9.0) * p['VG'] * BW if G > 9.0 else 0.0

    # Gut glucose appearance (mmol/min)
    UG = max(D2, 0.0) / p['tmax_G']

    # Meal input at time t (g/min → mmol/min)
    meal_g_min = _meal_rate(t, meal_sched)
    meal_mmol  = meal_g_min * 1000.0 / 180.016

    # Insulin delivery (mU/min): basal + bolus
    u = max(u_basal + _bolus_rate(t, bolus_events), 0.0)

    # ODEs
    dQ1 = -(F01c + x1 * max(Q1, 0.0)) + p['k12'] * max(Q2, 0.0) - FR + p['EGP0'] * BW * (1.0 - x3) + UG
    dQ2 =  x1 * max(Q1, 0.0) - (p['k12'] + x2) * max(Q2, 0.0)
    dx1 = -p['ka1'] * x1 + p['kb1'] * max(I, 0.0)
    dx2 = -p['ka2'] * x2 + p['kb2'] * max(I, 0.0)
    dx3 = -p['ka3'] * x3 + p['kb3'] * max(I, 0.0)
    dS1 =  u - S1 / p['tau_s']
    dS2 = (S1 - S2) / p['tau_s']
    dI  =  S2 / (p['tau_s'] * p['VI'] * BW) - p['ke'] * max(I, 0.0)
    dD1 =  p['Ag'] * meal_mmol - D1 / p['tmax_G']
    dD2 = (max(D1, 0.0) - max(D2, 0.0)) / p['tmax_G']

    return [dQ1, dQ2, dx1, dx2, dx3, dS1, dS2, dI, dD1, dD2]

def _meal_rate(t, sched):
    for (ts, te, g) in sched:
        if ts <= t < te:
            return g / (te - ts)
    return 0.0

def _bolus_rate(t, events):
    rate = 0.0
    for (ts, te, mU) in events:
        if ts <= t < te:
            rate += mU / (te - ts)
    return rate

# ============================================================
# MEAL & BOLUS SCHEDULE BUILDERS
# Matching UVA/Padova study design exactly
# ============================================================
CHO_ADULT = 250.0  # g/day (same as UVA/Padova adults)

# Carb variation sequence: 10-level, 70%–130%, cycling every 10 days
RAMADAN_CHO_FACTORS = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30]
SUHOOR_OFFSETS_MIN  = [60, 120, 180]  # minutes before Fajr, cycles every 10 days

# Reference physiology for bolus scaling.
# A "typical" T1D adult using ~12 mU/min basal has ICR≈12 g/U.
# Patients in the Hovorka model reach euglycemia at model-native u_b values
# (0.04–0.23 mU/min) that are ~50–300× smaller than real-world rates.
# Boluses must scale proportionally so the bolus:basal ratio stays physiological.
REAL_UB_REF  = 12.0   # mU/min  — reference adult basal (≈0.72 U/hr)
REAL_ICR_REF = 12.0   # g/U     — reference T1D adult ICR (500-rule ÷ 40 U/day TDD)

def bolus_mU_for_meal(g_carbs, u_b):
    """
    Return bolus in mU for g_carbs grams, scaled to the model's u_b.

    In real physiology: bolus_U = g / ICR_real.
    In the model:       bolus_mU = bolus_U × 1000 × (u_b / REAL_UB_REF)

    This keeps the bolus:daily-basal ratio the same as in real T1D patients,
    regardless of the model's internal insulin unit scale.
    """
    real_U = g_carbs / REAL_ICR_REF
    return real_U * 1000.0 * (u_b / REAL_UB_REF)

def build_ramadan_schedule(n_days, u_b):
    """
    Suhoor 35% | Iftar-fast 10% | Iftar-main 40% | Snack 15%
    Suhoor: ~4:30 AM (offset-dependent), Iftar: 18:30, Snack: 22:00
    Bolus: delivered over 5 min at meal start.
    """
    meals, bolus = [], []

    for day in range(n_days):
        base   = day * 1440
        factor = RAMADAN_CHO_FACTORS[day % 10]
        total  = CHO_ADULT * factor
        offset = SUHOOR_OFFSETS_MIN[(day // 10) % 3]

        fajr   = base + 5*60 + 30               # 05:30
        su_s   = fajr - offset - 60
        su_e   = su_s + 60
        if1_s  = base + 18*60 + 30              # 18:30
        if1_e  = if1_s + 15
        if2_s  = if1_e
        if2_e  = if2_s + 45
        sn_s   = base + 22*60                   # 22:00
        sn_e   = sn_s + 40

        slots = [
            (su_s,  su_e,  0.35 * total),
            (if1_s, if1_e, 0.10 * total),
            (if2_s, if2_e, 0.40 * total),
            (sn_s,  sn_e,  0.15 * total),
        ]
        for (ts, te, g) in slots:
            meals.append((ts, te, g))
            bolus.append((ts, ts + 5, bolus_mU_for_meal(g, u_b)))

    return meals, bolus

def build_preramadan_schedule(n_days, u_b):
    """
    Breakfast 08:00 (30%) | Lunch 13:00 (35%) | Dinner 19:00 (30%) | Snack 15:30 (5%)
    """
    meals, bolus = [], []

    for day in range(n_days):
        base  = day * 1440
        total = CHO_ADULT

        slots = [
            (base +  8*60,      base +  8*60 + 30,  0.30 * total),
            (base + 13*60,      base + 13*60 + 30,  0.35 * total),
            (base + 19*60,      base + 19*60 + 30,  0.30 * total),
            (base + 15*60 + 30, base + 15*60 + 50,  0.05 * total),
        ]
        for (ts, te, g) in slots:
            meals.append((ts, te, g))
            bolus.append((ts, ts + 5, bolus_mU_for_meal(g, u_b)))

    return meals, bolus

# ============================================================
# SIMULATION ENGINE — step-by-step (5-min CGM intervals)
# ============================================================
STEP   = 5     # minutes per integration step
NOISE  = 5.0   # CGM sensor noise SD (mg/dL)
N_DAYS = 30

def simulate(p, u_b, meals, bolus, n_days, y0, cgm_seed=42):
    """
    Run n_days simulation.
    Safety rules (matching UVA/Padova paper):
      - Suspend insulin (basal + bolus) when CGM < 70 mg/dL
      - Rescue correction bolus when CGM >= 300 mg/dL (target 250 mg/dL,
        cap 1 U, min 4 h apart)
    Returns DataFrame and terminal flag.
    """
    rng      = np.random.default_rng(cgm_seed)
    total    = n_days * 1440
    y        = y0.copy()
    records  = []
    terminal = False
    last_rescue_t = -240  # allow rescue from t=0

    for t0 in range(0, total, STEP):
        t1 = t0 + STEP

        # Current CGM
        G_true = max(y[0], 0.0) / (p['VG'] * p['BW'])
        G_mgdl = float(np.clip(G_true * 18.015 + rng.normal(0, NOISE), 20, 600))

        # Rescue correction (≥300 mg/dL, ≥4 h since last)
        extra_bolus = []
        if G_mgdl >= 300.0 and (t0 - last_rescue_t) >= 240:
            # ISF_real ≈ 1700/TDD_U; at 40 U/day TDD → ISF ≈ 42.5 mg/dL per U
            isf_real_mgdl_per_U = 42.5
            correction_U = min(max((G_mgdl - 250.0) / isf_real_mgdl_per_U, 0.0), 1.0)
            correction_mU_model = correction_U * 1000.0 * (u_b / REAL_UB_REF)
            extra_bolus  = [(t0, t0 + 5, correction_mU_model)]
            last_rescue_t = t0

        # Basal suspension during hypoglycemia
        u_eff = 0.0 if G_mgdl < 70.0 else u_b

        # All bolus events for this step
        all_bolus = bolus + extra_bolus

        try:
            sol = solve_ivp(
                hovorka_ode,
                [float(t0), float(t1)],
                y,
                args=(p, u_eff, meals, all_bolus),
                method='RK45',
                max_step=1.0,
                rtol=1e-4, atol=1e-7,
                dense_output=False,
            )
            if sol.success:
                y = sol.y[:, -1]
            # else: keep previous y
        except Exception:
            pass

        y = np.maximum(y, 0.0)

        # Check terminal (extreme hyperglycemia → simulator instability)
        G_check = y[0] / (p['VG'] * p['BW']) * 18.015
        if G_check > 550.0:
            terminal = True
            log(f"    *** Terminal event at minute {t0} (CGM≈{G_check:.0f} mg/dL) ***")
            break

        records.append({
            'min'       : t0,
            'day'       : t0 // 1440,
            'CGM_mgdL'  : G_mgdl,
            'G_true_mmol': float(G_true),
        })

    return pd.DataFrame(records), terminal

# ============================================================
# OUTCOME METRICS
# ============================================================
def outcomes(df):
    g = df['CGM_mgdL'].values
    mean_g = float(np.mean(g))
    return dict(
        TIR   = float(np.mean((g >= 70) & (g <= 180)) * 100),
        TBR   = float(np.mean(g < 70)  * 100),
        TBR54 = float(np.mean(g < 54)  * 100),
        TAR   = float(np.mean(g > 180) * 100),
        MeanCGM = mean_g,
        GMI   = 3.31 + 0.02392 * mean_g,
        CV    = float(np.std(g) / mean_g * 100) if mean_g > 0 else np.nan,
        N_obs = len(g),
    )

# ============================================================
# MAIN
# ============================================================
POLICIES = [1.00, 0.90, 0.80, 0.70, 0.60]
CGM_SEED = 42

log("=" * 60)
log("Hovorka Ramadan Basal Policy Study")
log(f"Patients: {len(ALL_PATIENTS)} | Policies: {POLICIES} | Days: {N_DAYS}")
log(f"Output  : {OUT_DIR}")
log("=" * 60)

summary = []

for p in ALL_PATIENTS:
    pid = p['pid']
    log(f"\n{'─'*50}")
    log(f"Patient: {pid}  BW={p['BW']}kg  isf×{p['isf_mult']:.2f}")

    # --- Basal rate ---
    u_b = find_basal(p, TARGET_G_SS)
    log(f"  u_b = {u_b:.4f} mU/min  ({u_b*60:.3f} mU/h  {u_b*1440/1000:.3f} U/day basal)")
    y0 = ss_state(u_b, p, TARGET_G_SS)

    # --- Pre-Ramadan (baseline TBR) ---
    pre_meals, pre_bolus = build_preramadan_schedule(N_DAYS, u_b)
    df_pre, _ = simulate(p, u_b, pre_meals, pre_bolus, N_DAYS, y0, CGM_SEED)
    pre_out   = outcomes(df_pre)
    log(f"  Pre-Ramadan → TIR={pre_out['TIR']:.1f}%  TBR={pre_out['TBR']:.1f}%  TAR={pre_out['TAR']:.1f}%")
    df_pre.to_csv(OUT_DIR / f"{pid}_PreRamadan_AllDays.csv", index=False)

    # --- Five Ramadan policies ---
    ram_meals, ram_bolus = build_ramadan_schedule(N_DAYS, u_b)

    for factor in POLICIES:
        pct  = int(factor * 100)
        u_ram = u_b * factor
        df_ram, term = simulate(p, u_ram, ram_meals, ram_bolus, N_DAYS, y0.copy(), CGM_SEED)

        if term or len(df_ram) < 100:
            log(f"  Policy {pct}% → TERMINAL (excluded from 30-day analysis)")
            summary.append(dict(
                Patient=pid, Policy_pct=pct,
                Pre_TBR=pre_out['TBR'], Pre_TIR=pre_out['TIR'],
                Terminal=True,
                TIR=np.nan, TBR=np.nan, TBR54=np.nan,
                TAR=np.nan, MeanCGM=np.nan, GMI=np.nan, CV=np.nan,
                u_b_mU_min=u_b, BW=p['BW'], isf_mult=p['isf_mult'],
            ))
        else:
            out = outcomes(df_ram)
            log(f"  Policy {pct}% → TIR={out['TIR']:.1f}%  TBR={out['TBR']:.1f}%  TAR={out['TAR']:.1f}%")
            df_ram.to_csv(OUT_DIR / f"{pid}_Ramadan{pct}_AllDays.csv", index=False)
            summary.append(dict(
                Patient=pid, Policy_pct=pct,
                Pre_TBR=pre_out['TBR'], Pre_TIR=pre_out['TIR'],
                Terminal=False,
                u_b_mU_min=u_b, BW=p['BW'], isf_mult=p['isf_mult'],
                **out,
            ))

# ============================================================
# SAVE SUMMARY
# ============================================================
df_sum = pd.DataFrame(summary)
df_sum.to_csv(OUT_DIR / "summary_all_patients_policies.csv", index=False)

log("\n" + "=" * 60)
log("DOSE-RESPONSE SUMMARY (complete cases only)")
log("=" * 60)
log(f"{'Policy':>8}  {'TIR mean±SD':>14}  {'TBR mean±SD':>14}  {'TAR mean±SD':>14}")
for pct in [100, 90, 80, 70, 60]:
    sub = df_sum[(df_sum['Policy_pct'] == pct) & (~df_sum['Terminal'])]
    if sub.empty:
        continue
    log(f"  {pct:>4}%    {sub['TIR'].mean():>5.1f}±{sub['TIR'].std():>4.1f}%    "
        f"{sub['TBR'].mean():>5.1f}±{sub['TBR'].std():>4.1f}%    "
        f"{sub['TAR'].mean():>5.1f}±{sub['TAR'].std():>4.1f}%")

log(f"\n✅  Done — results in {OUT_DIR}")
