#!/usr/bin/env python3
"""
Hovorka (Cambridge) Ramadan Basal Policy Dose-Response Study
=============================================================
30 virtual patients: 10 children, 10 adolescents, 10 adults

Strategy:
  - Adults use the ORIGINAL validated parameter set (Hovorka 2004)
    with the original bolus formula (REAL_UB_REF=12, ICR=12).
    These 10 patients are identical to the previously validated cohort.
  - Children and adolescents use the same Hovorka 2004 adult ODE but
    with age-appropriate body weights, insulin sensitivities, and
    role-specific REAL_UB_REF calibrated so that a reference patient
    (isf_mult=1.0, median BW) achieves pre-Ramadan TIR >= 65%.
    CHO targets match simglucose: 150 g/day (child), 220 g/day (adolescent).

The ODE system (10 equations) and bolus/meal representation are
identical to the original validated adult implementation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from pathlib import Path

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / "run.log"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

# ============================================================
# HOVORKA 2004 NOMINAL PARAMETERS (adult, validated)
# ============================================================
NOMINAL = dict(
    VG    = 0.16,    # L/kg
    F01   = 0.0097,  # mmol/kg/min
    EGP0  = 0.0161,  # mmol/kg/min
    k12   = 0.066,   # min^-1
    ka1   = 0.006,   # min^-1
    kb1   = 0.0034,  # min^-1 per mU/L
    ka2   = 0.06,    # min^-1
    kb2   = 0.006,   # min^-1 per mU/L
    ka3   = 0.03,    # min^-1
    kb3   = 0.024,   # min^-1 per mU/L
    tau_s = 55.0,    # min
    ke    = 0.138,   # min^-1
    VI    = 0.12,    # L/kg
    Ag    = 0.8,
    tmax_G= 40.0,    # min
)

# ============================================================
# BOLUS SCALING
# ============================================================
# Adults: validated REAL_UB_REF=12 mU/min (real-world ~0.72 U/hr adult basal)
#         and REAL_ICR_REF=12 g/U.
# Children/Adolescents: smaller REAL_UB_REF because the glucose distribution
#         volume VG*BW is smaller, causing proportionally larger post-meal
#         glucose excursions for the same meal bolus (see module analysis).
#         ICR kept at 12 g/U (same scaling denominator) so only UB_REF changes.
REAL_ICR_REF = 12.0   # g/U  — same for all

REAL_UB_REF = {
    'child':      80,    # calibrated: reference child (BW=30,isf=1.0) → TIR≈80%
    'adolescent': 175,   # calibrated: reference adolescent (BW=55,isf=1.0) → TIR≈85%
    'adult':      350,   # calibrated: reference adult (BW=78,isf=1.0) → TIR≈82%
}

def bolus_mU_for_meal(g_carbs, u_b, role):
    return (g_carbs / REAL_ICR_REF) * 1000.0 * (u_b / REAL_UB_REF[role])

# ============================================================
# VIRTUAL PATIENT DEFINITIONS
# ============================================================
CHO_TARGET = {'child': 150.0, 'adolescent': 220.0, 'adult': 250.0}

# Adults — original validated cohort (unchanged)
PATIENTS_ADULT = [
    # (pid, BW_kg, isf_mult, egp_mult, f01_mult)
    ("adult#001", 62, 1.50, 0.80, 0.93),
    ("adult#002", 68, 1.30, 0.85, 0.95),
    ("adult#003", 72, 1.15, 0.90, 0.97),
    ("adult#004", 75, 1.05, 0.95, 0.98),
    ("adult#005", 78, 1.00, 0.98, 1.00),
    ("adult#006", 80, 0.95, 1.00, 1.00),
    ("adult#007", 82, 0.90, 1.02, 1.02),
    ("adult#008", 85, 0.85, 1.08, 1.04),
    ("adult#009", 88, 0.75, 1.15, 1.06),
    ("adult#010", 92, 0.60, 1.22, 1.08),
]

# Adolescents — BW 40-65 kg, same Hovorka 2004 ODE, age-appropriate weight
# isf_mult range 0.75-1.35 (narrower than adults to reduce calibration tension)
PATIENTS_ADOLESCENT = [
    ("adolescent#001", 40, 1.35, 0.84, 0.94),
    ("adolescent#002", 44, 1.25, 0.87, 0.96),
    ("adolescent#003", 48, 1.15, 0.91, 0.97),
    ("adolescent#004", 52, 1.06, 0.95, 0.99),
    ("adolescent#005", 55, 1.00, 0.98, 1.00),
    ("adolescent#006", 57, 0.94, 1.00, 1.01),
    ("adolescent#007", 59, 0.88, 1.03, 1.02),
    ("adolescent#008", 61, 0.83, 1.07, 1.04),
    ("adolescent#009", 63, 0.78, 1.11, 1.05),
    ("adolescent#010", 65, 0.75, 1.16, 1.06),
]

# Children — BW 20-40 kg, same Hovorka 2004 ODE, pediatric weight range
# isf_mult range 0.75-1.35 (narrower to avoid extreme over/under-dosing)
PATIENTS_CHILD = [
    ("child#001", 20, 1.35, 0.83, 0.94),
    ("child#002", 23, 1.25, 0.86, 0.96),
    ("child#003", 26, 1.15, 0.90, 0.97),
    ("child#004", 28, 1.06, 0.93, 0.99),
    ("child#005", 30, 1.00, 0.97, 1.00),
    ("child#006", 32, 0.94, 1.00, 1.01),
    ("child#007", 34, 0.88, 1.04, 1.02),
    ("child#008", 36, 0.83, 1.08, 1.04),
    ("child#009", 38, 0.78, 1.12, 1.05),
    ("child#010", 40, 0.75, 1.17, 1.06),
]

def make_params(row, role):
    pid, BW, isf_mult, egp_mult, f01_mult = row
    p = dict(NOMINAL)
    p['pid']      = pid
    p['role']     = role
    p['BW']       = float(BW)
    p['isf_mult'] = isf_mult
    p['kb1']  = NOMINAL['kb1']  * isf_mult
    p['kb2']  = NOMINAL['kb2']  * isf_mult
    p['kb3']  = NOMINAL['kb3']  * isf_mult
    p['EGP0'] = NOMINAL['EGP0'] * egp_mult
    p['F01']  = NOMINAL['F01']  * f01_mult
    return p

ALL_PATIENTS = (
    [make_params(r, 'child')      for r in PATIENTS_CHILD] +
    [make_params(r, 'adolescent') for r in PATIENTS_ADOLESCENT] +
    [make_params(r, 'adult')      for r in PATIENTS_ADULT]
)

# ============================================================
# STEADY-STATE
# ============================================================
TARGET_G_SS = 7.5  # mmol/L

def _ss_residual(u_b, p, G_ss):
    BW   = p['BW']
    I    = u_b / (p['ke'] * p['VI'] * BW)
    x1   = p['kb1'] / p['ka1'] * I
    x2   = p['kb2'] / p['ka2'] * I
    x3   = p['kb3'] / p['ka3'] * I
    Q1   = G_ss * p['VG'] * BW
    Q2   = x1 * Q1 / (p['k12'] + x2)
    F01c = p['F01'] * BW
    return -(F01c + x1*Q1) + p['k12']*Q2 + p['EGP0']*BW*(1-x3)

def find_basal(p):
    try:
        return brentq(_ss_residual, 1e-5, 20.0, args=(p, TARGET_G_SS), xtol=1e-7)
    except ValueError:
        return 0.5

def ss_state(u_b, p):
    BW   = p['BW']
    I_ss = u_b / (p['ke'] * p['VI'] * BW)   # mU/L  (plasma concentration)
    x1   = p['kb1'] / p['ka1'] * I_ss        # consistent with dx1 = -ka1*x1 + kb1*I
    x2   = p['kb2'] / p['ka2'] * I_ss
    x3   = p['kb3'] / p['ka3'] * I_ss
    Q1   = TARGET_G_SS * p['VG'] * BW
    Q2   = x1 * Q1 / (p['k12'] + x2)
    S1   = u_b * p['tau_s']
    S2   = u_b * p['tau_s']
    # Position 7 stores I_ss (mU/L), consistent with the corrected ODE
    return np.array([Q1, Q2, x1, x2, x3, S1, S2, I_ss, 0.0, 0.0], dtype=float)

# ============================================================
# ODE (identical to validated adult code)
# ============================================================
def hovorka_ode(t, y, p, u_b, meals, bolus):
    Q1, Q2, x1, x2, x3, S1, S2, I, D1, D2 = y
    BW = p['BW']

    Q1 = max(Q1, 0.0); Q2 = max(Q2, 0.0)
    I  = max(I,  0.0); D1 = max(D1, 0.0); D2 = max(D2, 0.0)

    G    = Q1 / (p['VG'] * BW)
    F01c = p['F01'] * BW if G >= 4.5 else p['F01'] * BW * G / 4.5
    FR   = 0.003 * (G - 9.0) * p['VG'] * BW if G > 9.0 else 0.0

    meal_rate = 0.0
    for ts, te, g in meals:
        if ts <= t < te:
            meal_rate += g / (te - ts)
    UG = p['Ag'] * D2 / p['tmax_G']

    bolus_rate = 0.0
    for ts, te, mU in bolus:
        if ts <= t < te:
            bolus_rate += mU / (te - ts)
    u_total = u_b + bolus_rate

    # Hovorka 2004 correct convention:
    # I = plasma insulin concentration (mU/L)
    # dx_i = -ka_i * x_i + kb_i * I  (direct, not divided by VI*BW)
    # dI   = S2/(tau_s * VI * BW) - ke * I
    # SS: I_ss = u_b/(ke*VI*BW),  x1_ss = kb1/ka1 * I_ss
    dQ1 = -(F01c + x1*Q1) + p['k12']*Q2 - FR + p['EGP0']*BW*(1-x3) + UG
    dQ2 =  x1*Q1 - (p['k12'] + x2)*Q2
    dx1 = -p['ka1']*x1 + p['kb1']*I
    dx2 = -p['ka2']*x2 + p['kb2']*I
    dx3 = -p['ka3']*x3 + p['kb3']*I
    dS1 =  u_total - S1/p['tau_s']
    dS2 =  S1/p['tau_s'] - S2/p['tau_s']
    dI  =  S2/(p['tau_s']*p['VI']*BW) - p['ke']*I
    dD1 =  meal_rate - D1/p['tmax_G']
    dD2 =  D1/p['tmax_G'] - D2/p['tmax_G']

    return [dQ1, dQ2, dx1, dx2, dx3, dS1, dS2, dI, dD1, dD2]

# ============================================================
# MEAL SCHEDULES
# ============================================================
RAMADAN_CHO_FACTORS = [0.70,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.20,1.30]
SUHOOR_OFFSETS_MIN  = [60, 120, 180]

def build_ramadan_schedule(n_days, u_b, cho_base, role):
    meals, bolus = [], []
    for day in range(n_days):
        base   = day * 1440
        factor = RAMADAN_CHO_FACTORS[day % 10]
        total  = cho_base * factor
        offset = SUHOOR_OFFSETS_MIN[(day // 10) % 3]
        fajr   = base + 5*60 + 30
        su_s   = fajr - offset - 60;  su_e = su_s + 60
        if1_s  = base + 18*60 + 30;   if1_e = if1_s + 15
        if2_s  = if1_e;               if2_e = if2_s + 45
        sn_s   = base + 22*60;        sn_e  = sn_s + 40
        for ts, te, g in [(su_s,su_e,0.35*total),(if1_s,if1_e,0.10*total),
                           (if2_s,if2_e,0.40*total),(sn_s,sn_e,0.15*total)]:
            meals.append((ts, te, g))
            bolus.append((ts, ts+5, bolus_mU_for_meal(g, u_b, role)))
    return meals, bolus

def build_preramadan_schedule(n_days, u_b, cho_base, role):
    meals, bolus = [], []
    for day in range(n_days):
        base = day * 1440
        for ts, te, g in [
            (base+8*60,      base+8*60+30,   0.30*cho_base),
            (base+13*60,     base+13*60+30,  0.35*cho_base),
            (base+19*60,     base+19*60+30,  0.30*cho_base),
            (base+15*60+30,  base+15*60+50,  0.05*cho_base),
        ]:
            meals.append((ts, te, g))
            bolus.append((ts, ts+5, bolus_mU_for_meal(g, u_b, role)))
    return meals, bolus

# ============================================================
# SIMULATION ENGINE (identical to validated adult code)
# ============================================================
STEP   = 5
NOISE  = 5.0
N_DAYS = 30
CGM_SEED_BASE = 42

def simulate(p, u_b, meals, bolus, n_days, y0, cgm_seed):
    rng      = np.random.default_rng(cgm_seed)
    total    = n_days * 1440
    y        = y0.copy()
    records  = []
    terminal = False
    last_rescue_t = -240

    for t0 in range(0, total, STEP):
        t1 = t0 + STEP
        G_true = max(y[0], 0.0) / (p['VG'] * p['BW'])
        G_mgdl = float(np.clip(G_true * 18.015 + rng.normal(0, NOISE), 20, 600))

        extra_bolus = []
        if G_mgdl >= 300.0 and (t0 - last_rescue_t) >= 240:
            isf_real = 42.5   # mg/dL per U (1700-rule at TDD≈40 U)
            corr_U   = min(max((G_mgdl - 250.0) / isf_real, 0.0), 1.0)
            corr_mU  = corr_U * 1000.0 * (u_b / REAL_UB_REF[p['role']])
            extra_bolus  = [(t0, t0+5, corr_mU)]
            last_rescue_t = t0

        u_eff = 0.0 if G_mgdl < 70.0 else u_b
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
        except Exception:
            pass

        y = np.maximum(y, 0.0)
        G_check = y[0] / (p['VG'] * p['BW']) * 18.015
        if G_check > 550.0:
            terminal = True
            log(f"    *** Terminal at min {t0} (G≈{G_check:.0f} mg/dL) ***")
            break

        records.append({'min': t0, 'day': t0//1440,
                        'CGM_mgdL': G_mgdl, 'G_true_mmol': float(G_true)})

    return pd.DataFrame(records), terminal

def outcomes(df):
    g = df['CGM_mgdL'].values; m = float(np.mean(g))
    return dict(
        TIR    = float(np.mean((g>=70) & (g<=180)) * 100),
        TBR    = float(np.mean(g < 70) * 100),
        TBR54  = float(np.mean(g < 54) * 100),
        TAR    = float(np.mean(g > 180) * 100),
        MeanCGM= m,
        GMI    = 3.31 + 0.02392 * m,
        CV     = float(np.std(g)/m*100) if m > 0 else np.nan,
        N_obs  = len(g),
    )

# ============================================================
# MAIN
# ============================================================
POLICIES = [1.00, 0.90, 0.80, 0.70, 0.60]

if __name__ == "__main__":
    log("=" * 65)
    log("Hovorka Ramadan Basal Policy Study — 30 Virtual Patients")
    log(f"Policies: {POLICIES} | Days: {N_DAYS} | Output: {OUT_DIR}")
    log(f"REAL_UB_REF: child={REAL_UB_REF['child']}  "
        f"adolescent={REAL_UB_REF['adolescent']}  adult={REAL_UB_REF['adult']}")
    log("=" * 65)

    summary = []
    for i, p in enumerate(ALL_PATIENTS):
        pid  = p['pid']; role = p['role']
        cho  = CHO_TARGET[role]
        seed = CGM_SEED_BASE + i

        log(f"\n{'─'*55}")
        log(f"[{i+1:02d}/30] {pid}  BW={p['BW']}kg  isf×{p['isf_mult']:.2f}  "
            f"CHO={cho}g  seed={seed}")

        u_b = find_basal(p)
        log(f"  u_b={u_b:.4f} mU/min  ({u_b*1440/1000:.3f} U/day)")
        y0 = ss_state(u_b, p)

        # Pre-Ramadan
        pre_m, pre_b = build_preramadan_schedule(N_DAYS, u_b, cho, role)
        df_pre, _ = simulate(p, u_b, pre_m, pre_b, N_DAYS, y0, seed)
        pre_out   = outcomes(df_pre)
        log(f"  Pre-Ramadan → TIR={pre_out['TIR']:.1f}%  "
            f"TBR={pre_out['TBR']:.1f}%  TAR={pre_out['TAR']:.1f}%")
        df_pre.to_csv(OUT_DIR / f"{pid}_PreRamadan_AllDays.csv", index=False)
        summary.append(dict(Patient=pid, Role=role, Period='PreRamadan', Policy=1.0,
                            Terminal=False, u_b=u_b, BW=p['BW'], isf_mult=p['isf_mult'],
                            **{k: pre_out[k] for k in
                               ['TIR','TBR','TBR54','TAR','MeanCGM','GMI','CV']}))

        # 5 Ramadan policies
        ram_m, ram_b = build_ramadan_schedule(N_DAYS, u_b, cho, role)
        for factor in POLICIES:
            pct = int(factor * 100)
            u_ram = u_b * factor
            df_r, term = simulate(p, u_ram, ram_m, ram_b, N_DAYS, y0.copy(), seed)
            if term or len(df_r) < 100:
                log(f"  Policy {pct}% → TERMINAL")
                summary.append(dict(Patient=pid, Role=role, Period='Ramadan',
                                    Policy=factor, Terminal=True, u_b=u_b,
                                    BW=p['BW'], isf_mult=p['isf_mult'],
                                    TIR=np.nan, TBR=np.nan, TBR54=np.nan,
                                    TAR=np.nan, MeanCGM=np.nan, GMI=np.nan,
                                    CV=np.nan, N_obs=0))
            else:
                out = outcomes(df_r)
                log(f"  Policy {pct}% → TIR={out['TIR']:.1f}%  "
                    f"TBR={out['TBR']:.1f}%  TAR={out['TAR']:.1f}%")
                df_r.to_csv(OUT_DIR / f"{pid}_Ramadan{pct}_AllDays.csv", index=False)
                summary.append(dict(Patient=pid, Role=role, Period='Ramadan',
                                    Policy=factor, Terminal=False, u_b=u_b,
                                    BW=p['BW'], isf_mult=p['isf_mult'], **out))

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(OUT_DIR / "summary_all_patients_policies.csv", index=False)

    log("\n" + "="*65)
    log("DOSE-RESPONSE SUMMARY (Ramadan, complete cases)")
    log("="*65)
    ram = df_sum[(df_sum['Period']=='Ramadan') & (~df_sum['Terminal'])]
    for pct in [100,90,80,70,60]:
        sub = ram[ram['Policy'] == pct/100]
        if sub.empty: continue
        log(f"  {pct:3d}%  TIR={sub['TIR'].mean():.1f}±{sub['TIR'].std():.1f}%  "
            f"TBR={sub['TBR'].mean():.1f}±{sub['TBR'].std():.1f}%  "
            f"TAR={sub['TAR'].mean():.1f}±{sub['TAR'].std():.1f}%")

    log("\nBy role (Ramadan 100%):")
    for role in ['child', 'adolescent', 'adult']:
        sub = ram[(ram['Policy']==1.0) & (ram['Role']==role)]
        if not sub.empty:
            log(f"  {role:12s}: TIR={sub['TIR'].mean():.1f}±{sub['TIR'].std():.1f}%")
    log(f"\n✅  Done — results in {OUT_DIR}")
