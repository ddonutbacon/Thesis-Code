# -*- coding: utf-8 -*-
"""
05_bess_pso_comparison_R5.py

PSO comparison script for the thesis BESS sizing workflow.

Purpose
-------
This script adds Particle Swarm Optimization (PSO) as a comparison method
against the deterministic grid search used in the main thesis script.
It DOES NOT change the BESS control logic. The same ramp-rate smoothing
simulation is used; only the optimizer that searches P_BESS and E_BESS differs.

Recommended thesis use
----------------------
- Keep deterministic grid search as the main method.
- Use this PSO script as a comparison/validation method, preferably for R5.
- Compare PSO result against:
  bess_sizing_outputs/03_bess_sizing_ramp_scenarios.csv

Default use
-----------
python 05_bess_pso_comparison_R5.py \
    --input soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv \
    --scenario R5

Optional all-scenario run
-------------------------
python 05_bess_pso_comparison_R5.py --scenario ALL

Main outputs
------------
- pso_comparison_outputs/07_pso_summary.csv
- pso_comparison_outputs/07_pso_vs_grid_comparison.csv  (if grid result exists)
- pso_comparison_outputs/08_pso_convergence_<scenario>.csv
- pso_comparison_outputs/08_pso_convergence_<scenario>.png
- pso_comparison_outputs/09_pso_best_timeseries_<scenario>.csv

Important notes
---------------
1. PSO is stochastic. Use --seed to reproduce results.
2. Candidate P_BESS and E_BESS are rounded to 0.1 MW/MWh to make comparison
   with fine-grid deterministic search easier.
3. The objective function applies large penalties to infeasible candidates.
4. The economic parameters below are aligned with the latest thesis/PPT values.
   If your final book uses different values, update them here too.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func


# =============================================================================
# THESIS PARAMETERS - UPDATE ONLY IF FINAL BOOK VALUES CHANGE
# =============================================================================

PV_CAPACITY_MW = 100.0

RAMP_SCENARIOS = {
    "R20": 0.20,  # 20 MW/min for 100 MW PV
    "R10": 0.10,  # 10 MW/min
    "R5":  0.05,  # 5 MW/min, recommended PSO comparison baseline
    "R3":  0.03,  # 3 MW/min
}

# BESS technical assumptions
SOC_MIN = 0.20
SOC_MAX = 0.90
SOC_TARGET = 0.55
ETA_ROUND_TRIP = 0.95
ETA_ONE_WAY = ETA_ROUND_TRIP ** 0.5
C_RATE_MAX = 1.0
DURATION_MIN = 60.0  # minutes; 60 minutes is consistent with 1C E >= P sizing
MIN_BESS_POWER_FRAC = 0.10
P_MIN_MW = MIN_BESS_POWER_FRAC * PV_CAPACITY_MW  # 10 MW
P_MAX_MW = 35.0
E_MIN_MWH = P_MIN_MW
E_MAX_MWH = 90.0

# Economic assumptions aligned with latest PPT/book direction
CAPEX_POWER_USD_PER_MW = 427_420.0
CAPEX_ENERGY_USD_PER_MWH = 307_990.0
FIXED_OM_FRAC_PER_YEAR = 0.0207
DISCOUNT_RATE = 0.08
PROJECT_LIFE_YEARS = 20

# PSO settings
DEFAULT_N_PARTICLES = 30
DEFAULT_N_ITER = 60
DEFAULT_SEED = 42
ROUND_STEP = 0.1  # MW/MWh candidate rounding for easier thesis comparison

# Penalty settings. These are deliberately much larger than annualized cost.
PENALTY_INVALID = 1e12
PENALTY_PER_RAMP_VIOLATION = 1e9
PENALTY_SOC = 1e11
PENALTY_C_RATE = 1e11
RAMP_TOL = 1e-6
MAX_RAMP_VIOLATIONS = 0

DEFAULT_INPUT = "soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv"
DEFAULT_GRID_SUMMARY = "bess_sizing_outputs/03_bess_sizing_ramp_scenarios.csv"
DEFAULT_OUTPUT_DIR = "pso_comparison_outputs"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PSOResult:
    scenario: str
    ramp_pct_per_min: float
    ramp_limit_mw_per_min: float
    method: str
    p_bess_mw: float
    e_bess_mwh: float
    duration_min: float
    c_rate: float
    feasible: bool
    violations: int
    compliance_pct: float
    max_ramp_after_mw_per_min: float
    min_soc: float
    max_soc: float
    annual_discharge_mwh: float
    charged_energy_mwh: float
    capex_usd: float
    annualized_cost_usd_per_year: float
    lcos_usd_per_mwh: float
    fitness_value: float
    n_particles: int
    n_iter: int
    seed: int
    n_function_evaluations: int
    runtime_s: float


# =============================================================================
# BASIC FUNCTIONS
# =============================================================================

def capital_recovery_factor(r: float, n: int) -> float:
    if r == 0:
        return 1.0 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def calc_capex_annualized_cost(p_bess_mw: float, e_bess_mwh: float) -> tuple[float, float]:
    capex = (
        p_bess_mw * CAPEX_POWER_USD_PER_MW
        + e_bess_mwh * CAPEX_ENERGY_USD_PER_MWH
    )
    crf = capital_recovery_factor(DISCOUNT_RATE, PROJECT_LIFE_YEARS)
    annual = capex * crf + FIXED_OM_FRAC_PER_YEAR * capex
    return capex, annual


def load_pv_profile(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if "HighRes" in df.columns:
        s = df["HighRes"].copy()
    elif "Daya_Sintetik_MW" in df.columns:
        s = df["Daya_Sintetik_MW"].copy()
    elif "PV_MW" in df.columns:
        s = df["PV_MW"].copy()
    else:
        raise ValueError(
            f"PV column not found. Available columns: {df.columns.tolist()}. "
            "Expected 'HighRes', 'Daya_Sintetik_MW', or 'PV_MW'."
        )
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = s.astype(float).fillna(0.0)
    s[s < 0] = 0.0
    return s


def round_to_step(x: float, step: float = ROUND_STEP) -> float:
    return float(np.round(np.round(x / step) * step, 6))


def repair_candidate(p: float, e: float) -> tuple[float, float]:
    """Keep candidate inside bounds and enforce basic E >= P/C-rate and duration constraints."""
    p = float(np.clip(p, P_MIN_MW, P_MAX_MW))
    e = float(np.clip(e, E_MIN_MWH, E_MAX_MWH))

    # Enforce the minimum energy required by C-rate and duration.
    e_required = max(p / C_RATE_MAX, p * DURATION_MIN / 60.0, E_MIN_MWH)
    if e < e_required:
        e = e_required

    e = float(np.clip(e, E_MIN_MWH, E_MAX_MWH))

    # Round after repair for comparability with grid search.
    p = round_to_step(p)
    e = round_to_step(e)

    # A second repair after rounding.
    e_required = round_to_step(max(p / C_RATE_MAX, p * DURATION_MIN / 60.0, E_MIN_MWH))
    if e < e_required:
        e = e_required
    e = round_to_step(min(e, E_MAX_MWH))
    return p, e


def candidate_basic_valid(p: float, e: float) -> bool:
    if p < P_MIN_MW or p > P_MAX_MW:
        return False
    if e < E_MIN_MWH or e > E_MAX_MWH:
        return False
    if p <= 0.0 or e <= 0.0:
        return False
    if (p / e) > C_RATE_MAX + 1e-12:
        return False
    if e < p * DURATION_MIN / 60.0 - 1e-12:
        return False
    return True


# =============================================================================
# BESS SIMULATION - SAME CONTROL LOGIC AS GRID SEARCH WORKFLOW
# =============================================================================

@njit
def simulate_bess_core(
    pv: np.ndarray,
    p_inv_max: float,
    e_bat_max: float,
    ramp_limit: float,
    soc_min: float,
    soc_max: float,
    soc_target: float,
    eta_ow: float,
    ramp_tol: float,
):
    n = len(pv)
    dt = 1.0 / 60.0

    soc_prev = soc_target
    p_grid_prev = pv[0]

    discharge_energy = 0.0
    charge_energy = 0.0
    ramp_violations = 0
    max_ramp_after = 0.0
    min_soc_seen = soc_prev
    max_soc_seen = soc_prev

    for t in range(1, n):
        delta = pv[t] - p_grid_prev
        req = 0.0

        # Main ramp-rate control.
        if delta > ramp_limit:
            req = -(delta - ramp_limit)       # charge/absorb power
        elif delta < -ramp_limit:
            req = -(delta + ramp_limit)       # discharge/inject power
        else:
            # Gentle SOC recovery while preserving ramp limit.
            soc_err = soc_prev - soc_target
            max_rec = 0.10 * p_inv_max
            abs_err = soc_err if soc_err >= 0.0 else -soc_err
            ratio = abs_err / 0.10
            if ratio > 1.0:
                ratio = 1.0
            rec_pwr = ratio * max_rec

            if soc_err > 0.05:
                req = rec_pwr       # discharge gently
            elif soc_err < -0.05:
                req = -rec_pwr      # charge gently

            test_ramp = (pv[t] + req) - p_grid_prev
            if test_ramp > ramp_limit:
                req = ramp_limit - delta
            elif test_ramp < -ramp_limit:
                req = -(delta + ramp_limit)

        # Power limit.
        p_bess = req
        if p_bess > p_inv_max:
            p_bess = p_inv_max
        elif p_bess < -p_inv_max:
            p_bess = -p_inv_max

        # SOC update. Positive P_BESS = discharge.
        if p_bess >= 0.0:
            e_change = (p_bess / eta_ow) * dt
        else:
            e_change = (p_bess * eta_ow) * dt

        soc_new = soc_prev - (e_change / e_bat_max)

        # SOC lower clamp.
        if soc_new < soc_min:
            available_mwh = (soc_prev - soc_min) * e_bat_max
            if available_mwh < 0.0:
                available_mwh = 0.0
            p_bess = available_mwh * eta_ow / dt
            soc_new = soc_min

        # SOC upper clamp.
        elif soc_new > soc_max:
            room_mwh = (soc_max - soc_prev) * e_bat_max
            if room_mwh < 0.0:
                room_mwh = 0.0
            p_bess = -room_mwh / eta_ow / dt
            soc_new = soc_max

        p_grid = pv[t] + p_bess
        actual_ramp = p_grid - p_grid_prev
        if actual_ramp < 0.0:
            actual_ramp = -actual_ramp

        if actual_ramp > max_ramp_after:
            max_ramp_after = actual_ramp
        if actual_ramp > ramp_limit + ramp_tol:
            ramp_violations += 1

        if p_bess > 0.0:
            discharge_energy += p_bess * dt
        elif p_bess < 0.0:
            charge_energy += (-p_bess) * dt

        soc_prev = soc_new
        p_grid_prev = p_grid

        if soc_prev < min_soc_seen:
            min_soc_seen = soc_prev
        if soc_prev > max_soc_seen:
            max_soc_seen = soc_prev

    return (
        discharge_energy,
        charge_energy,
        ramp_violations,
        max_ramp_after,
        min_soc_seen,
        max_soc_seen,
    )


@njit
def simulate_bess_trace(
    pv: np.ndarray,
    p_inv_max: float,
    e_bat_max: float,
    ramp_limit: float,
    soc_min: float,
    soc_max: float,
    soc_target: float,
    eta_ow: float,
    ramp_tol: float,
):
    n = len(pv)
    dt = 1.0 / 60.0

    p_grid_arr = np.zeros(n)
    p_bess_arr = np.zeros(n)
    soc_arr = np.zeros(n)
    ramp_arr = np.zeros(n)

    soc_prev = soc_target
    p_grid_prev = pv[0]
    p_grid_arr[0] = p_grid_prev
    soc_arr[0] = soc_prev

    for t in range(1, n):
        delta = pv[t] - p_grid_prev
        req = 0.0

        if delta > ramp_limit:
            req = -(delta - ramp_limit)
        elif delta < -ramp_limit:
            req = -(delta + ramp_limit)
        else:
            soc_err = soc_prev - soc_target
            max_rec = 0.10 * p_inv_max
            abs_err = soc_err if soc_err >= 0.0 else -soc_err
            ratio = abs_err / 0.10
            if ratio > 1.0:
                ratio = 1.0
            rec_pwr = ratio * max_rec

            if soc_err > 0.05:
                req = rec_pwr
            elif soc_err < -0.05:
                req = -rec_pwr

            test_ramp = (pv[t] + req) - p_grid_prev
            if test_ramp > ramp_limit:
                req = ramp_limit - delta
            elif test_ramp < -ramp_limit:
                req = -(delta + ramp_limit)

        p_bess = req
        if p_bess > p_inv_max:
            p_bess = p_inv_max
        elif p_bess < -p_inv_max:
            p_bess = -p_inv_max

        if p_bess >= 0.0:
            e_change = (p_bess / eta_ow) * dt
        else:
            e_change = (p_bess * eta_ow) * dt

        soc_new = soc_prev - (e_change / e_bat_max)

        if soc_new < soc_min:
            available_mwh = (soc_prev - soc_min) * e_bat_max
            if available_mwh < 0.0:
                available_mwh = 0.0
            p_bess = available_mwh * eta_ow / dt
            soc_new = soc_min
        elif soc_new > soc_max:
            room_mwh = (soc_max - soc_prev) * e_bat_max
            if room_mwh < 0.0:
                room_mwh = 0.0
            p_bess = -room_mwh / eta_ow / dt
            soc_new = soc_max

        p_grid = pv[t] + p_bess
        rr = p_grid - p_grid_prev
        ramp_arr[t] = -rr if rr < 0.0 else rr

        p_bess_arr[t] = p_bess
        p_grid_arr[t] = p_grid
        soc_arr[t] = soc_new

        soc_prev = soc_new
        p_grid_prev = p_grid

    return p_grid_arr, p_bess_arr, soc_arr, ramp_arr


# =============================================================================
# OBJECTIVE AND RESULT EVALUATION
# =============================================================================

def evaluate_candidate_full(
    pv: np.ndarray,
    p: float,
    e: float,
    ramp_limit: float,
    scenario: str,
    fitness_value: float,
    n_particles: int,
    n_iter: int,
    seed: int,
    n_evals: int,
    runtime_s: float,
) -> PSOResult:
    p, e = repair_candidate(p, e)
    capex, annual = calc_capex_annualized_cost(p, e)

    if not candidate_basic_valid(p, e):
        return PSOResult(
            scenario=scenario,
            ramp_pct_per_min=ramp_limit / PV_CAPACITY_MW,
            ramp_limit_mw_per_min=ramp_limit,
            method="PSO",
            p_bess_mw=p,
            e_bess_mwh=e,
            duration_min=math.inf,
            c_rate=math.inf,
            feasible=False,
            violations=10**9,
            compliance_pct=0.0,
            max_ramp_after_mw_per_min=math.inf,
            min_soc=math.nan,
            max_soc=math.nan,
            annual_discharge_mwh=0.0,
            charged_energy_mwh=0.0,
            capex_usd=capex,
            annualized_cost_usd_per_year=annual,
            lcos_usd_per_mwh=math.inf,
            fitness_value=fitness_value,
            n_particles=n_particles,
            n_iter=n_iter,
            seed=seed,
            n_function_evaluations=n_evals,
            runtime_s=runtime_s,
        )

    d_mwh, c_mwh, viol, max_ramp, min_soc, max_soc = simulate_bess_core(
        pv, p, e, ramp_limit, SOC_MIN, SOC_MAX, SOC_TARGET, ETA_ONE_WAY, RAMP_TOL
    )
    feasible = int(viol) <= MAX_RAMP_VIOLATIONS
    compliance = (1.0 - int(viol) / (len(pv) - 1)) * 100.0
    lcos = annual / d_mwh if d_mwh > 0.0 else math.inf
    duration = (e / p) * 60.0
    c_rate = p / e

    return PSOResult(
        scenario=scenario,
        ramp_pct_per_min=ramp_limit / PV_CAPACITY_MW,
        ramp_limit_mw_per_min=ramp_limit,
        method="PSO",
        p_bess_mw=p,
        e_bess_mwh=e,
        duration_min=duration,
        c_rate=c_rate,
        feasible=feasible,
        violations=int(viol),
        compliance_pct=compliance,
        max_ramp_after_mw_per_min=max_ramp,
        min_soc=min_soc,
        max_soc=max_soc,
        annual_discharge_mwh=d_mwh,
        charged_energy_mwh=c_mwh,
        capex_usd=capex,
        annualized_cost_usd_per_year=annual,
        lcos_usd_per_mwh=lcos,
        fitness_value=fitness_value,
        n_particles=n_particles,
        n_iter=n_iter,
        seed=seed,
        n_function_evaluations=n_evals,
        runtime_s=runtime_s,
    )


def objective_with_penalty(pv: np.ndarray, p_raw: float, e_raw: float, ramp_limit: float) -> float:
    p, e = repair_candidate(p_raw, e_raw)
    capex, annual = calc_capex_annualized_cost(p, e)

    penalty = 0.0
    if not candidate_basic_valid(p, e):
        return annual + PENALTY_INVALID

    d_mwh, c_mwh, viol, max_ramp, min_soc, max_soc = simulate_bess_core(
        pv, p, e, ramp_limit, SOC_MIN, SOC_MAX, SOC_TARGET, ETA_ONE_WAY, RAMP_TOL
    )

    if viol > MAX_RAMP_VIOLATIONS:
        penalty += PENALTY_PER_RAMP_VIOLATION * viol
    if min_soc < SOC_MIN - 1e-9:
        penalty += PENALTY_SOC * (SOC_MIN - min_soc)
    if max_soc > SOC_MAX + 1e-9:
        penalty += PENALTY_SOC * (max_soc - SOC_MAX)
    if (p / e) > C_RATE_MAX:
        penalty += PENALTY_C_RATE * ((p / e) - C_RATE_MAX)

    return annual + penalty


# =============================================================================
# PSO IMPLEMENTATION
# =============================================================================

def run_pso_for_scenario(
    pv: np.ndarray,
    scenario: str,
    n_particles: int,
    n_iter: int,
    seed: int,
    out_dir: Path,
    save_trace_index: pd.DatetimeIndex,
) -> PSOResult:
    rng = np.random.default_rng(seed)
    ramp_limit = RAMP_SCENARIOS[scenario] * PV_CAPACITY_MW

    # Warm-up numba before timing the PSO loop.
    _ = simulate_bess_core(
        pv[: min(len(pv), 2000)],
        P_MIN_MW,
        E_MIN_MWH,
        ramp_limit,
        SOC_MIN,
        SOC_MAX,
        SOC_TARGET,
        ETA_ONE_WAY,
        RAMP_TOL,
    )

    bounds_min = np.array([P_MIN_MW, E_MIN_MWH], dtype=float)
    bounds_max = np.array([P_MAX_MW, E_MAX_MWH], dtype=float)
    span = bounds_max - bounds_min

    # Initialize particles.
    x = bounds_min + rng.random((n_particles, 2)) * span
    for i in range(n_particles):
        x[i, 0], x[i, 1] = repair_candidate(x[i, 0], x[i, 1])

    # Small random initial velocity.
    v = (rng.random((n_particles, 2)) - 0.5) * 0.20 * span
    v_max = 0.20 * span

    pbest = x.copy()
    pbest_val = np.full(n_particles, np.inf)
    gbest = None
    gbest_val = np.inf

    convergence_rows = []
    n_evals = 0
    t0 = time.time()

    for it in range(n_iter):
        # Inertia decreases from 0.90 to 0.40.
        if n_iter > 1:
            w = 0.90 - (0.50 * it / (n_iter - 1))
        else:
            w = 0.70
        c1 = 1.50
        c2 = 1.50

        feasible_count = 0
        iter_best_val = np.inf
        iter_best_xy = None

        for i in range(n_particles):
            p, e = repair_candidate(x[i, 0], x[i, 1])
            x[i, 0], x[i, 1] = p, e

            val = objective_with_penalty(pv, p, e, ramp_limit)
            n_evals += 1

            # Check feasibility for monitoring only.
            d_mwh, c_mwh, viol, max_ramp, min_soc, max_soc = simulate_bess_core(
                pv, p, e, ramp_limit, SOC_MIN, SOC_MAX, SOC_TARGET, ETA_ONE_WAY, RAMP_TOL
            )
            if candidate_basic_valid(p, e) and viol <= MAX_RAMP_VIOLATIONS:
                feasible_count += 1

            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest[i, :] = x[i, :]
            if val < gbest_val:
                gbest_val = val
                gbest = x[i, :].copy()
            if val < iter_best_val:
                iter_best_val = val
                iter_best_xy = x[i, :].copy()

        convergence_rows.append({
            "iteration": it + 1,
            "best_fitness": gbest_val,
            "iteration_best_fitness": iter_best_val,
            "best_p_mw": float(gbest[0]),
            "best_e_mwh": float(gbest[1]),
            "iteration_best_p_mw": float(iter_best_xy[0]),
            "iteration_best_e_mwh": float(iter_best_xy[1]),
            "feasible_particles": feasible_count,
        })

        # Update velocity and position.
        r1 = rng.random((n_particles, 2))
        r2 = rng.random((n_particles, 2))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -v_max, v_max)
        x = x + v
        x = np.clip(x, bounds_min, bounds_max)

        for i in range(n_particles):
            x[i, 0], x[i, 1] = repair_candidate(x[i, 0], x[i, 1])

        print(
            f"{scenario} | iter {it+1:03d}/{n_iter} | "
            f"best fitness={gbest_val:,.2f} | "
            f"P={gbest[0]:.2f} MW | E={gbest[1]:.2f} MWh | "
            f"feasible particles={feasible_count}/{n_particles}"
        )

    runtime_s = time.time() - t0
    best_p, best_e = repair_candidate(float(gbest[0]), float(gbest[1]))
    result = evaluate_candidate_full(
        pv=pv,
        p=best_p,
        e=best_e,
        ramp_limit=ramp_limit,
        scenario=scenario,
        fitness_value=gbest_val,
        n_particles=n_particles,
        n_iter=n_iter,
        seed=seed,
        n_evals=n_evals,
        runtime_s=runtime_s,
    )

    # Save convergence data and plot.
    conv_df = pd.DataFrame(convergence_rows)
    conv_csv = out_dir / f"08_pso_convergence_{scenario}.csv"
    conv_df.to_csv(conv_csv, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(conv_df["iteration"], conv_df["best_fitness"])
    ax.set_title(f"PSO Convergence - {scenario}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness value")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"08_pso_convergence_{scenario}.png", dpi=180)
    plt.close()

    # Save best time series.
    p_grid, p_bess, soc, ramp = simulate_bess_trace(
        pv,
        result.p_bess_mw,
        result.e_bess_mwh,
        ramp_limit,
        SOC_MIN,
        SOC_MAX,
        SOC_TARGET,
        ETA_ONE_WAY,
        RAMP_TOL,
    )
    ts = pd.DataFrame({
        "PV_MW": pv,
        "P_BESS_MW": p_bess,
        "P_Grid_MW": p_grid,
        "SOC": soc,
        "Ramp_Grid_MW_per_min": ramp,
    }, index=save_trace_index)
    ts.to_csv(out_dir / f"09_pso_best_timeseries_{scenario}.csv")

    return result


# =============================================================================
# COMPARISON WITH GRID SEARCH
# =============================================================================

def make_grid_comparison(pso_df: pd.DataFrame, grid_summary_path: Path, out_dir: Path) -> None:
    if not grid_summary_path.exists():
        print(f"Grid summary not found: {grid_summary_path}. Skipping comparison CSV.")
        return

    grid = pd.read_csv(grid_summary_path)
    rows = []
    for _, pso in pso_df.iterrows():
        scen = pso["scenario"]
        g = grid[grid["scenario"] == scen]
        if g.empty:
            continue
        g = g.iloc[0]
        rows.append({
            "scenario": scen,
            "grid_p_bess_mw": g.get("p_bess_mw", np.nan),
            "pso_p_bess_mw": pso["p_bess_mw"],
            "delta_p_mw": pso["p_bess_mw"] - g.get("p_bess_mw", np.nan),
            "grid_e_bess_mwh": g.get("e_bess_mwh", np.nan),
            "pso_e_bess_mwh": pso["e_bess_mwh"],
            "delta_e_mwh": pso["e_bess_mwh"] - g.get("e_bess_mwh", np.nan),
            "grid_annualized_cost_usd_per_year": g.get("annualized_cost_usd_per_year", np.nan),
            "pso_annualized_cost_usd_per_year": pso["annualized_cost_usd_per_year"],
            "delta_cost_usd_per_year": pso["annualized_cost_usd_per_year"] - g.get("annualized_cost_usd_per_year", np.nan),
            "grid_violations": g.get("violations", np.nan),
            "pso_violations": pso["violations"],
            "grid_compliance_pct": g.get("compliance_pct", np.nan),
            "pso_compliance_pct": pso["compliance_pct"],
            "pso_n_function_evaluations": pso["n_function_evaluations"],
            "pso_runtime_s": pso["runtime_s"],
        })

    if rows:
        comp = pd.DataFrame(rows)
        comp.to_csv(out_dir / "07_pso_vs_grid_comparison.csv", index=False)
        print("Saved PSO vs grid comparison CSV.")
    else:
        print("No matching scenarios found in grid summary. Skipping comparison CSV.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="PSO comparison for BESS ramp-rate sizing.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input SoDa final CSV.")
    parser.add_argument("--scenario", type=str, default="R5", choices=["R20", "R10", "R5", "R3", "ALL"])
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--grid-summary", type=str, default=DEFAULT_GRID_SUMMARY, help="Grid search summary CSV for comparison.")
    parser.add_argument("--particles", type=int, default=DEFAULT_N_PARTICLES, help="Number of PSO particles.")
    parser.add_argument("--iters", type=int, default=DEFAULT_N_ITER, help="Number of PSO iterations.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PSO BESS SIZING COMPARISON")
    print("=" * 72)
    print(f"Input PV profile: {input_path}")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"Particles: {args.particles} | Iterations: {args.iters} | Seed: {args.seed}")

    pv_series = load_pv_profile(input_path)
    pv = pv_series.values.astype(np.float64)
    print(f"Loaded timesteps: {len(pv_series):,}")
    print(f"Time range: {pv_series.index.min()} -> {pv_series.index.max()}")
    print(f"PV max: {pv_series.max():.3f} MW | PV mean: {pv_series.mean():.3f} MW")

    if args.scenario == "ALL":
        scenarios = ["R20", "R10", "R5", "R3"]
    else:
        scenarios = [args.scenario]

    results: list[PSOResult] = []
    total_t0 = time.time()
    for scen in scenarios:
        print("\n" + "=" * 72)
        print(f"Running PSO for {scen}")
        print("=" * 72)
        res = run_pso_for_scenario(
            pv=pv,
            scenario=scen,
            n_particles=args.particles,
            n_iter=args.iters,
            seed=args.seed,
            out_dir=out_dir,
            save_trace_index=pv_series.index,
        )
        results.append(res)
        print(
            f"\nFinal PSO {scen}: P={res.p_bess_mw:.2f} MW | E={res.e_bess_mwh:.2f} MWh | "
            f"viol={res.violations} | compliance={res.compliance_pct:.5f}% | "
            f"annualized cost={res.annualized_cost_usd_per_year:,.2f} USD/year | "
            f"LCOS={res.lcos_usd_per_mwh:,.2f} USD/MWh"
        )

    pso_df = pd.DataFrame([asdict(r) for r in results])
    pso_df.to_csv(out_dir / "07_pso_summary.csv", index=False)
    make_grid_comparison(pso_df, Path(args.grid_summary), out_dir)

    print("\n" + "=" * 72)
    print("PSO SUMMARY")
    print("=" * 72)
    print(pso_df[[
        "scenario", "p_bess_mw", "e_bess_mwh", "duration_min", "c_rate",
        "violations", "compliance_pct", "annualized_cost_usd_per_year",
        "annual_discharge_mwh", "n_function_evaluations", "runtime_s"
    ]].to_string(index=False))
    print(f"\nTotal runtime: {(time.time() - total_t0) / 60:.2f} minutes")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
