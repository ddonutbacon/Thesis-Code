# -*- coding: utf-8 -*-
"""
05c_bess_pso_deb_rules_multiseed_4dp.py

PSO comparison script for thesis BESS sizing using Deb's Feasibility Rules.

Purpose
-------
This script is designed as the final/clean PSO comparator for the thesis workflow.
It keeps the same BESS ramp-rate smoothing logic and economic assumptions used in
main grid-search sizing, but handles constraints using Deb's feasibility rules
instead of large death-penalty constants.

Key features
------------
1. Deb's Feasibility Rules for constrained PSO:
   - feasible solution beats infeasible solution;
   - if both feasible, lower annualized cost wins;
   - if both infeasible, lower cumulative violation wins.
2. Multi-seed support in one run.
3. Four-decimal candidate resolution by default (ROUND_STEP = 0.0001 MW/MWh).
4. No huge penalty constants.
5. Outputs all-run summary, scenario-level summary, grid-vs-PSO deltas,
   convergence CSVs/plots, and best time series per scenario.
6. Plot style follows thesis supervisor preference:
   - no title inside figure;
   - only red, black, or blue;
   - different line styles for readability in black-and-white printing.

Default use
-----------
python 05c_bess_pso_deb_rules_multiseed_4dp.py \
    --input soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv \
    --scenario ALL \
    --seeds 1,2,3,4,5,6,7,8,9,10 \
    --particles 30 \
    --iters 60 \
    --grid-summary bess_sizing_outputs/03_bess_sizing_ramp_scenarios.csv \
    --output pso_deb_multiseed_outputs
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func


# =============================================================================
# THESIS PARAMETERS
# =============================================================================

PV_CAPACITY_MW = 100.0

RAMP_SCENARIOS = {
    "R20": 0.20,
    "R10": 0.10,
    "R5":  0.05,
    "R3":  0.03,
}

# BESS technical assumptions
SOC_MIN = 0.20
SOC_MAX = 0.90
SOC_TARGET = 0.55
ETA_ROUND_TRIP = 0.95
ETA_ONE_WAY = ETA_ROUND_TRIP ** 0.5
C_RATE_MAX = 1.0
DURATION_MIN = 60.0  # minutes; 1C-consistent sizing
MIN_BESS_POWER_FRAC = 0.10
P_MIN_MW = MIN_BESS_POWER_FRAC * PV_CAPACITY_MW  # 10 MW
P_MAX_MW = 35.0
E_MIN_MWH = P_MIN_MW
E_MAX_MWH = 90.0

# Economic assumptions
CAPEX_POWER_USD_PER_MW = 427_420.0
CAPEX_ENERGY_USD_PER_MWH = 307_990.0
FIXED_OM_FRAC_PER_YEAR = 0.0207
DISCOUNT_RATE = 0.08
PROJECT_LIFE_YEARS = 20

# PSO settings
DEFAULT_N_PARTICLES = 30
DEFAULT_N_ITER = 60
DEFAULT_SEEDS = "1,2,3,4,5,6,7,8,9,10"
DEFAULT_ROUND_STEP = 0.0001  # MW/MWh; four-decimal candidate resolution

# Numerical tolerance
RAMP_TOL = 1e-6
MAX_RAMP_VIOLATIONS = 0
VIOL_TOL = 1e-12

DEFAULT_INPUT = "soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv"
DEFAULT_GRID_SUMMARY = "bess_sizing_outputs/03_bess_sizing_ramp_scenarios.csv"
DEFAULT_OUTPUT_DIR = "pso_deb_multiseed_outputs"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PSOResult:
    scenario: str
    seed: int
    ramp_pct_per_min: float
    ramp_limit_mw_per_min: float
    method: str
    p_bess_mw: float
    e_bess_mwh: float
    duration_min: float
    c_rate: float
    feasible: bool
    violation_score: float
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
    best_cost_value: float
    best_violation_value: float
    n_particles: int
    n_iter: int
    n_function_evaluations: int
    runtime_s: float


# =============================================================================
# BASIC FUNCTIONS
# =============================================================================

def parse_seed_list(seed_text: str) -> list[int]:
    seeds = []
    for part in seed_text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid seed value: {part!r}") from exc
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


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
    if not csv_path.exists():
        raise FileNotFoundError(f"Input PV file not found: {csv_path}")

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
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s[s < 0] = 0.0
    return s.astype(float)


def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return float(np.round(np.round(x / step) * step, 8))


def repair_candidate(p: float, e: float, round_step: float) -> tuple[float, float]:
    """Keep candidate inside bounds and enforce E >= P/C-rate and duration constraints.

    The repair function is intentionally retained because the physical candidate
    space must respect rating bounds and 1C/duration feasibility before the BESS
    time-domain simulation is evaluated. With round_step=0.0001, the reported
    PSO candidate can be shown to four decimals.
    """
    p = float(np.clip(p, P_MIN_MW, P_MAX_MW))
    e = float(np.clip(e, E_MIN_MWH, E_MAX_MWH))

    e_required = max(p / C_RATE_MAX, p * DURATION_MIN / 60.0, E_MIN_MWH)
    if e < e_required:
        e = e_required
    e = float(np.clip(e, E_MIN_MWH, E_MAX_MWH))

    p = round_to_step(p, round_step)
    e = round_to_step(e, round_step)

    e_required = round_to_step(max(p / C_RATE_MAX, p * DURATION_MIN / 60.0, E_MIN_MWH), round_step)
    if e < e_required:
        e = e_required
    e = round_to_step(min(e, E_MAX_MWH), round_step)
    return p, e


def candidate_basic_valid(p: float, e: float) -> bool:
    if p < P_MIN_MW - 1e-12 or p > P_MAX_MW + 1e-12:
        return False
    if e < E_MIN_MWH - 1e-12 or e > E_MAX_MWH + 1e-12:
        return False
    if p <= 0.0 or e <= 0.0:
        return False
    if (p / e) > C_RATE_MAX + 1e-12:
        return False
    if e < p * DURATION_MIN / 60.0 - 1e-12:
        return False
    return True


def basic_violation_score(p: float, e: float) -> float:
    """Normalized scalar violation for basic candidate constraints.

    This is only used if a candidate remains invalid after repair, which should
    be rare. It keeps Deb's rule comparison well-defined for infeasible points.
    """
    v = 0.0
    v += max(P_MIN_MW - p, 0.0) / max(P_MIN_MW, 1.0)
    v += max(p - P_MAX_MW, 0.0) / max(P_MAX_MW, 1.0)
    v += max(E_MIN_MWH - e, 0.0) / max(E_MIN_MWH, 1.0)
    v += max(e - E_MAX_MWH, 0.0) / max(E_MAX_MWH, 1.0)
    if p <= 0.0 or e <= 0.0:
        v += 1.0
    else:
        v += max((p / e) - C_RATE_MAX, 0.0)
        v += max((p * DURATION_MIN / 60.0) - e, 0.0) / max(e, 1.0)
    return float(v)


# =============================================================================
# BESS SIMULATION
# =============================================================================

@njit
def simulate_bess_core_deb(
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
    ramp_excess_sum = 0.0
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

        excess = actual_ramp - ramp_limit
        if excess > ramp_tol:
            ramp_violations += 1
            ramp_excess_sum += excess

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
        ramp_excess_sum,
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
# DEB'S FEASIBILITY RULES
# =============================================================================

def is_feasible(violation_score: float) -> bool:
    return np.isfinite(violation_score) and violation_score <= VIOL_TOL


def deb_is_better(new_cost: float, new_viol: float, old_cost: float, old_viol: float) -> bool:
    """Return True if new candidate is better than old according to Deb's rules."""
    new_feas = is_feasible(new_viol)
    old_feas = is_feasible(old_viol)

    if new_feas and not old_feas:
        return True
    if new_feas and old_feas:
        return new_cost < old_cost
    if (not new_feas) and (not old_feas):
        return new_viol < old_viol
    return False


def evaluate_cost_violation(pv: np.ndarray, p: float, e: float, ramp_limit: float) -> dict:
    capex, annual = calc_capex_annualized_cost(p, e)

    if not candidate_basic_valid(p, e):
        return {
            "cost": annual,
            "violation_score": 1e9 + basic_violation_score(p, e),
            "violations": 10**9,
            "ramp_excess_sum": math.inf,
            "max_ramp_after": math.inf,
            "min_soc": math.nan,
            "max_soc": math.nan,
            "annual_discharge_mwh": 0.0,
            "charged_energy_mwh": 0.0,
        }

    d_mwh, c_mwh, viol, ramp_excess_sum, max_ramp, min_soc, max_soc = simulate_bess_core_deb(
        pv, p, e, ramp_limit, SOC_MIN, SOC_MAX, SOC_TARGET, ETA_ONE_WAY, RAMP_TOL
    )

    # Deb violation score: not added to cost; used only for feasibility comparison.
    # Ramp violation count dominates; normalized ramp excess helps differentiate
    # candidates with the same violation count.
    violation_score = 0.0
    if viol > MAX_RAMP_VIOLATIONS:
        violation_score += float(viol - MAX_RAMP_VIOLATIONS)
        violation_score += float(ramp_excess_sum) / max(ramp_limit, 1e-9)

    # These terms are normally zero because SOC is clamped and candidates are repaired,
    # but they are retained for explicit constraint tracking.
    if min_soc < SOC_MIN - 1e-9:
        violation_score += (SOC_MIN - min_soc) * 100.0
    if max_soc > SOC_MAX + 1e-9:
        violation_score += (max_soc - SOC_MAX) * 100.0
    if (p / e) > C_RATE_MAX + 1e-12:
        violation_score += ((p / e) - C_RATE_MAX) * 10.0

    return {
        "cost": annual,
        "violation_score": float(violation_score),
        "violations": int(viol),
        "ramp_excess_sum": float(ramp_excess_sum),
        "max_ramp_after": float(max_ramp),
        "min_soc": float(min_soc),
        "max_soc": float(max_soc),
        "annual_discharge_mwh": float(d_mwh),
        "charged_energy_mwh": float(c_mwh),
    }


def build_result(
    pv: np.ndarray,
    scenario: str,
    seed: int,
    p: float,
    e: float,
    ramp_limit: float,
    best_cost_value: float,
    best_violation_value: float,
    n_particles: int,
    n_iter: int,
    n_evals: int,
    runtime_s: float,
    round_step: float,
) -> PSOResult:
    p, e = repair_candidate(p, e, round_step)
    capex, annual = calc_capex_annualized_cost(p, e)
    ev = evaluate_cost_violation(pv, p, e, ramp_limit)

    feasible = is_feasible(ev["violation_score"])
    compliance = (1.0 - int(ev["violations"]) / (len(pv) - 1)) * 100.0
    lcos = annual / ev["annual_discharge_mwh"] if ev["annual_discharge_mwh"] > 0.0 else math.inf

    return PSOResult(
        scenario=scenario,
        seed=seed,
        ramp_pct_per_min=ramp_limit / PV_CAPACITY_MW,
        ramp_limit_mw_per_min=ramp_limit,
        method="PSO_DebRules_4dp",
        p_bess_mw=p,
        e_bess_mwh=e,
        duration_min=(e / p) * 60.0,
        c_rate=p / e,
        feasible=feasible,
        violation_score=ev["violation_score"],
        violations=int(ev["violations"]),
        compliance_pct=compliance,
        max_ramp_after_mw_per_min=ev["max_ramp_after"],
        min_soc=ev["min_soc"],
        max_soc=ev["max_soc"],
        annual_discharge_mwh=ev["annual_discharge_mwh"],
        charged_energy_mwh=ev["charged_energy_mwh"],
        capex_usd=capex,
        annualized_cost_usd_per_year=annual,
        lcos_usd_per_mwh=lcos,
        best_cost_value=best_cost_value,
        best_violation_value=best_violation_value,
        n_particles=n_particles,
        n_iter=n_iter,
        n_function_evaluations=n_evals,
        runtime_s=runtime_s,
    )


# =============================================================================
# PSO IMPLEMENTATION
# =============================================================================

def run_pso_for_scenario_seed(
    pv: np.ndarray,
    scenario: str,
    seed: int,
    n_particles: int,
    n_iter: int,
    out_dir: Path,
    save_trace_index: pd.DatetimeIndex,
    round_step: float,
) -> tuple[PSOResult, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ramp_limit = RAMP_SCENARIOS[scenario] * PV_CAPACITY_MW

    # Warm-up numba before timing the PSO loop.
    _ = simulate_bess_core_deb(
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

    x = bounds_min + rng.random((n_particles, 2)) * span
    for i in range(n_particles):
        x[i, 0], x[i, 1] = repair_candidate(x[i, 0], x[i, 1], round_step)

    # Small random initial velocity.
    v = (rng.random((n_particles, 2)) - 0.5) * 0.20 * span
    v_max = 0.20 * span

    pbest = x.copy()
    pbest_cost = np.full(n_particles, np.inf)
    pbest_viol = np.full(n_particles, np.inf)

    gbest: np.ndarray | None = None
    gbest_cost = np.inf
    gbest_viol = np.inf

    convergence_rows = []
    n_evals = 0
    t0 = time.time()

    for it in range(n_iter):
        # Inertia decreases from 0.90 to 0.40.
        w = 0.90 - (0.50 * it / (n_iter - 1)) if n_iter > 1 else 0.70
        c1 = 1.50
        c2 = 1.50

        feasible_count = 0
        iter_best_cost = np.inf
        iter_best_viol = np.inf
        iter_best_xy: np.ndarray | None = None

        for i in range(n_particles):
            p, e = repair_candidate(x[i, 0], x[i, 1], round_step)
            x[i, 0], x[i, 1] = p, e
            ev = evaluate_cost_violation(pv, p, e, ramp_limit)
            n_evals += 1

            current_cost = ev["cost"]
            current_viol = ev["violation_score"]

            if is_feasible(current_viol):
                feasible_count += 1

            if deb_is_better(current_cost, current_viol, pbest_cost[i], pbest_viol[i]):
                pbest_cost[i] = current_cost
                pbest_viol[i] = current_viol
                pbest[i, :] = x[i, :].copy()

            if deb_is_better(current_cost, current_viol, gbest_cost, gbest_viol):
                gbest_cost = current_cost
                gbest_viol = current_viol
                gbest = x[i, :].copy()

            if deb_is_better(current_cost, current_viol, iter_best_cost, iter_best_viol):
                iter_best_cost = current_cost
                iter_best_viol = current_viol
                iter_best_xy = x[i, :].copy()

        if gbest is None:
            raise RuntimeError("PSO failed to initialize a global best candidate.")
        if iter_best_xy is None:
            iter_best_xy = gbest.copy()
            iter_best_cost = gbest_cost
            iter_best_viol = gbest_viol

        convergence_rows.append({
            "scenario": scenario,
            "seed": seed,
            "iteration": it + 1,
            "best_cost": float(gbest_cost),
            "best_violation": float(gbest_viol),
            "best_p_mw": float(gbest[0]),
            "best_e_mwh": float(gbest[1]),
            "iteration_best_cost": float(iter_best_cost),
            "iteration_best_violation": float(iter_best_viol),
            "iteration_best_p_mw": float(iter_best_xy[0]),
            "iteration_best_e_mwh": float(iter_best_xy[1]),
            "feasible_particles": int(feasible_count),
        })

        # Update velocity and position.
        r1 = rng.random((n_particles, 2))
        r2 = rng.random((n_particles, 2))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -v_max, v_max)
        x = x + v
        x = np.clip(x, bounds_min, bounds_max)

        for i in range(n_particles):
            x[i, 0], x[i, 1] = repair_candidate(x[i, 0], x[i, 1], round_step)

        if (it + 1) == 1 or (it + 1) % 5 == 0 or (it + 1) == n_iter:
            status = f"cost={gbest_cost:,.2f}" if is_feasible(gbest_viol) else f"viol={gbest_viol:.6f}"
            print(
                f"{scenario} | seed={seed} | iter {it+1:03d}/{n_iter} | "
                f"{status} | P={gbest[0]:.4f} MW | E={gbest[1]:.4f} MWh | "
                f"feasible={feasible_count}/{n_particles}"
            )

    runtime_s = time.time() - t0
    result = build_result(
        pv=pv,
        scenario=scenario,
        seed=seed,
        p=float(gbest[0]),
        e=float(gbest[1]),
        ramp_limit=ramp_limit,
        best_cost_value=gbest_cost,
        best_violation_value=gbest_viol,
        n_particles=n_particles,
        n_iter=n_iter,
        n_evals=n_evals,
        runtime_s=runtime_s,
        round_step=round_step,
    )

    conv_df = pd.DataFrame(convergence_rows)
    conv_csv = out_dir / f"08_pso_deb_convergence_{scenario}_seed{seed}.csv"
    conv_df.to_csv(conv_csv, index=False)

    # Convergence plot: no title inside figure, red/blue/black only.
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(conv_df["iteration"], conv_df["best_cost"], color="blue", linestyle="-", linewidth=1.4, label="Best cost")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Annualized cost (USD/year)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(conv_df["iteration"], conv_df["best_violation"], color="red", linestyle="--", linewidth=1.2, label="Constraint violation")
    ax2.set_ylabel("Violation score")

    # Combined legend for both axes.
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.tight_layout()
    plt.savefig(out_dir / f"08_pso_deb_convergence_{scenario}_seed{seed}.png", dpi=180)
    plt.close()

    return result, conv_df


def save_best_timeseries_by_scenario(
    pv: np.ndarray,
    pv_index: pd.DatetimeIndex,
    best_results: Iterable[PSOResult],
    out_dir: Path,
) -> None:
    for res in best_results:
        ramp_limit = res.ramp_limit_mw_per_min
        p_grid, p_bess, soc, ramp = simulate_bess_trace(
            pv,
            res.p_bess_mw,
            res.e_bess_mwh,
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
        }, index=pv_index)
        ts.to_csv(out_dir / f"09_pso_deb_best_timeseries_{res.scenario}.csv")


def make_summary_by_scenario(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scen, group in all_df.groupby("scenario", sort=False):
        rows.append({
            "scenario": scen,
            "n_runs": int(len(group)),
            "n_feasible_runs": int(group["feasible"].sum()),
            "p_mean_mw": group["p_bess_mw"].mean(),
            "p_std_mw": group["p_bess_mw"].std(ddof=0),
            "p_min_mw": group["p_bess_mw"].min(),
            "p_max_mw": group["p_bess_mw"].max(),
            "e_mean_mwh": group["e_bess_mwh"].mean(),
            "e_std_mwh": group["e_bess_mwh"].std(ddof=0),
            "e_min_mwh": group["e_bess_mwh"].min(),
            "e_max_mwh": group["e_bess_mwh"].max(),
            "cost_mean_usd_per_year": group["annualized_cost_usd_per_year"].mean(),
            "cost_std_usd_per_year": group["annualized_cost_usd_per_year"].std(ddof=0),
            "cost_min_usd_per_year": group["annualized_cost_usd_per_year"].min(),
            "cost_max_usd_per_year": group["annualized_cost_usd_per_year"].max(),
            "violation_score_max": group["violation_score"].max(),
            "violations_max": group["violations"].max(),
            "compliance_min_pct": group["compliance_pct"].min(),
            "best_seed_by_cost": int(group.sort_values(["feasible", "annualized_cost_usd_per_year"], ascending=[False, True]).iloc[0]["seed"]),
        })
    return pd.DataFrame(rows)


def make_grid_deltas(all_df: pd.DataFrame, grid_summary_path: Path, out_dir: Path) -> pd.DataFrame:
    if not grid_summary_path.exists():
        print(f"Grid summary not found: {grid_summary_path}. Skipping grid comparison.")
        return pd.DataFrame()

    grid = pd.read_csv(grid_summary_path)
    rows = []
    for _, pso in all_df.iterrows():
        scen = pso["scenario"]
        g = grid[grid["scenario"] == scen]
        if g.empty:
            continue
        g = g.iloc[0]
        grid_p = float(g.get("p_bess_mw", np.nan))
        grid_e = float(g.get("e_bess_mwh", np.nan))
        grid_cost = float(g.get("annualized_cost_usd_per_year", np.nan))
        rows.append({
            "scenario": scen,
            "seed": int(pso["seed"]),
            "grid_p_bess_mw": grid_p,
            "pso_p_bess_mw": float(pso["p_bess_mw"]),
            "delta_p_mw": float(pso["p_bess_mw"] - grid_p),
            "grid_e_bess_mwh": grid_e,
            "pso_e_bess_mwh": float(pso["e_bess_mwh"]),
            "delta_e_mwh": float(pso["e_bess_mwh"] - grid_e),
            "grid_annualized_cost_usd_per_year": grid_cost,
            "pso_annualized_cost_usd_per_year": float(pso["annualized_cost_usd_per_year"]),
            "delta_cost_usd_per_year": float(pso["annualized_cost_usd_per_year"] - grid_cost),
            "grid_violations": int(g.get("violations", np.nan)),
            "pso_violations": int(pso["violations"]),
            "grid_compliance_pct": float(g.get("compliance_pct", np.nan)),
            "pso_compliance_pct": float(pso["compliance_pct"]),
            "pso_violation_score": float(pso["violation_score"]),
            "pso_feasible": bool(pso["feasible"]),
            "pso_runtime_s": float(pso["runtime_s"]),
            "pso_n_function_evaluations": int(pso["n_function_evaluations"]),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(out_dir / "pso_grid_deltas_all_runs.csv", index=False)
    return out


def select_best_by_scenario(all_df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for scen, group in all_df.groupby("scenario", sort=False):
        # Prefer feasible; among feasible use lowest annualized cost. If all infeasible,
        # choose lowest violation score.
        feasible = group[group["feasible"] == True]
        if not feasible.empty:
            best = feasible.sort_values(["annualized_cost_usd_per_year", "p_bess_mw", "e_bess_mwh"]).iloc[0]
        else:
            best = group.sort_values(["violation_score", "annualized_cost_usd_per_year"]).iloc[0]
        best_rows.append(best)
    return pd.DataFrame(best_rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="PSO Deb's Rules multi-seed BESS sizing comparator.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input SoDa final CSV.")
    parser.add_argument("--scenario", type=str, default="ALL", choices=["R20", "R10", "R5", "R3", "ALL"])
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--grid-summary", type=str, default=DEFAULT_GRID_SUMMARY, help="Grid search summary CSV for comparison.")
    parser.add_argument("--particles", type=int, default=DEFAULT_N_PARTICLES, help="Number of PSO particles.")
    parser.add_argument("--iters", type=int, default=DEFAULT_N_ITER, help="Number of PSO iterations.")
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS, help="Comma-separated seed list, e.g. 1,2,3,4,5.")
    parser.add_argument("--round-step", type=float, default=DEFAULT_ROUND_STEP, help="Candidate rounding step in MW/MWh. Default 0.0001 for 4 decimals.")
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    scenarios = ["R20", "R10", "R5", "R3"] if args.scenario == "ALL" else [args.scenario]

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PSO BESS SIZING WITH DEB'S FEASIBILITY RULES")
    print("=" * 80)
    print(f"Input PV profile : {input_path}")
    print(f"Output directory : {out_dir.resolve()}")
    print(f"Scenarios        : {scenarios}")
    print(f"Seeds            : {seeds}")
    print(f"Particles/iters  : {args.particles}/{args.iters}")
    print(f"Round step       : {args.round_step} MW/MWh")
    print(f"Numba available  : {NUMBA_AVAILABLE}")

    pv_series = load_pv_profile(input_path)
    pv = pv_series.values.astype(np.float64)
    print(f"Loaded timesteps : {len(pv_series):,}")
    print(f"Time range       : {pv_series.index.min()} -> {pv_series.index.max()}")
    print(f"PV max/mean      : {pv_series.max():.4f} / {pv_series.mean():.4f} MW")

    results: list[PSOResult] = []
    all_conv = []
    total_t0 = time.time()

    for scenario in scenarios:
        for seed in seeds:
            print("\n" + "-" * 80)
            print(f"Running {scenario} | seed={seed}")
            print("-" * 80)
            res, conv_df = run_pso_for_scenario_seed(
                pv=pv,
                scenario=scenario,
                seed=seed,
                n_particles=args.particles,
                n_iter=args.iters,
                out_dir=out_dir,
                save_trace_index=pv_series.index,
                round_step=args.round_step,
            )
            results.append(res)
            all_conv.append(conv_df)
            print(
                f"Final {scenario} seed={seed}: "
                f"P={res.p_bess_mw:.4f} MW | E={res.e_bess_mwh:.4f} MWh | "
                f"feasible={res.feasible} | viol={res.violations} | "
                f"score={res.violation_score:.6e} | "
                f"cost={res.annualized_cost_usd_per_year:,.2f} USD/year"
            )

    all_df = pd.DataFrame([asdict(r) for r in results])
    all_df.to_csv(out_dir / "summary_all_runs.csv", index=False)
    # Compatibility output name, in case existing workflow expects this file.
    all_df.to_csv(out_dir / "07_pso_summary_deb.csv", index=False)

    if all_conv:
        pd.concat(all_conv, ignore_index=True).to_csv(out_dir / "convergence_all_runs.csv", index=False)

    summary_df = make_summary_by_scenario(all_df)
    summary_df.to_csv(out_dir / "summary_by_scenario.csv", index=False)

    deltas_df = make_grid_deltas(all_df, Path(args.grid_summary), out_dir)

    best_df = select_best_by_scenario(all_df)
    best_df.to_csv(out_dir / "best_run_by_scenario.csv", index=False)
    best_results = [PSOResult(**row.to_dict()) for _, row in best_df.iterrows()]
    save_best_timeseries_by_scenario(pv, pv_series.index, best_results, out_dir)

    print("\n" + "=" * 80)
    print("SUMMARY BY SCENARIO")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    if not deltas_df.empty:
        print("\n" + "=" * 80)
        print("GRID VS PSO DELTAS - FIRST ROWS")
        print("=" * 80)
        print(deltas_df.head(12).to_string(index=False))

    print(f"\nTotal runtime: {(time.time() - total_t0) / 60.0:.2f} minutes")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
