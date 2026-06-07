# -*- coding: utf-8 -*-
"""
08_economic_sensitivity_rerun_optimization_grid_pso.py

Backup thesis script: economic sensitivity WITH re-optimization.

Purpose
-------
This script reruns the BESS sizing optimization for each economic sensitivity case,
using BOTH:
1) deterministic grid search, and
2) Particle Swarm Optimization (PSO), as an independent comparison method.

This is different from a post-processing sensitivity analysis, because each
sensitivity case re-evaluates candidate P_BESS and E_BESS using the modified
cost parameters.

Recommended backup use for thesis
---------------------------------
Run R5 only first, because R5 is usually the main baseline case:

python 08_economic_sensitivity_rerun_optimization_grid_pso.py \
    --input soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv \
    --scenario R5 \
    --case-set extended

Optional all-ramp run:

python 08_economic_sensitivity_rerun_optimization_grid_pso.py \
    --input soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv \
    --all-ramps \
    --case-set extended

Main outputs
------------
- economic_sensitivity_reopt_outputs/08_sensitivity_cases.csv
- economic_sensitivity_reopt_outputs/08_grid_reopt_summary.csv
- economic_sensitivity_reopt_outputs/08_pso_reopt_summary.csv
- economic_sensitivity_reopt_outputs/08_grid_pso_reopt_comparison.csv
- economic_sensitivity_reopt_outputs/08_pso_convergence_<case>_<scenario>.csv

Notes
-----
- Positive uniform scaling of all BESS costs or discount rate changes may not
  change the optimum physical size when the feasible technical set is unchanged,
  because all feasible candidates are still ranked almost identically.
- Separate changes to power-component cost and energy-component cost can change
  P/E trade-offs if the feasible set contains alternatives with different P/E ratios.
- This script is therefore useful as a backup verification, not necessarily as a
  required main thesis result.
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

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func


# =============================================================================
# BASELINE THESIS PARAMETERS
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
DURATION_MIN = 60.0       # 60 min is consistent with 1C E >= P.
MIN_BESS_POWER_FRAC = 0.10
P_MIN_MW = MIN_BESS_POWER_FRAC * PV_CAPACITY_MW
P_MAX_MW = 35.0
E_MIN_MWH = P_MIN_MW
E_MAX_MWH = 90.0

# Baseline economic assumptions
BASE_POWER_COST_USD_PER_MW = 427_420.0
BASE_ENERGY_COST_USD_PER_MWH = 307_990.0
BASE_FIXED_OM_FRAC_PER_YEAR = 0.0207
BASE_DISCOUNT_RATE = 0.08
PROJECT_LIFE_YEARS = 20

# Grid search settings
P_STEP_COARSE = 0.50
E_STEP_COARSE = 0.50
P_STEP_FINE = 0.10
E_STEP_FINE = 0.10
P_FINE_WINDOW_MW = 2.0
E_FINE_WINDOW_MWH = 5.0

# Constraint settings
MAX_RAMP_VIOLATIONS = 0
RAMP_TOL = 1e-6

# PSO settings
DEFAULT_N_PARTICLES = 30
DEFAULT_N_ITER = 60
DEFAULT_SEED = 42
ROUND_STEP = 0.1
PENALTY_INVALID = 1e12
PENALTY_PER_RAMP_VIOLATION = 1e9
PENALTY_SOC = 1e11
PENALTY_C_RATE = 1e11

DEFAULT_INPUT = "soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv"
DEFAULT_OUTPUT_DIR = "economic_sensitivity_reopt_outputs"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class EconomicCase:
    case_id: str
    description: str
    power_cost_multiplier: float
    energy_cost_multiplier: float
    fixed_om_frac_per_year: float
    discount_rate: float

    @property
    def power_cost_usd_per_mw(self) -> float:
        return BASE_POWER_COST_USD_PER_MW * self.power_cost_multiplier

    @property
    def energy_cost_usd_per_mwh(self) -> float:
        return BASE_ENERGY_COST_USD_PER_MWH * self.energy_cost_multiplier


@dataclass
class OptResult:
    case_id: str
    description: str
    method: str
    scenario: str
    ramp_pct_per_min: float
    ramp_limit_mw_per_min: float
    power_cost_usd_per_mw: float
    energy_cost_usd_per_mwh: float
    fixed_om_frac_per_year: float
    discount_rate: float
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
    n_evaluations: int
    runtime_s: float


# =============================================================================
# IO AND BASIC FUNCTIONS
# =============================================================================

def load_pv_profile(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input PV CSV not found: {csv_path}\n"
            "Use --input to point to your final 1-minute SoDa PV CSV, e.g. "
            "soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv"
        )

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    for col in ["HighRes", "Daya_Sintetik_MW", "PV_MW", "pv_mw"]:
        if col in df.columns:
            s = df[col].copy()
            break
    else:
        # If the file has one numeric column, use it as a practical fallback.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 1:
            s = df[numeric_cols[0]].copy()
        else:
            raise ValueError(
                f"PV column not found. Available columns: {df.columns.tolist()}. "
                "Expected HighRes, Daya_Sintetik_MW, PV_MW, or one numeric column."
            )

    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = s.astype(float).fillna(0.0)
    s[s < 0] = 0.0
    return s


def capital_recovery_factor(r: float, n: int) -> float:
    if abs(r) < 1e-15:
        return 1.0 / n
    return (r * (1.0 + r) ** n) / ((1.0 + r) ** n - 1.0)


def calc_costs(p_mw: float, e_mwh: float, econ: EconomicCase) -> tuple[float, float]:
    capex = p_mw * econ.power_cost_usd_per_mw + e_mwh * econ.energy_cost_usd_per_mwh
    crf = capital_recovery_factor(econ.discount_rate, PROJECT_LIFE_YEARS)
    annual = capex * crf + econ.fixed_om_frac_per_year * capex
    return float(capex), float(annual)


def make_grid(start: float, stop: float, step: float) -> np.ndarray:
    if stop < start:
        return np.array([], dtype=float)
    n = int(round((stop - start) / step))
    return np.round(start + np.arange(n + 1) * step, 6)


def round_to_step(x: float, step: float = ROUND_STEP) -> float:
    return float(np.round(np.round(x / step) * step, 6))


def candidate_basic_valid(p: float, e: float) -> bool:
    if p < P_MIN_MW - 1e-12 or p > P_MAX_MW + 1e-12:
        return False
    if e < E_MIN_MWH - 1e-12 or e > E_MAX_MWH + 1e-12:
        return False
    if e <= 0.0 or p <= 0.0:
        return False
    if (p / e) > C_RATE_MAX + 1e-12:
        return False
    if e < p * DURATION_MIN / 60.0 - 1e-12:
        return False
    return True


def repair_candidate(p: float, e: float) -> tuple[float, float]:
    p = float(np.clip(p, P_MIN_MW, P_MAX_MW))
    e = float(np.clip(e, E_MIN_MWH, E_MAX_MWH))
    e_required = max(p / C_RATE_MAX, p * DURATION_MIN / 60.0, E_MIN_MWH)
    if e < e_required:
        e = e_required
    e = float(np.clip(e, E_MIN_MWH, E_MAX_MWH))
    return round_to_step(p), round_to_step(e)


# =============================================================================
# BESS SIMULATION CORE
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

        if delta > ramp_limit:
            req = -(delta - ramp_limit)       # charge/absorb
        elif delta < -ramp_limit:
            req = -(delta + ramp_limit)       # discharge/inject
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

    return discharge_energy, charge_energy, ramp_violations, max_ramp_after, min_soc_seen, max_soc_seen


# =============================================================================
# CANDIDATE EVALUATION
# =============================================================================

def evaluate_candidate(
    pv: np.ndarray,
    p: float,
    e: float,
    ramp_limit: float,
    scenario: str,
    econ: EconomicCase,
    method: str,
    fitness_value: float = math.nan,
    n_evaluations: int = 0,
    runtime_s: float = 0.0,
) -> OptResult:
    capex, annual = calc_costs(p, e, econ)

    if not candidate_basic_valid(p, e):
        return OptResult(
            case_id=econ.case_id,
            description=econ.description,
            method=method,
            scenario=scenario,
            ramp_pct_per_min=ramp_limit / PV_CAPACITY_MW,
            ramp_limit_mw_per_min=ramp_limit,
            power_cost_usd_per_mw=econ.power_cost_usd_per_mw,
            energy_cost_usd_per_mwh=econ.energy_cost_usd_per_mwh,
            fixed_om_frac_per_year=econ.fixed_om_frac_per_year,
            discount_rate=econ.discount_rate,
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
            n_evaluations=n_evaluations,
            runtime_s=runtime_s,
        )

    d_mwh, c_mwh, viol, max_ramp, min_soc, max_soc = simulate_bess_core(
        pv, p, e, ramp_limit, SOC_MIN, SOC_MAX, SOC_TARGET, ETA_ONE_WAY, RAMP_TOL
    )
    feasible = int(viol) <= MAX_RAMP_VIOLATIONS
    compliance = (1.0 - int(viol) / (len(pv) - 1)) * 100.0
    lcos = annual / d_mwh if d_mwh > 0 else math.inf
    duration = (e / p) * 60.0
    c_rate = p / e

    return OptResult(
        case_id=econ.case_id,
        description=econ.description,
        method=method,
        scenario=scenario,
        ramp_pct_per_min=ramp_limit / PV_CAPACITY_MW,
        ramp_limit_mw_per_min=ramp_limit,
        power_cost_usd_per_mw=econ.power_cost_usd_per_mw,
        energy_cost_usd_per_mwh=econ.energy_cost_usd_per_mwh,
        fixed_om_frac_per_year=econ.fixed_om_frac_per_year,
        discount_rate=econ.discount_rate,
        p_bess_mw=p,
        e_bess_mwh=e,
        duration_min=duration,
        c_rate=c_rate,
        feasible=feasible,
        violations=int(viol),
        compliance_pct=compliance,
        max_ramp_after_mw_per_min=float(max_ramp),
        min_soc=float(min_soc),
        max_soc=float(max_soc),
        annual_discharge_mwh=float(d_mwh),
        charged_energy_mwh=float(c_mwh),
        capex_usd=capex,
        annualized_cost_usd_per_year=annual,
        lcos_usd_per_mwh=lcos,
        fitness_value=fitness_value,
        n_evaluations=n_evaluations,
        runtime_s=runtime_s,
    )


# =============================================================================
# GRID SEARCH
# =============================================================================

def grid_search_stage(
    pv: np.ndarray,
    p_values: Iterable[float],
    e_values: Iterable[float],
    ramp_limit: float,
    scenario: str,
    econ: EconomicCase,
) -> tuple[OptResult | None, int]:
    best = None
    n_eval = 0
    for p in p_values:
        for e in e_values:
            if not candidate_basic_valid(float(p), float(e)):
                continue
            n_eval += 1
            res = evaluate_candidate(pv, float(p), float(e), ramp_limit, scenario, econ, method="grid_search")
            if not res.feasible:
                continue
            if best is None or res.annualized_cost_usd_per_year < best.annualized_cost_usd_per_year:
                best = res
    return best, n_eval


def run_grid_reopt(pv: np.ndarray, scenario: str, econ: EconomicCase) -> OptResult:
    t0 = time.time()
    ramp_limit = RAMP_SCENARIOS[scenario] * PV_CAPACITY_MW

    # Warm up numba on a small slice.
    _ = simulate_bess_core(
        pv[: min(len(pv), 2000)], P_MIN_MW, E_MIN_MWH, ramp_limit,
        SOC_MIN, SOC_MAX, SOC_TARGET, ETA_ONE_WAY, RAMP_TOL
    )

    p_coarse = make_grid(P_MIN_MW, P_MAX_MW, P_STEP_COARSE)
    e_coarse = make_grid(E_MIN_MWH, E_MAX_MWH, E_STEP_COARSE)
    best_coarse, n1 = grid_search_stage(pv, p_coarse, e_coarse, ramp_limit, scenario, econ)
    if best_coarse is None:
        raise RuntimeError(f"No feasible grid candidate found for {econ.case_id} {scenario} in coarse search.")

    p0, e0 = best_coarse.p_bess_mw, best_coarse.e_bess_mwh
    p_fine = make_grid(max(P_MIN_MW, p0 - P_FINE_WINDOW_MW), min(P_MAX_MW, p0 + P_FINE_WINDOW_MW), P_STEP_FINE)
    e_fine = make_grid(max(E_MIN_MWH, e0 - E_FINE_WINDOW_MWH), min(E_MAX_MWH, e0 + E_FINE_WINDOW_MWH), E_STEP_FINE)
    best_fine, n2 = grid_search_stage(pv, p_fine, e_fine, ramp_limit, scenario, econ)
    if best_fine is None:
        best_fine = best_coarse

    runtime = time.time() - t0
    best_fine.n_evaluations = n1 + n2
    best_fine.runtime_s = runtime
    return best_fine


# =============================================================================
# PSO
# =============================================================================

def objective_with_penalty(pv: np.ndarray, p_raw: float, e_raw: float, ramp_limit: float, econ: EconomicCase) -> float:
    p, e = repair_candidate(p_raw, e_raw)
    capex, annual = calc_costs(p, e, econ)
    if not candidate_basic_valid(p, e):
        return annual + PENALTY_INVALID

    d_mwh, c_mwh, viol, max_ramp, min_soc, max_soc = simulate_bess_core(
        pv, p, e, ramp_limit, SOC_MIN, SOC_MAX, SOC_TARGET, ETA_ONE_WAY, RAMP_TOL
    )

    penalty = 0.0
    if int(viol) > MAX_RAMP_VIOLATIONS:
        penalty += PENALTY_PER_RAMP_VIOLATION * int(viol)
    if min_soc < SOC_MIN - 1e-9:
        penalty += PENALTY_SOC * (SOC_MIN - min_soc)
    if max_soc > SOC_MAX + 1e-9:
        penalty += PENALTY_SOC * (max_soc - SOC_MAX)
    if (p / e) > C_RATE_MAX + 1e-12:
        penalty += PENALTY_C_RATE * ((p / e) - C_RATE_MAX)
    return float(annual + penalty)


def run_pso_reopt(
    pv: np.ndarray,
    scenario: str,
    econ: EconomicCase,
    out_dir: Path,
    n_particles: int,
    n_iter: int,
    seed: int,
) -> OptResult:
    rng = np.random.default_rng(seed)
    ramp_limit = RAMP_SCENARIOS[scenario] * PV_CAPACITY_MW

    # Warm up numba.
    _ = simulate_bess_core(
        pv[: min(len(pv), 2000)], P_MIN_MW, E_MIN_MWH, ramp_limit,
        SOC_MIN, SOC_MAX, SOC_TARGET, ETA_ONE_WAY, RAMP_TOL
    )

    bounds_min = np.array([P_MIN_MW, E_MIN_MWH], dtype=float)
    bounds_max = np.array([P_MAX_MW, E_MAX_MWH], dtype=float)
    span = bounds_max - bounds_min

    x = bounds_min + rng.random((n_particles, 2)) * span
    for i in range(n_particles):
        x[i, 0], x[i, 1] = repair_candidate(x[i, 0], x[i, 1])

    v = (rng.random((n_particles, 2)) - 0.5) * 0.20 * span
    v_max = 0.20 * span

    pbest = x.copy()
    pbest_val = np.full(n_particles, np.inf)
    gbest = None
    gbest_val = np.inf

    w, c1, c2 = 0.72, 1.49, 1.49
    convergence_rows = []
    n_eval = 0
    t0 = time.time()

    for it in range(1, n_iter + 1):
        for i in range(n_particles):
            val = objective_with_penalty(pv, x[i, 0], x[i, 1], ramp_limit, econ)
            n_eval += 1
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest[i] = x[i].copy()
            if val < gbest_val:
                gbest_val = val
                gbest = x[i].copy()

        convergence_rows.append({
            "case_id": econ.case_id,
            "scenario": scenario,
            "iteration": it,
            "best_fitness_value": float(gbest_val),
            "best_p_mw": float(gbest[0]),
            "best_e_mwh": float(gbest[1]),
        })

        r1 = rng.random((n_particles, 2))
        r2 = rng.random((n_particles, 2))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -v_max, v_max)
        x = x + v
        for i in range(n_particles):
            x[i, 0], x[i, 1] = repair_candidate(x[i, 0], x[i, 1])

    runtime = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    conv_name = f"08_pso_convergence_{econ.case_id}_{scenario}.csv".replace("/", "_")
    pd.DataFrame(convergence_rows).to_csv(out_dir / conv_name, index=False)

    p_best, e_best = repair_candidate(float(gbest[0]), float(gbest[1]))
    return evaluate_candidate(
        pv, p_best, e_best, ramp_limit, scenario, econ,
        method="PSO", fitness_value=float(gbest_val), n_evaluations=n_eval, runtime_s=runtime
    )


# =============================================================================
# SENSITIVITY CASES
# =============================================================================

def build_economic_cases(case_set: str) -> list[EconomicCase]:
    """Create economic sensitivity cases.

    default  : baseline + conventional one-at-a-time cases.
    extended : default + separate power/energy cost cases and combined cases.
    cartesian: compact cartesian combination of power/energy cost and discount.
    """
    baseline = EconomicCase(
        "baseline", "Baseline thesis assumptions", 1.0, 1.0,
        BASE_FIXED_OM_FRAC_PER_YEAR, BASE_DISCOUNT_RATE,
    )

    default_cases = [
        baseline,
        EconomicCase("all_cost_minus20", "Power and energy cost -20%", 0.8, 0.8, BASE_FIXED_OM_FRAC_PER_YEAR, BASE_DISCOUNT_RATE),
        EconomicCase("all_cost_plus20", "Power and energy cost +20%", 1.2, 1.2, BASE_FIXED_OM_FRAC_PER_YEAR, BASE_DISCOUNT_RATE),
        EconomicCase("discount_6pct", "Discount rate 6%", 1.0, 1.0, BASE_FIXED_OM_FRAC_PER_YEAR, 0.06),
        EconomicCase("discount_10pct", "Discount rate 10%", 1.0, 1.0, BASE_FIXED_OM_FRAC_PER_YEAR, 0.10),
    ]

    if case_set == "default":
        return default_cases

    extended_extra = [
        EconomicCase("energy_cost_minus20", "Energy-component cost -20%", 1.0, 0.8, BASE_FIXED_OM_FRAC_PER_YEAR, BASE_DISCOUNT_RATE),
        EconomicCase("energy_cost_plus20", "Energy-component cost +20%", 1.0, 1.2, BASE_FIXED_OM_FRAC_PER_YEAR, BASE_DISCOUNT_RATE),
        EconomicCase("power_cost_minus20", "Power-component cost -20%", 0.8, 1.0, BASE_FIXED_OM_FRAC_PER_YEAR, BASE_DISCOUNT_RATE),
        EconomicCase("power_cost_plus20", "Power-component cost +20%", 1.2, 1.0, BASE_FIXED_OM_FRAC_PER_YEAR, BASE_DISCOUNT_RATE),
        EconomicCase("low_cost_low_discount", "All cost -20% and discount rate 6%", 0.8, 0.8, BASE_FIXED_OM_FRAC_PER_YEAR, 0.06),
        EconomicCase("high_cost_high_discount", "All cost +20% and discount rate 10%", 1.2, 1.2, BASE_FIXED_OM_FRAC_PER_YEAR, 0.10),
    ]

    if case_set == "extended":
        return default_cases + extended_extra

    if case_set == "cartesian":
        cases = [baseline]
        for p_mult in [0.8, 1.0, 1.2]:
            for e_mult in [0.8, 1.0, 1.2]:
                for dr in [0.06, 0.08, 0.10]:
                    if p_mult == 1.0 and e_mult == 1.0 and abs(dr - 0.08) < 1e-12:
                        continue
                    cases.append(EconomicCase(
                        f"p{p_mult:.1f}_e{e_mult:.1f}_r{int(round(dr*100))}pct".replace(".", "p"),
                        f"Power cost x{p_mult:.1f}, energy cost x{e_mult:.1f}, discount {dr:.0%}",
                        p_mult, e_mult, BASE_FIXED_OM_FRAC_PER_YEAR, dr,
                    ))
        return cases

    raise ValueError(f"Unknown case_set: {case_set}")


# =============================================================================
# COMPARISON OUTPUT
# =============================================================================

def make_comparison(grid_df: pd.DataFrame, pso_df: pd.DataFrame) -> pd.DataFrame:
    if grid_df.empty or pso_df.empty:
        return pd.DataFrame()

    g = grid_df.copy()
    p = pso_df.copy()
    key = ["case_id", "scenario"]
    keep_cols = [
        "case_id", "scenario", "p_bess_mw", "e_bess_mwh", "annualized_cost_usd_per_year",
        "feasible", "violations", "max_ramp_after_mw_per_min", "annual_discharge_mwh",
        "lcos_usd_per_mwh", "runtime_s",
    ]
    g = g[keep_cols].add_prefix("grid_")
    p = p[keep_cols].add_prefix("pso_")
    merged = g.merge(
        p,
        left_on=["grid_case_id", "grid_scenario"],
        right_on=["pso_case_id", "pso_scenario"],
        how="outer",
    )
    merged["case_id"] = merged["grid_case_id"].combine_first(merged["pso_case_id"])
    merged["scenario"] = merged["grid_scenario"].combine_first(merged["pso_scenario"])
    merged["delta_p_mw_pso_minus_grid"] = merged["pso_p_bess_mw"] - merged["grid_p_bess_mw"]
    merged["delta_e_mwh_pso_minus_grid"] = merged["pso_e_bess_mwh"] - merged["grid_e_bess_mwh"]
    merged["delta_annual_cost_usd_pso_minus_grid"] = merged["pso_annualized_cost_usd_per_year"] - merged["grid_annualized_cost_usd_per_year"]
    merged["delta_annual_cost_pct_pso_minus_grid"] = (
        merged["delta_annual_cost_usd_pso_minus_grid"] / merged["grid_annualized_cost_usd_per_year"] * 100.0
    )
    first_cols = ["case_id", "scenario", "delta_p_mw_pso_minus_grid", "delta_e_mwh_pso_minus_grid", "delta_annual_cost_usd_pso_minus_grid", "delta_annual_cost_pct_pso_minus_grid"]
    other_cols = [c for c in merged.columns if c not in first_cols]
    return merged[first_cols + other_cols]


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Economic sensitivity with re-optimized grid search and PSO.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Final 1-minute SoDa PV CSV.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", type=str, default="R5", choices=list(RAMP_SCENARIOS.keys()))
    parser.add_argument("--all-ramps", action="store_true", help="Run R20, R10, R5, and R3.")
    parser.add_argument("--case-set", type=str, default="extended", choices=["default", "extended", "cartesian"])
    parser.add_argument("--skip-pso", action="store_true", help="Run grid search only.")
    parser.add_argument("--particles", type=int, default=DEFAULT_N_PARTICLES)
    parser.add_argument("--iterations", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pv_series = load_pv_profile(Path(args.input))
    pv = pv_series.values.astype(np.float64)

    scenarios = list(RAMP_SCENARIOS.keys()) if args.all_ramps else [args.scenario]
    cases = build_economic_cases(args.case_set)

    pd.DataFrame([asdict(c) | {
        "power_cost_usd_per_mw": c.power_cost_usd_per_mw,
        "energy_cost_usd_per_mwh": c.energy_cost_usd_per_mwh,
    } for c in cases]).to_csv(out_dir / "08_sensitivity_cases.csv", index=False)

    grid_results: list[OptResult] = []
    pso_results: list[OptResult] = []

    print(f"Loaded PV profile: {len(pv):,} timesteps from {pv_series.index.min()} to {pv_series.index.max()}")
    print(f"Scenarios: {scenarios}")
    print(f"Economic cases: {len(cases)} ({args.case_set})")

    for econ_idx, econ in enumerate(cases):
        for scenario in scenarios:
            print(f"\n[GRID] case={econ.case_id} scenario={scenario}")
            grid_res = run_grid_reopt(pv, scenario, econ)
            grid_results.append(grid_res)
            print(
                f"  -> P={grid_res.p_bess_mw:.1f} MW, E={grid_res.e_bess_mwh:.1f} MWh, "
                f"annual={grid_res.annualized_cost_usd_per_year:,.0f} USD/y, feasible={grid_res.feasible}"
            )

            if not args.skip_pso:
                # Make each case-scenario reproducible but not identical in random stream.
                pso_seed = int(args.seed + 1000 * econ_idx + 17 * list(RAMP_SCENARIOS.keys()).index(scenario))
                print(f"[PSO ] case={econ.case_id} scenario={scenario} seed={pso_seed}")
                pso_res = run_pso_reopt(
                    pv, scenario, econ, out_dir,
                    n_particles=args.particles,
                    n_iter=args.iterations,
                    seed=pso_seed,
                )
                pso_results.append(pso_res)
                print(
                    f"  -> P={pso_res.p_bess_mw:.1f} MW, E={pso_res.e_bess_mwh:.1f} MWh, "
                    f"annual={pso_res.annualized_cost_usd_per_year:,.0f} USD/y, feasible={pso_res.feasible}"
                )

    grid_df = pd.DataFrame([asdict(r) for r in grid_results])
    pso_df = pd.DataFrame([asdict(r) for r in pso_results])
    comp_df = make_comparison(grid_df, pso_df)

    grid_df.to_csv(out_dir / "08_grid_reopt_summary.csv", index=False)
    if not pso_df.empty:
        pso_df.to_csv(out_dir / "08_pso_reopt_summary.csv", index=False)
    if not comp_df.empty:
        comp_df.to_csv(out_dir / "08_grid_pso_reopt_comparison.csv", index=False)

    # Excel workbook for convenient checking.
    xlsx_path = out_dir / "08_economic_sensitivity_reopt_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([asdict(c) | {
            "power_cost_usd_per_mw": c.power_cost_usd_per_mw,
            "energy_cost_usd_per_mwh": c.energy_cost_usd_per_mwh,
        } for c in cases]).to_excel(writer, sheet_name="cases", index=False)
        grid_df.to_excel(writer, sheet_name="grid_reopt", index=False)
        if not pso_df.empty:
            pso_df.to_excel(writer, sheet_name="pso_reopt", index=False)
        if not comp_df.empty:
            comp_df.to_excel(writer, sheet_name="grid_pso_comparison", index=False)

    print("\nDONE")
    print(f"Output folder: {out_dir.resolve()}")
    print(f"Excel summary: {xlsx_path.resolve()}")


if __name__ == "__main__":
    main()
