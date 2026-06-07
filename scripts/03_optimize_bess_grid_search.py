# -*- coding: utf-8 -*-
"""
01_bess_sizing_ramp_scenarios.py

Core thesis simulation:
SoDa final 1-minute PV profile -> ramp-rate analysis -> BESS optimum sizing
using deterministic grid search for R20, R10, R5, and R3.

Main output:
- bess_sizing_outputs/01_pv_summary.csv
- bess_sizing_outputs/02_ramp_statistics_before_bess.csv
- bess_sizing_outputs/03_bess_sizing_ramp_scenarios.csv
- bess_sizing_outputs/04_top_feasible_candidates.csv
- bess_sizing_outputs/05_best_timeseries_R20.csv / R10 / R5 / R3

Use:
python 01_bess_sizing_ramp_scenarios.py --input soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv

perubahan 26 mei 2026
data ekonomi dari penurunan NREL ATB 2025. Penurunan regresi linier. 
Data original (file feasibility study PLTS Subang 100 MW)
CAPEX_INV_USD_PER_MW = 95_000.0
CAPEX_BAT_USD_PER_MWH = 273_000.0
FIXED_OM_FRAC_PER_YEAR = 0.025

Data NREL ATB 2025

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
# USER PARAMETERS
# =============================================================================

PV_CAPACITY_MW = 100.0

# R20 = Indonesian regulatory reference/check.
# R10, R5, R3 = technical sensitivity cases.
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
DURATION_MIN = 60.0  # minutes

# Economic assumptions
CAPEX_INV_USD_PER_MW = 427_420.0
CAPEX_BAT_USD_PER_MWH = 307_990.0
FIXED_OM_FRAC_PER_YEAR = 0.0207
DISCOUNT_RATE = 0.08
PROJECT_LIFE_YEARS = 20

# Grid search
P_STEP_COARSE = 0.50
E_STEP_COARSE = 0.50
P_STEP_FINE = 0.01
E_STEP_FINE = 0.01

# Indonesian grid-code minimum BESS size assumption:
# minimum BESS power capacity = 10% of installed PV capacity.
MIN_BESS_POWER_FRAC = 0.10
P_MIN_MW = MIN_BESS_POWER_FRAC * PV_CAPACITY_MW  # 10 MW for 100 MW PLTS

P_MAX_MW = 35.0

# Because C_RATE_MAX = 1.0, E_BESS must be at least P_BESS.
# Therefore, with P_MIN_MW = 10 MW, the effective minimum energy is 10 MWh.
E_MIN_MWH = P_MIN_MW
E_MAX_MWH = 90.0

P_FINE_WINDOW_MW = 0.5
E_FINE_WINDOW_MWH = 0.5

MAX_RAMP_VIOLATIONS = 0
RAMP_TOL = 1e-6

DEFAULT_INPUT = "soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv"
DEFAULT_OUTPUT_DIR = "bess_sizing_outputs"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SizingResult:
    scenario: str
    ramp_pct_per_min: float
    ramp_limit_mw_per_min: float
    p_bess_mw: float
    e_bess_mwh: float
    duration_min: float
    c_rate: float
    feasible: bool
    violations: int
    compliance_pct: float
    max_ramp_after_mw_per_min: float
    max_soc: float
    min_soc: float
    annual_discharge_mwh: float
    charged_energy_mwh: float
    capex_usd: float
    annualized_cost_usd_per_year: float
    lcos_usd_per_mwh: float
    search_stage: str
    runtime_s: float


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def capital_recovery_factor(r: float, n: int) -> float:
    if r == 0:
        return 1.0 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def annualized_cost(p_bess_mw: float, e_bess_mwh: float) -> tuple[float, float]:
    capex = (
        p_bess_mw * CAPEX_INV_USD_PER_MW
        + e_bess_mwh * CAPEX_BAT_USD_PER_MWH
    )
    crf = capital_recovery_factor(DISCOUNT_RATE, PROJECT_LIFE_YEARS)
    annual = capex * crf + FIXED_OM_FRAC_PER_YEAR * capex
    return capex, annual


def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0 / 60.0
    diffs = index.to_series().diff().dropna().dt.total_seconds().values
    return float(np.median(diffs)) / 3600.0


def load_pv_profile(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if "HighRes" in df.columns:
        s = df["HighRes"].copy()
    elif "Daya_Sintetik_MW" in df.columns:
        s = df["Daya_Sintetik_MW"].copy()
    else:
        raise ValueError(
            f"PV column not found. Available columns: {df.columns.tolist()}. "
            "Expected 'HighRes' or 'Daya_Sintetik_MW'."
        )

    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = s.astype(float).fillna(0.0)
    s[s < 0] = 0.0
    return s


def compute_pv_summary(pv: pd.Series) -> pd.DataFrame:
    dt_h = infer_timestep_hours(pv.index)
    annual_mwh = pv.sum() * dt_h
    cf = annual_mwh / (PV_CAPACITY_MW * 8760.0) * 100.0

    return pd.DataFrame([{
        "n_timestep": len(pv),
        "start_time": str(pv.index.min()),
        "end_time": str(pv.index.max()),
        "timestep_hours": dt_h,
        "max_mw": float(pv.max()),
        "mean_mw": float(pv.mean()),
        "annual_mwh": float(annual_mwh),
        "capacity_factor_pct": float(cf),
        "missing_values": int(pv.isna().sum()),
        "duplicate_index": int(pv.index.duplicated().sum()),
        "negative_values": int((pv < 0).sum()),
    }])


def ramp_statistics(pv_values: np.ndarray, ramp_limit: float) -> dict:
    ramps = np.abs(np.diff(pv_values))
    return {
        "total_ramp_events": len(ramps),
        "mean_abs_ramp_mw_per_min": float(np.mean(ramps)),
        "std_abs_ramp_mw_per_min": float(np.std(ramps)),
        "p90_mw_per_min": float(np.percentile(ramps, 90)),
        "p95_mw_per_min": float(np.percentile(ramps, 95)),
        "p99_mw_per_min": float(np.percentile(ramps, 99)),
        "p999_mw_per_min": float(np.percentile(ramps, 99.9)),
        "max_mw_per_min": float(np.max(ramps)),
        "ramp_limit_mw_per_min": float(ramp_limit),
        "violations": int(np.sum(ramps > ramp_limit)),
        "violations_pct": float(np.mean(ramps > ramp_limit) * 100.0),
        "max_excess_mw": float(np.max(np.maximum(ramps - ramp_limit, 0.0))),
    }


def make_grid(start: float, stop: float, step: float) -> np.ndarray:
    if stop < start:
        return np.array([], dtype=float)
    n = int(round((stop - start) / step))
    return np.round(start + np.arange(n + 1) * step, 6)


# =============================================================================
# BESS SIMULATION
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

    # No BESS case
    if p_inv_max <= 0.0 or e_bat_max <= 0.0:
        ramp_violations = 0
        max_ramp = 0.0
        prev = pv[0]
        for t in range(1, n):
            rr = pv[t] - prev
            if rr < 0.0:
                rr = -rr
            if rr > ramp_limit + ramp_tol:
                ramp_violations += 1
            if rr > max_ramp:
                max_ramp = rr
            prev = pv[t]
        return 0.0, 0.0, ramp_violations, max_ramp, soc_target, soc_target

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

        # Ramp-rate control
        if delta > ramp_limit:
            req = -(delta - ramp_limit)       # charge/absorb
        elif delta < -ramp_limit:
            req = -(delta + ramp_limit)       # discharge/inject
        else:
            # SOC recovery while preserving ramp limit
            soc_err = soc_prev - soc_target
            max_rec = 0.10 * p_inv_max
            abs_err = soc_err if soc_err >= 0 else -soc_err
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

        # Inverter limit
        p_bess = req
        if p_bess > p_inv_max:
            p_bess = p_inv_max
        elif p_bess < -p_inv_max:
            p_bess = -p_inv_max

        # SOC update
        if p_bess >= 0.0:
            e_change = (p_bess / eta_ow) * dt
        else:
            e_change = (p_bess * eta_ow) * dt

        soc_new = soc_prev - (e_change / e_bat_max)

        # SOC lower clamp
        if soc_new < soc_min:
            available_mwh = (soc_prev - soc_min) * e_bat_max
            if available_mwh < 0.0:
                available_mwh = 0.0
            p_bess = available_mwh * eta_ow / dt
            soc_new = soc_min

        # SOC upper clamp
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

    if p_inv_max <= 0.0 or e_bat_max <= 0.0:
        p_grid_arr[:] = pv[:]
        soc_arr[:] = soc_target
        for t in range(1, n):
            rr = p_grid_arr[t] - p_grid_arr[t-1]
            ramp_arr[t] = -rr if rr < 0.0 else rr
        return p_grid_arr, p_bess_arr, soc_arr, ramp_arr

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
# OPTIMIZATION
# =============================================================================

def candidate_is_valid(p: float, e: float) -> bool:
    if p < 0.0 or e < 0.0:
        return False
    if p == 0.0 and e == 0.0:
        return True
    if p <= 0.0 or e <= 0.0:
        return False
    if (p / e) > C_RATE_MAX:
        return False
    if e < p * DURATION_MIN / 60.0:
        return False
    return True


def evaluate_candidate(
    pv: np.ndarray,
    p: float,
    e: float,
    ramp_limit: float,
    scenario: str,
    stage: str,
    runtime_s: float = 0.0,
) -> SizingResult:
    if not candidate_is_valid(p, e):
        return SizingResult(
            scenario=scenario,
            ramp_pct_per_min=ramp_limit / PV_CAPACITY_MW,
            ramp_limit_mw_per_min=ramp_limit,
            p_bess_mw=p,
            e_bess_mwh=e,
            duration_min=math.inf,
            c_rate=math.inf,
            feasible=False,
            violations=10**9,
            compliance_pct=0.0,
            max_ramp_after_mw_per_min=math.inf,
            max_soc=math.nan,
            min_soc=math.nan,
            annual_discharge_mwh=0.0,
            charged_energy_mwh=0.0,
            capex_usd=math.inf,
            annualized_cost_usd_per_year=math.inf,
            lcos_usd_per_mwh=math.inf,
            search_stage=stage,
            runtime_s=runtime_s,
        )

    d_mwh, c_mwh, viol, max_ramp, min_soc, max_soc = simulate_bess_core(
        pv, p, e, ramp_limit, SOC_MIN, SOC_MAX, SOC_TARGET, ETA_ONE_WAY, RAMP_TOL
    )

    feasible = viol <= MAX_RAMP_VIOLATIONS
    compliance = (1.0 - viol / (len(pv) - 1)) * 100.0

    capex, annual = annualized_cost(p, e)
    lcos = annual / d_mwh if d_mwh > 0.0 else math.inf

    duration = (e / p) * 60.0 if p > 0.0 else math.inf
    c_rate = p / e if e > 0.0 else math.inf

    return SizingResult(
        scenario=scenario,
        ramp_pct_per_min=ramp_limit / PV_CAPACITY_MW,
        ramp_limit_mw_per_min=ramp_limit,
        p_bess_mw=p,
        e_bess_mwh=e,
        duration_min=duration,
        c_rate=c_rate,
        feasible=feasible,
        violations=int(viol),
        compliance_pct=compliance,
        max_ramp_after_mw_per_min=max_ramp,
        max_soc=max_soc,
        min_soc=min_soc,
        annual_discharge_mwh=d_mwh,
        charged_energy_mwh=c_mwh,
        capex_usd=capex,
        annualized_cost_usd_per_year=annual,
        lcos_usd_per_mwh=lcos,
        search_stage=stage,
        runtime_s=runtime_s,
    )


def search_grid(
    pv: np.ndarray,
    scenario: str,
    ramp_limit: float,
    p_values: np.ndarray,
    e_values: np.ndarray,
    stage: str,
) -> tuple[SizingResult | None, list[dict]]:
    t0 = time.time()
    best: SizingResult | None = None
    feasible_rows: list[dict] = []

    total = len(p_values) * len(e_values)
    checked = 0

    for p in p_values:
        e_min_required = max(p / C_RATE_MAX if p > 0.0 else 0.0, p * DURATION_MIN / 60.0)
        for e in e_values:
            checked += 1
            if e < e_min_required:
                continue

            res = evaluate_candidate(
                pv=pv,
                p=float(p),
                e=float(e),
                ramp_limit=ramp_limit,
                scenario=scenario,
                stage=stage,
            )

            if not res.feasible:
                continue

            feasible_rows.append(asdict(res))

            if best is None or res.annualized_cost_usd_per_year < best.annualized_cost_usd_per_year:
                best = res

    runtime = time.time() - t0
    if best is not None:
        best.runtime_s = runtime

    print(
        f"    {stage}: checked {checked:,}/{total:,} candidates | "
        f"feasible {len(feasible_rows):,} | runtime {runtime:.1f}s"
    )
    return best, feasible_rows


def optimize_scenario(pv: np.ndarray, scenario: str, ramp_pct: float) -> tuple[SizingResult, list[dict]]:
    ramp_limit = ramp_pct * PV_CAPACITY_MW

    print(f"\n{'='*72}")
    print(f"SCENARIO {scenario}: ramp limit = {ramp_pct*100:.1f}%/min = {ramp_limit:.2f} MW/min")
    print(f"{'='*72}")

    # No-BESS check is disabled because the Indonesian grid-code assumption
    # requires minimum BESS power capacity of 10% of installed PV capacity.
    # no_bess = evaluate_candidate(pv, 0.0, 0.0, ramp_limit, scenario, "no_bess")
    # if no_bess.feasible:
    #     print("    No BESS is feasible for this ramp limit.")
    #     return no_bess, [asdict(no_bess)]

    p_grid = make_grid(P_MIN_MW, P_MAX_MW, P_STEP_COARSE)
    e_grid = make_grid(E_MIN_MWH, E_MAX_MWH, E_STEP_COARSE)

    best_coarse, rows_coarse = search_grid(pv, scenario, ramp_limit, p_grid, e_grid, "coarse")

    if best_coarse is None:
        raise RuntimeError(
            f"No feasible solution found for {scenario}. Increase P_MAX_MW/E_MAX_MWH."
        )

    print(
        f"    Coarse best: P={best_coarse.p_bess_mw:.2f} MW | "
        f"E={best_coarse.e_bess_mwh:.2f} MWh | "
        f"Annual cost={best_coarse.annualized_cost_usd_per_year:,.0f} USD/y"
    )

    p1 = max(P_MIN_MW, best_coarse.p_bess_mw - P_FINE_WINDOW_MW)
    p2 = min(P_MAX_MW, best_coarse.p_bess_mw + P_FINE_WINDOW_MW)
    e1 = max(E_MIN_MWH, best_coarse.e_bess_mwh - E_FINE_WINDOW_MWH)
    e2 = min(E_MAX_MWH, best_coarse.e_bess_mwh + E_FINE_WINDOW_MWH)

    p_fine = make_grid(p1, p2, P_STEP_FINE)
    e_fine = make_grid(e1, e2, E_STEP_FINE)

    best_fine, rows_fine = search_grid(pv, scenario, ramp_limit, p_fine, e_fine, "fine")

    if best_fine is None:
        best = best_coarse
        rows = rows_coarse
    else:
        best = best_fine
        rows = rows_coarse + rows_fine

    print(
        f"    FINAL best: P={best.p_bess_mw:.2f} MW | "
        f"E={best.e_bess_mwh:.2f} MWh | "
        f"Duration={best.duration_min:.1f} min | "
        f"Compliance={best.compliance_pct:.5f}% | "
        f"LCOS={best.lcos_usd_per_mwh:.2f} USD/MWh"
    )

    return best, rows


# =============================================================================
# EXPORT AND PLOTS
# =============================================================================

def save_best_timeseries(pv_series: pd.Series, result: SizingResult, out_dir: Path) -> None:
    pv = pv_series.values.astype(np.float64)

    if result.p_bess_mw <= 0.0 or result.e_bess_mwh <= 0.0:
        p_grid = pv.copy()
        p_bess = np.zeros_like(pv)
        soc = np.full_like(pv, SOC_TARGET)
        ramp = np.zeros_like(pv)
        ramp[1:] = np.abs(np.diff(p_grid))
    else:
        p_grid, p_bess, soc, ramp = simulate_bess_trace(
            pv, result.p_bess_mw, result.e_bess_mwh,
            result.ramp_limit_mw_per_min,
            SOC_MIN, SOC_MAX, SOC_TARGET, ETA_ONE_WAY, RAMP_TOL
        )

    df_ts = pd.DataFrame(
        {
            "PV_MW": pv,
            "P_BESS_MW": p_bess,
            "P_Grid_MW": p_grid,
            "SOC": soc,
            "Ramp_Grid_MW_per_min": ramp,
        },
        index=pv_series.index,
    )

    df_ts.to_csv(out_dir / f"05_best_timeseries_{result.scenario}.csv")


def make_figures(pv: pd.Series, ramp_stats_df: pd.DataFrame, sizing_df: pd.DataFrame, out_dir: Path) -> None:
    # PV sample week
    monthly = pv.resample("ME").sum()
    best_month = monthly.idxmax()
    start = pd.Timestamp(best_month.year, best_month.month, 15)
    end = start + pd.Timedelta(days=7)
    sample = pv.loc[start:end]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sample.index, sample.values, linewidth=0.8)
    ax.set_title("Sample SoDa Synthetic PV Profile")
    ax.set_ylabel("PV Power (MW)")
    ax.set_xlabel("Time")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_01_pv_profile_sample.png", dpi=180)
    plt.close()

    # Ramp histogram
    ramps = np.abs(np.diff(pv.values.astype(float)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ramps, bins=120, density=False, alpha=0.8)
    for _, row in ramp_stats_df.iterrows():
        ax.axvline(
            row["ramp_limit_mw_per_min"], linestyle="--", linewidth=1.2,
            label=f"{row['scenario']} ({row['ramp_limit_mw_per_min']:.0f} MW/min)"
        )
    ax.set_title("Distribution of Absolute PV Ramp Events")
    ax.set_xlabel("Absolute ramp (MW/min)")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_02_ramp_distribution.png", dpi=180)
    plt.close()

    # Sizing vs ramp limit
    plot_df = sizing_df.sort_values("ramp_limit_mw_per_min", ascending=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(plot_df["ramp_limit_mw_per_min"], plot_df["p_bess_mw"], marker="o", label="P_BESS (MW)")
    ax1.plot(plot_df["ramp_limit_mw_per_min"], plot_df["e_bess_mwh"], marker="s", label="E_BESS (MWh)")
    ax1.set_xlabel("Ramp limit (MW/min)")
    ax1.set_ylabel("BESS size")
    ax1.set_title("Optimum BESS Size vs Ramp-Rate Limit")
    ax1.invert_xaxis()
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        plot_df["ramp_limit_mw_per_min"],
        plot_df["annualized_cost_usd_per_year"] / 1e6,
        marker="^", linestyle=":", label="Annualized cost"
    )
    ax2.set_ylabel("Annualized cost (million USD/year)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_dir / "fig_03_sizing_vs_ramp_limit.png", dpi=180)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="BESS sizing for SoDa PV profile using grid search.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input SoDa final CSV.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading PV profile: {input_path}")
    pv_series = load_pv_profile(input_path)
    pv_values = pv_series.values.astype(np.float64)

    print(f"Loaded {len(pv_series):,} timesteps.")
    print(f"Time range: {pv_series.index.min()} → {pv_series.index.max()}")
    print(f"Max PV: {pv_series.max():.3f} MW")
    print(f"Mean PV: {pv_series.mean():.3f} MW")

    pv_summary = compute_pv_summary(pv_series)
    monthly_energy = (pv_series.resample("ME").sum() * infer_timestep_hours(pv_series.index)).rename("MWh")
    pv_summary.to_csv(out_dir / "01_pv_summary.csv", index=False)
    monthly_energy.to_csv(out_dir / "01b_monthly_energy_mwh.csv")

    print("\nPV summary:")
    print(pv_summary.T)

    ramp_rows = []
    for scen, pct in RAMP_SCENARIOS.items():
        limit = pct * PV_CAPACITY_MW
        row = ramp_statistics(pv_values, limit)
        row["scenario"] = scen
        row["ramp_pct_per_min"] = pct
        ramp_rows.append(row)

    ramp_stats_df = pd.DataFrame(ramp_rows)
    ramp_stats_df.to_csv(out_dir / "02_ramp_statistics_before_bess.csv", index=False)

    print("\nRamp statistics before BESS:")
    print(ramp_stats_df[
        ["scenario", "ramp_limit_mw_per_min", "violations", "violations_pct",
         "p95_mw_per_min", "p99_mw_per_min", "p999_mw_per_min", "max_mw_per_min"]
    ].to_string(index=False))

    print("\nWarming up simulation...")
    t_warm = time.time()
    _ = simulate_bess_core(
        pv_values[:2000], 5.0, 5.0, 5.0,
        SOC_MIN, SOC_MAX, SOC_TARGET, ETA_ONE_WAY, RAMP_TOL
    )
    print(f"Warm-up finished in {time.time() - t_warm:.2f}s | numba={NUMBA_AVAILABLE}")

    all_best: list[SizingResult] = []
    all_feasible_rows: list[dict] = []

    t_all = time.time()
    for scen, pct in RAMP_SCENARIOS.items():
        best, rows = optimize_scenario(pv_values, scen, pct)
        all_best.append(best)
        all_feasible_rows.extend(rows)
        save_best_timeseries(pv_series, best, out_dir)

    total_runtime = time.time() - t_all

    sizing_df = pd.DataFrame([asdict(r) for r in all_best])
    sizing_df.to_csv(out_dir / "03_bess_sizing_ramp_scenarios.csv", index=False)

    feasible_df = pd.DataFrame(all_feasible_rows)
    if not feasible_df.empty:
        feasible_df = feasible_df.sort_values(
            ["scenario", "annualized_cost_usd_per_year", "p_bess_mw", "e_bess_mwh"]
        )
        top_df = feasible_df.groupby("scenario", group_keys=False).head(100)
        top_df.to_csv(out_dir / "04_top_feasible_candidates.csv", index=False)

    make_figures(pv_series, ramp_stats_df, sizing_df, out_dir)

    print(f"\n{'='*72}")
    print("FINAL BESS SIZING SUMMARY")
    print(f"{'='*72}")
    print(sizing_df[
        ["scenario", "ramp_limit_mw_per_min", "p_bess_mw", "e_bess_mwh",
         "duration_min", "c_rate", "violations", "compliance_pct",
         "annualized_cost_usd_per_year", "lcos_usd_per_mwh"]
    ].to_string(index=False))

    print(f"\nTotal optimization runtime: {total_runtime/60:.2f} minutes")
    print(f"Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
