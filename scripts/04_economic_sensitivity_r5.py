# -*- coding: utf-8 -*-
"""
02_bess_economic_sensitivity_R5.py

Economic sensitivity for the R5 baseline technical case.

This script DOES NOT re-optimize the BESS size.
It evaluates how CAPEX battery and discount rate affect:
- total CAPEX,
- annualized cost,
- LCOS indicator,

using the R5 optimum capacity from:
bess_sizing_outputs/03_bess_sizing_ramp_scenarios.csv

Use:
python 02_bess_economic_sensitivity_R5.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Baseline economic assumptions
CAPEX_INV_USD_PER_MW = 427_420.0
CAPEX_BAT_BASE_USD_PER_MWH = 307_990.0
FIXED_OM_FRAC_PER_YEAR = 0.0207
DISCOUNT_RATE_BASE = 0.08
PROJECT_LIFE_YEARS = 20

DEFAULT_SIZING_SUMMARY = "bess_sizing_outputs/03_bess_sizing_ramp_scenarios.csv"
DEFAULT_OUTPUT_DIR = "economic_sensitivity_outputs"


def capital_recovery_factor(r: float, n: int) -> float:
    if r == 0:
        return 1.0 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def calc_costs(
    p_mw: float,
    e_mwh: float,
    annual_discharge_mwh: float,
    capex_bat: float,
    discount_rate: float,
) -> dict:
    capex = p_mw * CAPEX_INV_USD_PER_MW + e_mwh * capex_bat
    crf = capital_recovery_factor(discount_rate, PROJECT_LIFE_YEARS)
    annualized_cost = capex * crf + FIXED_OM_FRAC_PER_YEAR * capex
    lcos = annualized_cost / annual_discharge_mwh if annual_discharge_mwh > 0 else np.inf

    return {
        "capex_bat_usd_per_mwh": capex_bat,
        "discount_rate": discount_rate,
        "capex_total_usd": capex,
        "annualized_cost_usd_per_year": annualized_cost,
        "lcos_usd_per_mwh": lcos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Economic sensitivity for R5 BESS sizing.")
    parser.add_argument("--summary", type=str, default=DEFAULT_SIZING_SUMMARY)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    r5 = df[df["scenario"] == "R5"]

    if r5.empty:
        raise RuntimeError("Scenario R5 not found in sizing summary.")

    row = r5.iloc[0]
    p_mw = float(row["p_bess_mw"])
    e_mwh = float(row["e_bess_mwh"])
    annual_discharge_mwh = float(row["annual_discharge_mwh"])

    print("R5 baseline size:")
    print(f"  P_BESS = {p_mw:.3f} MW")
    print(f"  E_BESS = {e_mwh:.3f} MWh")
    print(f"  Annual discharge = {annual_discharge_mwh:.3f} MWh/year")

    scenarios = []

    # One-at-a-time CAPEX battery sensitivity
    for label, mult in [
        ("CAPEX_low_minus20pct", 0.80),
        ("CAPEX_baseline", 1.00),
        ("CAPEX_high_plus20pct", 1.20),
    ]:
        scenarios.append({
            "sensitivity_type": "CAPEX_battery",
            "case": label,
            **calc_costs(
                p_mw=p_mw,
                e_mwh=e_mwh,
                annual_discharge_mwh=annual_discharge_mwh,
                capex_bat=CAPEX_BAT_BASE_USD_PER_MWH * mult,
                discount_rate=DISCOUNT_RATE_BASE,
            )
        })

    # One-at-a-time discount-rate sensitivity
    for label, dr in [
        ("DR_low_6pct", 0.06),
        ("DR_baseline_8pct", 0.08),
        ("DR_high_10pct", 0.10),
    ]:
        scenarios.append({
            "sensitivity_type": "discount_rate",
            "case": label,
            **calc_costs(
                p_mw=p_mw,
                e_mwh=e_mwh,
                annual_discharge_mwh=annual_discharge_mwh,
                capex_bat=CAPEX_BAT_BASE_USD_PER_MWH,
                discount_rate=dr,
            )
        })

    # Optional 3x3 matrix for appendix
    for capex_mult in [0.80, 1.00, 1.20]:
        for dr in [0.06, 0.08, 0.10]:
            scenarios.append({
                "sensitivity_type": "matrix_CAPEX_DR",
                "case": f"BATCAPEX_{capex_mult:.2f}_DR_{dr:.2f}",
                **calc_costs(
                    p_mw=p_mw,
                    e_mwh=e_mwh,
                    annual_discharge_mwh=annual_discharge_mwh,
                    capex_bat=CAPEX_BAT_BASE_USD_PER_MWH * capex_mult,
                    discount_rate=dr,
                )
            })

    out = pd.DataFrame(scenarios)
    out.insert(0, "scenario", "R5")
    out.insert(1, "p_bess_mw", p_mw)
    out.insert(2, "e_bess_mwh", e_mwh)
    out.insert(3, "annual_discharge_mwh", annual_discharge_mwh)

    out.to_csv(out_dir / "06_economic_sensitivity_R5.csv", index=False)

    # Plot CAPEX sensitivity only
    capex_df = out[out["sensitivity_type"] == "CAPEX_battery"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(capex_df["case"], capex_df["annualized_cost_usd_per_year"] / 1e6)
    ax.set_ylabel("Annualized cost (million USD/year)")
    ax.set_title("R5 Economic Sensitivity: Battery CAPEX")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_06_sensitivity_battery_capex.png", dpi=180)
    plt.close()

    # Plot discount-rate sensitivity only
    dr_df = out[out["sensitivity_type"] == "discount_rate"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(dr_df["case"], dr_df["annualized_cost_usd_per_year"] / 1e6)
    ax.set_ylabel("Annualized cost (million USD/year)")
    ax.set_title("R5 Economic Sensitivity: Discount Rate")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_07_sensitivity_discount_rate.png", dpi=180)
    plt.close()

    print("\nSaved:")
    print(f"  {out_dir / '06_economic_sensitivity_R5.csv'}")
    print(f"  {out_dir / 'fig_06_sensitivity_battery_capex.png'}")
    print(f"  {out_dir / 'fig_07_sensitivity_discount_rate.png'}")


if __name__ == "__main__":
    main()
