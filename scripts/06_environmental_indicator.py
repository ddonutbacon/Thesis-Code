# -*- coding: utf-8 -*-
"""
06_environmental_indicator.py

Indicative environmental calculation for the thesis BESS ramp-rate workflow.

Purpose
-------
Calculate indicative CO2-equivalent associated with annual BESS discharge energy
for each ramp-rate scenario.

Important interpretation
------------------------
This script DOES NOT calculate verified emission reduction or carbon credit.
It only multiplies annual BESS discharge energy by a selected grid emission
factor as an indicative environmental indicator.

Default equation
----------------
CO2eq_ton_per_year = annual_discharge_mwh * emission_factor_tco2_per_mwh

Default emission factor
-----------------------
0.87 ton CO2/MWh, based on the JAMALI grid emission factor used in the thesis.

Use
---
python 06_environmental_indicator.py

Optional:
python 06_environmental_indicator.py \
    --input bess_sizing_outputs/03_bess_sizing_ramp_scenarios.csv \
    --output environmental_outputs \
    --ef 0.87

Main output
-----------
- environmental_outputs/10_environmental_indicator.csv
- environmental_outputs/10_environmental_indicator.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "bess_sizing_outputs/03_bess_sizing_ramp_scenarios.csv"
DEFAULT_OUTPUT_DIR = "environmental_outputs"
DEFAULT_EF_TCO2_PER_MWH = 0.87
DEFAULT_EF_SOURCE = "Faktor emisi GRK sistem ketenagalistrikan JAMALI tahun 2019, Ditjen Ketenagalistrikan Kementerian ESDM"


def read_sizing_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sizing summary not found: {path}")

    df = pd.read_csv(path)
    required = ["scenario", "annual_discharge_mwh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing required columns in {path}: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def build_environmental_table(
    sizing: pd.DataFrame,
    emission_factor_tco2_per_mwh: float,
    emission_factor_source: str,
) -> pd.DataFrame:
    out = sizing.copy()

    # Keep useful columns if available; do not fail if some are absent.
    preferred_cols = [
        "scenario",
        "ramp_limit_mw_per_min",
        "p_bess_mw",
        "e_bess_mwh",
        "annual_discharge_mwh",
        "charged_energy_mwh",
        "annualized_cost_usd_per_year",
        "compliance_pct",
        "violations",
    ]
    cols = [c for c in preferred_cols if c in out.columns]
    out = out[cols].copy()

    out["emission_factor_tco2_per_mwh"] = float(emission_factor_tco2_per_mwh)
    out["co2eq_ton_per_year"] = (
        out["annual_discharge_mwh"].astype(float) * float(emission_factor_tco2_per_mwh)
    )
    out["co2eq_kg_per_year"] = out["co2eq_ton_per_year"] * 1000.0
    out["interpretation"] = (
        "Indicative CO2eq associated with annual BESS discharge energy; "
        "not verified actual emission reduction, not carbon credit, and not LCA."
    )
    out["emission_factor_source"] = emission_factor_source

    # Put scenario first and sort in the usual thesis order.
    order = {"R20": 0, "R10": 1, "R5": 2, "R3": 3}
    if "scenario" in out.columns:
        out["_order"] = out["scenario"].map(order).fillna(99)
        out = out.sort_values(["_order", "scenario"]).drop(columns=["_order"])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create indicative environmental indicator from BESS annual discharge.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="BESS sizing summary CSV.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--ef", type=float, default=DEFAULT_EF_TCO2_PER_MWH, help="Emission factor in ton CO2/MWh.")
    parser.add_argument("--ef-source", type=str, default=DEFAULT_EF_SOURCE, help="Emission factor source note.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizing = read_sizing_summary(input_path)
    env = build_environmental_table(sizing, args.ef, args.ef_source)

    csv_path = out_dir / "10_environmental_indicator.csv"
    xlsx_path = out_dir / "10_environmental_indicator.xlsx"

    env.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        env.to_excel(writer, sheet_name="environmental_indicator", index=False)

    print("Saved environmental indicator outputs:")
    print(f"  CSV : {csv_path}")
    print(f"  XLSX: {xlsx_path}")
    print("\nPreview:")
    print(env.to_string(index=False))


if __name__ == "__main__":
    main()
