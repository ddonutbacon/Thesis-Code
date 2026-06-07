# -*- coding: utf-8 -*-
"""
03_prepare_digsilent_time_characteristics.py

Prepare DIgSILENT time characteristics from Python BESS simulation results.

This script selects ONE representative R5 day for quasi-dynamic verification:
- default: day with the highest number of raw PV ramp violations against R5.
- tie-breaker: highest daily maximum raw ramp.

It exports TWO main DIgSILENT input files:
1. PV plant time characteristic
2. BESS time characteristic

It also exports a combined file and a metadata summary.

Use:
python 03_prepare_digsilent_time_characteristics.py

Input expected:
bess_sizing_outputs/05_best_timeseries_R5.csv

Sign convention from Python:
- P_BESS_MW > 0 : BESS discharges / injects active power to grid
- P_BESS_MW < 0 : BESS charges / absorbs active power

For DIgSILENT:
- If your BESS element supports negative P, use P_BESS_MW directly.
- If it needs separate charge/discharge signals, use BESS_Discharge_MW and BESS_Charge_MW.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PV_CAPACITY_MW = 100.0
R5_LIMIT_MW_PER_MIN = 5.0

DEFAULT_R5_TIMESERIES = "bess_sizing_outputs/05_best_timeseries_R5.csv"
DEFAULT_OUTPUT_DIR = "digsilent_time_characteristics"


def load_r5_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    required = ["PV_MW", "P_BESS_MW", "P_Grid_MW", "SOC", "Ramp_Grid_MW_per_min"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in R5 time series: {missing}")
    return df.sort_index()


def select_ramp_stress_day(df: pd.DataFrame) -> pd.Timestamp:
    raw_ramp = pd.Series(np.r_[0.0, np.abs(np.diff(df["PV_MW"].values))], index=df.index)
    daily_viol = (raw_ramp > R5_LIMIT_MW_PER_MIN).resample("D").sum()
    daily_max = raw_ramp.resample("D").max()

    selector = pd.DataFrame({
        "raw_r5_violation_count": daily_viol,
        "raw_max_ramp_mw_per_min": daily_max,
    })

    selector = selector.sort_values(
        ["raw_r5_violation_count", "raw_max_ramp_mw_per_min"],
        ascending=[False, False]
    )

    return selector.index[0], selector


def select_highest_energy_day(df: pd.DataFrame) -> pd.Timestamp:
    daily_energy = df["PV_MW"].resample("D").sum() / 60.0
    return daily_energy.idxmax()


def prepare_day_export(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    day_df = df.loc[day.strftime("%Y-%m-%d")].copy()

    if len(day_df) != 1440:
        raise RuntimeError(f"Selected day {day.date()} has {len(day_df)} rows, expected 1440.")

    # Time axis for DIgSILENT time characteristic
    day_df.insert(0, "Time_min", np.arange(len(day_df), dtype=float))
    day_df.insert(1, "Time_h", day_df["Time_min"] / 60.0)

    # Per-unit options
    day_df["PV_pu_on_100MW"] = day_df["PV_MW"] / PV_CAPACITY_MW
    day_df["P_Grid_pu_on_100MW"] = day_df["P_Grid_MW"] / PV_CAPACITY_MW

    # BESS sign convention helpers
    day_df["BESS_Injection_MW"] = day_df["P_BESS_MW"]
    day_df["BESS_Discharge_MW"] = day_df["P_BESS_MW"].clip(lower=0.0)
    day_df["BESS_Charge_MW"] = (-day_df["P_BESS_MW"]).clip(lower=0.0)

    return day_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DIgSILENT time characteristics from R5 time series.")
    parser.add_argument("--input", type=str, default=DEFAULT_R5_TIMESERIES)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--day-mode",
        type=str,
        default="ramp_stress",
        choices=["ramp_stress", "highest_energy"],
        help="Select representative day for DIgSILENT.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_r5_timeseries(input_path)

    ramp_day, selector = select_ramp_stress_day(df)
    energy_day = select_highest_energy_day(df)

    selected_day = ramp_day if args.day_mode == "ramp_stress" else energy_day

    day_df = prepare_day_export(df, selected_day)

    day_label = selected_day.strftime("%Y-%m-%d")
    mode_label = args.day_mode

    # Main DIgSILENT files: two inputs
    pv_out = day_df[[
        "Time_h",
        "Time_min",
        "PV_MW",
        "PV_pu_on_100MW",
    ]].copy()

    bess_out = day_df[[
        "Time_h",
        "Time_min",
        "P_BESS_MW",
        "BESS_Injection_MW",
        "BESS_Discharge_MW",
        "BESS_Charge_MW",
        "SOC",
    ]].copy()

    combined_out = day_df[[
        "Time_h",
        "Time_min",
        "PV_MW",
        "P_BESS_MW",
        "P_Grid_MW",
        "SOC",
        "Ramp_Grid_MW_per_min",
        "PV_pu_on_100MW",
        "P_Grid_pu_on_100MW",
        "BESS_Discharge_MW",
        "BESS_Charge_MW",
    ]].copy()

    pv_file = out_dir / f"07_DIgSILENT_PV_time_characteristic_R5_{mode_label}_{day_label}.csv"
    bess_file = out_dir / f"08_DIgSILENT_BESS_time_characteristic_R5_{mode_label}_{day_label}.csv"
    combined_file = out_dir / f"09_DIgSILENT_combined_R5_{mode_label}_{day_label}.csv"

    pv_out.to_csv(pv_file, index=False)
    bess_out.to_csv(bess_file, index=False)
    combined_out.to_csv(combined_file, index=False)

    # Daily selector diagnostics
    selector.to_csv(out_dir / "daily_ramp_stress_selector_R5.csv")

    # Metadata
    raw_ramp = pd.Series(np.r_[0.0, np.abs(np.diff(df["PV_MW"].values))], index=df.index)
    selected_raw = raw_ramp.loc[day_label]

    meta = {
        "selected_day": day_label,
        "selection_mode": mode_label,
        "recommended_day_for_thesis": "ramp_stress",
        "ramp_stress_day": ramp_day.strftime("%Y-%m-%d"),
        "highest_energy_day": energy_day.strftime("%Y-%m-%d"),
        "selected_day_raw_max_ramp_mw_per_min": float(selected_raw.max()),
        "selected_day_raw_r5_violation_count": int((selected_raw > R5_LIMIT_MW_PER_MIN).sum()),
        "selected_day_pv_energy_mwh": float(day_df["PV_MW"].sum() / 60.0),
        "selected_day_bess_discharge_mwh": float(day_df["BESS_Discharge_MW"].sum() / 60.0),
        "selected_day_bess_charge_mwh": float(day_df["BESS_Charge_MW"].sum() / 60.0),
        "selected_day_soc_min": float(day_df["SOC"].min()),
        "selected_day_soc_max": float(day_df["SOC"].max()),
        "python_bess_sign_convention": "P_BESS_MW > 0 means discharge/injection; P_BESS_MW < 0 means charge/absorption.",
    }

    with open(out_dir / "digsilent_selected_day_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Plot
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(day_df["Time_h"], day_df["PV_MW"], label="PV before BESS", linewidth=1.2)
    ax1.plot(day_df["Time_h"], day_df["P_Grid_MW"], label="Grid after BESS", linewidth=1.2)
    ax1.set_xlabel("Time (hour)")
    ax1.set_ylabel("Power (MW)")
    ax1.set_title(f"DIgSILENT representative day R5: {day_label} ({mode_label})")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(day_df["Time_h"], day_df["P_BESS_MW"], label="BESS P", linewidth=0.9, alpha=0.8)
    ax2.set_ylabel("BESS Power (MW)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_dir / f"fig_08_digsilent_selected_day_R5_{mode_label}_{day_label}.png", dpi=180)
    plt.close()

    print("\nSelected representative day for DIgSILENT:")
    print(json.dumps(meta, indent=2))

    print("\nMain files for DIgSILENT:")
    print(f"  PV   : {pv_file}")
    print(f"  BESS : {bess_file}")
    print(f"  Combined/debug: {combined_file}")


if __name__ == "__main__":
    main()
