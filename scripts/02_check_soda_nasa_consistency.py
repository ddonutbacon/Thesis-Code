# -*- coding: utf-8 -*-
"""
04_level1_nasa_power_consistency_FIXED.py

Level-1 consistency check:
SoDa final monthly PV energy vs NASA POWER daily solar resource.

Fixes:
1. Robust monthly merge when index name becomes DATE.
2. Calendar alignment: if SoDa excludes Feb 29, NASA Feb 29 is removed too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PV_CAPACITY_MW = 100.0
EXPECTED_YEAR = 2020

DEFAULT_SODA = "soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv"
DEFAULT_NASA = "POWER_Point_Daily_20200101_20201231_006d58S_107d90E_LST.csv"
DEFAULT_OUTPUT_DIR = "consistency_level1_outputs"

NASA_REQUIRED_COLUMNS = [
    "YEAR", "MO", "DY",
    "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M", "WS10M"
]


def find_nasa_data_start(csv_path: Path) -> int:
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.strip() == "-END HEADER-":
                return i + 1
    return 0


def parse_nasa_header(csv_path: Path) -> dict:
    meta = {"raw_header_found": False, "date_range": None, "location": None, "time_standard": None, "parameters": []}
    lines = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
            if line.strip() == "-END HEADER-":
                break

    if not lines or lines[0].strip() != "-BEGIN HEADER-":
        return meta

    meta["raw_header_found"] = True
    for line in lines:
        s = line.strip()
        if s.startswith("Dates"):
            meta["date_range"] = s
            if " in " in s:
                meta["time_standard"] = s.split(" in ")[-1].strip()
        elif s.startswith("Location"):
            meta["location"] = s
        elif s.startswith(("ALLSKY_", "CLRSKY_", "T2M", "WS10M")):
            meta["parameters"].append(s)
    return meta


def load_nasa_power_daily(csv_path: Path, expected_year: int, allow_year_mismatch: bool):
    skiprows = find_nasa_data_start(csv_path)
    meta = parse_nasa_header(csv_path)

    df = pd.read_csv(csv_path, skiprows=skiprows)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in NASA_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"NASA POWER file missing columns: {missing}\nAvailable: {df.columns.tolist()}")

    for c in NASA_REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace(-999, np.nan)
    df["DATE"] = pd.to_datetime(dict(year=df["YEAR"].astype(int), month=df["MO"].astype(int), day=df["DY"].astype(int)))
    df = df.set_index("DATE").sort_index()

    years = sorted(df.index.year.unique().tolist())
    if years != [expected_year] and not allow_year_mismatch:
        raise RuntimeError(f"NASA POWER year mismatch. Expected {expected_year}, got {years}.")

    data_cols = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M", "WS10M"]
    if df[data_cols].isna().any().any():
        raise RuntimeError(f"NASA POWER contains missing values: {df[data_cols].isna().sum().to_dict()}")

    return df, meta


def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0 / 60.0
    diffs = index.to_series().diff().dropna().dt.total_seconds().values
    return float(np.median(diffs)) / 3600.0


def load_soda_profile(csv_path: Path, expected_year: int, allow_year_mismatch: bool) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if "HighRes" in df.columns:
        s = df["HighRes"].copy()
    elif "Daya_Sintetik_MW" in df.columns:
        s = df["Daya_Sintetik_MW"].copy()
    else:
        raise RuntimeError(f"SoDa PV column not found. Available columns: {df.columns.tolist()}")

    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s[s < 0] = 0.0

    years = sorted(s.index.year.unique().tolist())
    if years != [expected_year] and not allow_year_mismatch:
        raise RuntimeError(f"SoDa year mismatch. Expected {expected_year}, got {years}.")

    return s


def align_nasa_to_soda_calendar(nasa_daily: pd.DataFrame, soda: pd.Series):
    soda_days = pd.DatetimeIndex(soda.index.normalize().unique())
    keep = nasa_daily.index.normalize().isin(soda_days)

    removed = sorted(list(set(nasa_daily.loc[~keep].index.strftime("%Y-%m-%d").tolist())))
    aligned = nasa_daily.loc[keep].copy()

    info = {
        "soda_unique_days": int(len(soda_days)),
        "nasa_unique_days_before_alignment": int(len(nasa_daily.index.normalize().unique())),
        "nasa_unique_days_after_alignment": int(len(aligned.index.normalize().unique())),
        "removed_nasa_dates": removed,
    }

    return aligned, info


def monthly_soda_energy(soda: pd.Series) -> pd.DataFrame:
    dt_h = infer_timestep_hours(soda.index)
    out = (soda.resample("ME").sum() * dt_h).to_frame("SoDa_MWh")
    out["Month"] = out.index.month
    out["Month_Name"] = out.index.strftime("%b")
    out["SoDa_Share"] = out["SoDa_MWh"] / out["SoDa_MWh"].sum()
    return out


def monthly_nasa_resource(nasa: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=nasa.resample("ME").sum().index)
    out["NASA_ALLSKY_GHI_kWh_m2_month"] = nasa["ALLSKY_SFC_SW_DWN"].resample("ME").sum()
    out["NASA_CLRSKY_GHI_kWh_m2_month"] = nasa["CLRSKY_SFC_SW_DWN"].resample("ME").sum()
    out["NASA_T2M_mean_C"] = nasa["T2M"].resample("ME").mean()
    out["NASA_WS10M_mean_m_s"] = nasa["WS10M"].resample("ME").mean()
    out["NASA_Clearness_Ratio"] = out["NASA_ALLSKY_GHI_kWh_m2_month"] / out["NASA_CLRSKY_GHI_kWh_m2_month"]
    out["Month"] = out.index.month
    out["Month_Name"] = out.index.strftime("%b")
    out["NASA_GHI_Share"] = out["NASA_ALLSKY_GHI_kWh_m2_month"] / out["NASA_ALLSKY_GHI_kWh_m2_month"].sum()
    return out


def reset_with_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    out = out.rename(columns={out.columns[0]: "Date"})
    return out


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def calc_metrics(comp: pd.DataFrame) -> pd.DataFrame:
    soda_mwh = comp["SoDa_MWh"].to_numpy(float)
    nasa_scaled = comp["NASA_scaled_MWh_to_SoDa_Annual"].to_numpy(float)
    soda_share = comp["SoDa_Share"].to_numpy(float)
    nasa_share = comp["NASA_GHI_Share"].to_numpy(float)

    err_mwh = soda_mwh - nasa_scaled
    err_share = soda_share - nasa_share

    annual_ghi = comp["NASA_ALLSKY_GHI_kWh_m2_month"].sum()
    annual_soda_mwh = comp["SoDa_MWh"].sum()
    specific_yield = annual_soda_mwh / PV_CAPACITY_MW  # MWh/MW = kWh/kW
    implied_pr = specific_yield / annual_ghi if annual_ghi > 0 else np.nan

    return pd.DataFrame([{
        "n_months": len(comp),
        "annual_soda_mwh": annual_soda_mwh,
        "annual_nasa_allsky_ghi_kwh_m2": annual_ghi,
        "soda_specific_yield_kwh_per_kwp": specific_yield,
        "implied_performance_ratio_soda_vs_nasa_ghi": implied_pr,
        "pearson_r_monthly_share": pearson_r(soda_share, nasa_share),
        "rmse_monthly_share": float(np.sqrt(np.mean(err_share**2))),
        "mbe_monthly_share": float(np.mean(err_share)),
        "mae_monthly_share": float(np.mean(np.abs(err_share))),
        "rrmse_monthly_share_pct": float(np.sqrt(np.mean(err_share**2)) / np.mean(nasa_share) * 100.0),
        "pearson_r_soda_mwh_vs_nasa_scaled_mwh": pearson_r(soda_mwh, nasa_scaled),
        "rmse_soda_mwh_vs_nasa_scaled_mwh": float(np.sqrt(np.mean(err_mwh**2))),
        "mbe_soda_mwh_vs_nasa_scaled_mwh": float(np.mean(err_mwh)),
        "mae_soda_mwh_vs_nasa_scaled_mwh": float(np.mean(np.abs(err_mwh))),
        "rrmse_soda_mwh_vs_nasa_scaled_mwh_pct": float(np.sqrt(np.mean(err_mwh**2)) / np.mean(nasa_scaled) * 100.0),
        "max_positive_monthly_error_mwh": float(np.max(err_mwh)),
        "max_negative_monthly_error_mwh": float(np.min(err_mwh)),
        "max_abs_monthly_error_mwh": float(np.max(np.abs(err_mwh))),
    }])


def make_plots(comp: pd.DataFrame, nasa_daily: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(comp["Month_Name"], comp["SoDa_Share"] * 100, marker="o", label="SoDa monthly energy share")
    ax.plot(comp["Month_Name"], comp["NASA_GHI_Share"] * 100, marker="s", label="NASA POWER GHI share")
    ax.set_ylabel("Monthly share of annual total (%)")
    ax.set_title("Level-1 Consistency: SoDa Energy Share vs NASA POWER GHI Share")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_01_soda_vs_nasa_monthly_share.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(comp["Month_Name"], comp["SoDa_MWh"], marker="o", label="SoDa monthly energy")
    ax.plot(comp["Month_Name"], comp["NASA_scaled_MWh_to_SoDa_Annual"], marker="s", label="NASA-scaled monthly energy")
    ax.set_ylabel("Monthly energy (MWh)")
    ax.set_title("SoDa Monthly Energy vs NASA POWER Pattern Scaled to Same Annual Energy")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_02_soda_mwh_vs_nasa_scaled_mwh.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(nasa_daily.index, nasa_daily["ALLSKY_SFC_SW_DWN"], linewidth=0.8)
    ax.set_ylabel("Daily GHI (kWh/m²/day)")
    ax.set_title("NASA POWER Daily All-Sky Surface Shortwave Irradiance")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_03_nasa_daily_ghi.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Level-1 consistency: SoDa monthly energy vs NASA POWER daily GHI.")
    parser.add_argument("--soda", type=str, default=DEFAULT_SODA)
    parser.add_argument("--nasa", type=str, default=DEFAULT_NASA)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-year", type=int, default=EXPECTED_YEAR)
    parser.add_argument("--allow-year-mismatch", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading SoDa final profile...")
    soda = load_soda_profile(Path(args.soda), args.expected_year, args.allow_year_mismatch)
    print(f"  SoDa: {soda.index.min()} → {soda.index.max()} | {len(soda):,} rows")
    print(f"  SoDa unique days: {len(soda.index.normalize().unique())}")

    print("\nLoading NASA POWER daily data...")
    nasa_daily, nasa_meta = load_nasa_power_daily(Path(args.nasa), args.expected_year, args.allow_year_mismatch)
    print(f"  NASA: {nasa_daily.index.min()} → {nasa_daily.index.max()} | {len(nasa_daily):,} rows")
    print(f"  NASA metadata: {nasa_meta}")

    nasa_daily_aligned, alignment_info = align_nasa_to_soda_calendar(nasa_daily, soda)
    print("\nCalendar alignment:")
    print(json.dumps(alignment_info, indent=2))

    soda_monthly = monthly_soda_energy(soda)
    nasa_monthly = monthly_nasa_resource(nasa_daily_aligned)

    comp = pd.merge(
        reset_with_date(soda_monthly),
        reset_with_date(nasa_monthly),
        on=["Date", "Month", "Month_Name"],
        how="inner",
    )

    if len(comp) != 12:
        raise RuntimeError(f"Expected 12 monthly rows after merging, got {len(comp)}.")

    annual_soda_mwh = comp["SoDa_MWh"].sum()
    comp["NASA_scaled_MWh_to_SoDa_Annual"] = comp["NASA_GHI_Share"] * annual_soda_mwh
    comp["Error_SoDa_minus_NASA_scaled_MWh"] = comp["SoDa_MWh"] - comp["NASA_scaled_MWh_to_SoDa_Annual"]
    comp["Error_Share"] = comp["SoDa_Share"] - comp["NASA_GHI_Share"]
    comp["Abs_Error_Share"] = comp["Error_Share"].abs()

    metrics = calc_metrics(comp)

    soda_monthly.to_csv(out_dir / "01_soda_monthly_energy.csv", index=True)
    nasa_monthly.to_csv(out_dir / "02_nasa_monthly_resource.csv", index=True)
    comp.to_csv(out_dir / "03_level1_monthly_comparison.csv", index=False)
    metrics.to_csv(out_dir / "04_level1_metrics.csv", index=False)

    with open(out_dir / "nasa_power_metadata.json", "w", encoding="utf-8") as f:
        json.dump(nasa_meta, f, indent=2)
    with open(out_dir / "calendar_alignment_info.json", "w", encoding="utf-8") as f:
        json.dump(alignment_info, f, indent=2)

    make_plots(comp, nasa_daily_aligned, out_dir)

    print("\nLEVEL-1 METRICS")
    print(metrics.T)

    print("\nMonthly comparison:")
    cols = [
        "Month_Name",
        "SoDa_MWh",
        "NASA_ALLSKY_GHI_kWh_m2_month",
        "SoDa_Share",
        "NASA_GHI_Share",
        "NASA_scaled_MWh_to_SoDa_Annual",
        "Error_SoDa_minus_NASA_scaled_MWh",
    ]
    print(comp[cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {out_dir / '01_soda_monthly_energy.csv'}")
    print(f"  {out_dir / '02_nasa_monthly_resource.csv'}")
    print(f"  {out_dir / '03_level1_monthly_comparison.csv'}")
    print(f"  {out_dir / '04_level1_metrics.csv'}")
    print(f"  {out_dir / 'fig_01_soda_vs_nasa_monthly_share.png'}")
    print(f"  {out_dir / 'fig_02_soda_mwh_vs_nasa_scaled_mwh.png'}")
    print(f"  {out_dir / 'fig_03_nasa_daily_ghi.png'}")


if __name__ == "__main__":
    main()
