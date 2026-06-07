# -*- coding: utf-8 -*-
"""
===============================================================================
FINAL SoDa 1-Year 1-Minute Synthetic PV Generator
===============================================================================

Purpose
-------
Generate a 1-year, 1-minute synthetic PV power profile using the original
SoDa implementation in solar_data.py.

This script intentionally DOES NOT modify the original SoDa stochastic model.
It only:
1. Imports SolarSite from solar_data.py.
2. Downloads NSRDB/Himawari resource data.
3. Generates baseline PV power using PySAM PVWatts.
4. Calls the original SoDa high-resolution generator day by day.
5. Concatenates all days into a full-year 1-minute profile.
6. Performs strict quality checks.
7. Exports final CSV, monthly energy, and summary statistics.

Important
---------
- Output column: HighRes
- Unit assumed in this thesis workflow: MW
- Energy calculation: MWh = MW × timestep_hours
- For 1-minute data: MWh = MW × (1/60)

Author note
-----------
Use this file for the final thesis dataset.
Do not edit solar_data.py again unless absolutely necessary.

Perubahan 26 Mei 2025
Data Ekonomi Sesuai NREL ATB 2025 
Azimuth dari 180 ke 0 sesuai pvsam
===============================================================================
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from solar_data import SolarSite


# =============================================================================
# FINAL USER PARAMETERS
# =============================================================================

# -------------------------------------------------------------------------
# Location
# Replace these with the EXACT coordinates you want to use for Mekarwaru/Subang.
# Use the same coordinates you used in your previous successful SoDa run.
# -------------------------------------------------------------------------
LATITUDE = -6.58558
LONGITUDE = 107.89692

# -------------------------------------------------------------------------
# Time settings
# -------------------------------------------------------------------------
YEAR = 2020
LEAP_YEAR = False          # False = 365 days = 525,600 minutes
INTERVAL = "30"            # NSRDB/Himawari interval: "30" or "60"
UTC = False                # SAM/PVWatts generally expects local time

# -------------------------------------------------------------------------
# PV system settings
# Keep these consistent with your thesis assumption.
# -------------------------------------------------------------------------
PV_CAPACITY_MW = 100.0
DC_AC_RATIO = 1.10
TILT_DEG = 10.0
AZIMUTH_DEG = 0.0
INV_EFF_PERCENT = 96.0
LOSSES_PERCENT = 14.0

# PVWatts array_type:
# 0 = Fixed open rack
# 1 = Fixed roof mount
# 2 = 1-axis tracking
# 3 = 1-axis backtracking
# 4 = 2-axis tracking
ARRAY_TYPE = 0

# -------------------------------------------------------------------------
# Output settings
# -------------------------------------------------------------------------
OUTPUT_RESOLUTION = "1min"
BASE_SEED = 42

OUTPUT_DIR = Path("soda_final_outputs")
OUTPUT_CSV = OUTPUT_DIR / "synthetic_pv_soda_2020_FINAL.csv"

# Optional physical cleanup.
# Leave True for final thesis dataset to prevent impossible >100 MW output.
CLIP_TO_CAPACITY = True
MAX_POWER_MW = PV_CAPACITY_MW


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    """Infer median timestep in hours from datetime index."""
    if len(index) < 2:
        return 1.0 / 60.0

    diffs = index.to_series().diff().dropna().dt.total_seconds().values
    median_seconds = float(np.median(diffs))
    return median_seconds / 3600.0


def expected_timestep_count(year: int, leap_year: bool) -> int:
    """Expected number of 1-minute timesteps."""
    return (366 if leap_year else 365) * 24 * 60


def clean_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize final profile."""
    if "HighRes" not in df.columns:
        raise ValueError("Final dataframe must contain column 'HighRes'.")

    out = df.copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out["HighRes"] = out["HighRes"].astype(float).fillna(0.0)

    # Remove tiny or impossible negative values.
    out.loc[out["HighRes"] < 0.0, "HighRes"] = 0.0

    if CLIP_TO_CAPACITY:
        out.loc[out["HighRes"] > MAX_POWER_MW, "HighRes"] = MAX_POWER_MW

    return out


def strict_quality_check(df: pd.DataFrame) -> None:
    """Raise error if final dataset is not thesis-ready."""
    expected_n = expected_timestep_count(YEAR, LEAP_YEAR)

    if len(df) != expected_n:
        raise RuntimeError(
            f"Unexpected timestep count. Expected {expected_n:,}, got {len(df):,}."
        )

    if df.index.duplicated().any():
        raise RuntimeError("Duplicate timestamps found.")

    if df["HighRes"].isna().any():
        raise RuntimeError("NaN values found in HighRes.")

    if (df["HighRes"] < -1e-9).any():
        raise RuntimeError("Negative values found in HighRes.")

    if CLIP_TO_CAPACITY and df["HighRes"].max() > MAX_POWER_MW + 1e-6:
        raise RuntimeError("HighRes exceeds PV capacity after clipping.")

    dt_h = infer_timestep_hours(df.index)
    if abs(dt_h - 1.0 / 60.0) > 1e-9:
        raise RuntimeError(f"Expected 1-minute timestep, got {dt_h:.12f} hours.")

    start_expected = pd.Timestamp(f"{YEAR}-01-01 00:00:00")
    end_expected = pd.Timestamp(f"{YEAR}-12-31 23:59:00")

    if df.index.min() != start_expected:
        raise RuntimeError(
            f"Unexpected start timestamp. Expected {start_expected}, got {df.index.min()}."
        )

    if df.index.max() != end_expected:
        raise RuntimeError(
            f"Unexpected end timestamp. Expected {end_expected}, got {df.index.max()}."
        )


def summarize_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Create final summary table."""
    pv = df["HighRes"]
    dt_h = infer_timestep_hours(df.index)

    annual_mwh = pv.sum() * dt_h
    cf_pct = annual_mwh / (PV_CAPACITY_MW * 8760.0) * 100.0

    ramps = np.abs(np.diff(pv.values))

    summary = {
        "year": YEAR,
        "n_timestep": len(df),
        "start_time": str(df.index.min()),
        "end_time": str(df.index.max()),
        "timestep_hours": dt_h,
        "pv_capacity_mw": PV_CAPACITY_MW,
        "max_mw": float(pv.max()),
        "mean_mw": float(pv.mean()),
        "annual_mwh": float(annual_mwh),
        "capacity_factor_pct": float(cf_pct),
        "missing_values": int(pv.isna().sum()),
        "duplicate_index": int(df.index.duplicated().sum()),
        "negative_values": int((pv < 0).sum()),
        "zero_values": int((pv == 0).sum()),
        "mean_abs_ramp_mw_per_min": float(np.mean(ramps)),
        "std_abs_ramp_mw_per_min": float(np.std(ramps)),
        "p90_ramp_mw_per_min": float(np.percentile(ramps, 90)),
        "p95_ramp_mw_per_min": float(np.percentile(ramps, 95)),
        "p99_ramp_mw_per_min": float(np.percentile(ramps, 99)),
        "p999_ramp_mw_per_min": float(np.percentile(ramps, 99.9)),
        "max_ramp_mw_per_min": float(np.max(ramps)),
        "violations_R20_count": int((ramps > 20.0).sum()),
        "violations_R20_pct": float((ramps > 20.0).mean() * 100.0),
        "violations_R10_count": int((ramps > 10.0).sum()),
        "violations_R10_pct": float((ramps > 10.0).mean() * 100.0),
        "violations_R5_count": int((ramps > 5.0).sum()),
        "violations_R5_pct": float((ramps > 5.0).mean() * 100.0),
        "violations_R3_count": int((ramps > 3.0).sum()),
        "violations_R3_pct": float((ramps > 3.0).mean() * 100.0),
    }

    return pd.DataFrame([summary])


def save_generation_parameters() -> None:
    """Save all generation assumptions."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "year": YEAR,
        "leap_year": LEAP_YEAR,
        "interval": INTERVAL,
        "utc": UTC,
        "pv_capacity_mw": PV_CAPACITY_MW,
        "dc_ac_ratio": DC_AC_RATIO,
        "tilt_deg": TILT_DEG,
        "azimuth_deg": AZIMUTH_DEG,
        "inv_eff_percent": INV_EFF_PERCENT,
        "losses_percent": LOSSES_PERCENT,
        "array_type": ARRAY_TYPE,
        "output_resolution": OUTPUT_RESOLUTION,
        "base_seed": BASE_SEED,
        "clip_to_capacity": CLIP_TO_CAPACITY,
        "max_power_mw": MAX_POWER_MW,
        "soda_implementation": "original solar_data.py",
    }

    with open(OUTPUT_DIR / "generation_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


# =============================================================================
# MAIN GENERATION
# =============================================================================

def main() -> None:
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FINAL SoDa 1-Year 1-Minute Synthetic PV Generation")
    print("=" * 80)
    print(f"Location        : lat={LATITUDE}, lon={LONGITUDE}")
    print(f"Year            : {YEAR}")
    print(f"Leap year       : {LEAP_YEAR}")
    print(f"NSRDB interval  : {INTERVAL} minutes")
    print(f"UTC             : {UTC}")
    print(f"PV capacity     : {PV_CAPACITY_MW:.2f} MW")
    print(f"DC/AC ratio     : {DC_AC_RATIO}")
    print(f"Tilt / Azimuth  : {TILT_DEG} / {AZIMUTH_DEG}")
    print(f"Array type      : {ARRAY_TYPE}")
    print(f"Resolution      : {OUTPUT_RESOLUTION}")
    print(f"Random seed     : {BASE_SEED}")
    print(f"Output CSV      : {OUTPUT_CSV}")
    print("=" * 80)

    save_generation_parameters()

    # One global seed. This preserves a continuous stochastic realization
    # across the full yearly generation loop.
    np.random.seed(BASE_SEED)

    site = SolarSite(LATITUDE, LONGITUDE)

    print("\n[1/5] Downloading NSRDB/Himawari resource data...")
    resource_df = site.get_nsrdb_data(
        year=YEAR,
        leap_year=LEAP_YEAR,
        interval=INTERVAL,
        utc=UTC,
    )

    print(f"Resource shape      : {resource_df.shape}")
    print(f"Resource time range : {resource_df.index.min()} → {resource_df.index.max()}")
    print(f"Resource columns    : {list(resource_df.columns)}")

    print("\n[2/5] Generating baseline PVWatts power profile...")
    baseline_df = site.generate_solar_power_from_nsrdb(
        clearsky=False,
        capacity=PV_CAPACITY_MW,
        DC_AC_ratio=DC_AC_RATIO,
        tilt=TILT_DEG,
        azimuth=AZIMUTH_DEG,
        inv_eff=INV_EFF_PERCENT,
        losses=LOSSES_PERCENT,
        array_type=ARRAY_TYPE,
    )

    if "generation" not in baseline_df.columns:
        raise RuntimeError("PVWatts output does not contain 'generation' column.")

    print(f"Baseline shape      : {baseline_df.shape}")
    print(f"Baseline time range : {baseline_df.index.min()} → {baseline_df.index.max()}")
    print(f"Baseline max        : {baseline_df['generation'].max():.6f}")
    print(f"Baseline mean       : {baseline_df['generation'].mean():.6f}")

    print("\n[3/5] Preparing daily generation list...")
    dates = pd.date_range(
        start=f"{YEAR}-01-01",
        end=f"{YEAR}-12-31",
        freq="D",
    )

    if not LEAP_YEAR:
        dates = dates[~((dates.month == 2) & (dates.day == 29))]

    print(f"Number of days      : {len(dates)}")

    print("\n[4/5] Generating high-resolution SoDa profile day by day...")
    all_days = []

    for i, day in enumerate(dates, start=1):
        date_str = day.strftime("%Y-%m-%d")

        try:
            day_df = site.generate_high_resolution_power_data(
                resolution=OUTPUT_RESOLUTION,
                date=date_str,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed generating high-resolution data for {date_str}: {exc}") from exc

        if "HighRes" not in day_df.columns:
            raise RuntimeError(f"{date_str}: generated dataframe has no HighRes column.")

        if len(day_df) != 1440:
            raise RuntimeError(f"{date_str}: expected 1440 rows, got {len(day_df)}.")

        all_days.append(day_df)

        if i % 30 == 0 or i == len(dates):
            print(f"  generated {i:>3}/{len(dates)} days | latest: {date_str}")

    print("\n[5/5] Concatenating, cleaning, checking, and saving...")
    solar_year = pd.concat(all_days)
    solar_year = clean_profile(solar_year)

    strict_quality_check(solar_year)

    summary_df = summarize_profile(solar_year)
    monthly_energy = (
        solar_year["HighRes"].resample("ME").sum()
        * infer_timestep_hours(solar_year.index)
    ).rename("Monthly_MWh")

    solar_year.to_csv(OUTPUT_CSV)
    summary_df.to_csv(OUTPUT_DIR / "soda_final_summary.csv", index=False)
    monthly_energy.to_csv(OUTPUT_DIR / "soda_final_monthly_energy.csv")

    print("\nFINAL PROFILE SUMMARY")
    print(summary_df.T)

    print("\nMONTHLY ENERGY (MWh)")
    print(monthly_energy.round(2))

    print("\nSaved files:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_DIR / 'soda_final_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'soda_final_monthly_energy.csv'}")
    print(f"  {OUTPUT_DIR / 'generation_parameters.json'}")

    elapsed = time.time() - t_start
    print(f"\nDONE. Total runtime: {elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()