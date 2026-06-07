# -*- coding: utf-8 -*-
"""
07_master_summary_for_bab4.py

Master summary builder for Bab 4 thesis writing.

Purpose
-------
Collect final output files from the SoDa + BESS ramp-rate workflow into one
traceable Excel workbook and a compact key-metrics CSV. This script does not
change any result and does not perform optimization. It only reads outputs,
standardizes key tables, and records file availability.

Recommended pipeline order
--------------------------
1. python generate_soda_1year_final.py
2. python 04_level1_nasa_power_consistency.py
3. python 01_bess_sizing_ramp_scenarios.py --input soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv
4. python 05_bess_pso_all_ramp_scenarios.py --input soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv --scenario ALL
5. python 02_bess_economic_sensitivity_R5.py
6. python 03_prepare_digsilent_time_characteristics.py
7. python 06_environmental_indicator.py
8. python 07_master_summary_for_bab4.py

Default output
--------------
- master_summary_outputs/00_master_results_for_bab4.xlsx
- master_summary_outputs/00_key_metrics_for_bab4.csv
- master_summary_outputs/00_file_availability.csv

Notes
-----
If a file is missing, this script does not fail by default. It records the
missing file in the availability sheet. Use --strict to fail on missing files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "master_summary_outputs"

EXPECTED_FILES = {
    "soda_summary": "soda_final_outputs/soda_final_summary.csv",
    "soda_monthly_energy": "soda_final_outputs/soda_monthly_energy.csv",
    "soda_generation_parameters": "soda_final_outputs/generation_parameters.json",
    "consistency_monthly": "consistency_level1_outputs/03_level1_monthly_comparison.csv",
    "consistency_metrics": "consistency_level1_outputs/04_level1_metrics.csv",
    "calendar_alignment": "consistency_level1_outputs/calendar_alignment_info.json",
    "nasa_metadata": "consistency_level1_outputs/nasa_power_metadata.json",
    "pv_summary": "bess_sizing_outputs/01_pv_summary.csv",
    "ramp_before_bess": "bess_sizing_outputs/02_ramp_statistics_before_bess.csv",
    "grid_sizing": "bess_sizing_outputs/03_bess_sizing_ramp_scenarios.csv",
    "top_feasible_candidates": "bess_sizing_outputs/04_top_feasible_candidates.csv",
    "pso_summary": "pso_comparison_outputs/07_pso_summary.csv",
    "pso_vs_grid": "pso_comparison_outputs/07_pso_vs_grid_comparison.csv",
    "economic_sensitivity_r5": "economic_sensitivity_outputs/06_economic_sensitivity_R5.csv",
    "environmental_indicator": "environmental_outputs/10_environmental_indicator.csv",
    "digsilent_metadata": "digsilent_time_characteristics/digsilent_selected_day_metadata.json",
}


def safe_sheet_name(name: str) -> str:
    # Excel sheet names max 31 chars and cannot contain []:*?/\\
    bad = '[]:*?/\\'
    out = ''.join('_' if ch in bad else ch for ch in name)
    return out[:31]


def read_json_flat(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        rows.append({"parameter": k, "value": v})
    return pd.DataFrame(rows)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return read_json_flat(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise RuntimeError(f"Unsupported file format: {path}")


def file_availability(base_dir: Path) -> pd.DataFrame:
    rows = []
    for label, rel in EXPECTED_FILES.items():
        p = base_dir / rel
        rows.append({
            "label": label,
            "relative_path": rel,
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
        })
    return pd.DataFrame(rows)


def first_value(df: pd.DataFrame, candidates: list[str]) -> Any:
    for c in candidates:
        if c in df.columns and len(df) > 0:
            return df[c].iloc[0]
    return None


def build_key_metrics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(group: str, metric: str, value: Any, unit: str = "", source_table: str = "") -> None:
        if value is not None:
            rows.append({
                "group": group,
                "metric": metric,
                "value": value,
                "unit": unit,
                "source_table": source_table,
            })

    # SoDa/PV summary.
    for key in ["soda_summary", "pv_summary"]:
        if key in tables:
            df = tables[key]
            add("SoDa profile", "n_timestep", first_value(df, ["n_timestep"]), "count", key)
            add("SoDa profile", "start_time", first_value(df, ["start_time"]), "", key)
            add("SoDa profile", "end_time", first_value(df, ["end_time"]), "", key)
            add("SoDa profile", "max_mw", first_value(df, ["max_mw"]), "MW", key)
            add("SoDa profile", "mean_mw", first_value(df, ["mean_mw"]), "MW", key)
            add("SoDa profile", "annual_mwh", first_value(df, ["annual_mwh"]), "MWh/year", key)
            add("SoDa profile", "capacity_factor_pct", first_value(df, ["capacity_factor_pct"]), "%", key)
            add("SoDa profile", "missing_values", first_value(df, ["missing_values"]), "count", key)
            add("SoDa profile", "duplicate_index", first_value(df, ["duplicate_index", "duplicate_timestamp"]), "count", key)
            add("SoDa profile", "negative_values", first_value(df, ["negative_values"]), "count", key)
            break

    # Consistency metrics.
    if "consistency_metrics" in tables:
        df = tables["consistency_metrics"]
        add("Consistency", "pearson_r_monthly_share", first_value(df, ["pearson_r_monthly_share"]), "-", "consistency_metrics")
        add("Consistency", "rrmse_monthly_share_pct", first_value(df, ["rrmse_monthly_share_pct"]), "%", "consistency_metrics")
        add("Consistency", "annual_nasa_allsky_ghi", first_value(df, ["annual_nasa_allsky_ghi_kwh_m2"]), "kWh/m2/year", "consistency_metrics")
        add("Consistency", "implied_pr", first_value(df, ["implied_performance_ratio_soda_vs_nasa_ghi"]), "-", "consistency_metrics")

    # Ramp before BESS.
    if "ramp_before_bess" in tables:
        df = tables["ramp_before_bess"]
        if "scenario" in df.columns:
            for _, r in df.iterrows():
                scen = r.get("scenario", "")
                add("Ramp before BESS", f"{scen}_violations", r.get("violations"), "count", "ramp_before_bess")
                add("Ramp before BESS", f"{scen}_violations_pct", r.get("violations_pct"), "%", "ramp_before_bess")
                add("Ramp before BESS", f"{scen}_max_ramp", r.get("max_mw_per_min"), "MW/min", "ramp_before_bess")
        else:
            for col in df.columns:
                if "violations" in col.lower() or "max" in col.lower() or "p99" in col.lower():
                    add("Ramp before BESS", col, first_value(df, [col]), "", "ramp_before_bess")

    # Grid sizing.
    if "grid_sizing" in tables:
        df = tables["grid_sizing"]
        for _, r in df.iterrows():
            scen = r.get("scenario", "")
            add("Grid sizing", f"{scen}_p_bess_mw", r.get("p_bess_mw"), "MW", "grid_sizing")
            add("Grid sizing", f"{scen}_e_bess_mwh", r.get("e_bess_mwh"), "MWh", "grid_sizing")
            add("Grid sizing", f"{scen}_annualized_cost", r.get("annualized_cost_usd_per_year"), "USD/year", "grid_sizing")
            add("Grid sizing", f"{scen}_annual_discharge", r.get("annual_discharge_mwh"), "MWh/year", "grid_sizing")
            add("Grid sizing", f"{scen}_violations_after", r.get("violations"), "count", "grid_sizing")
            add("Grid sizing", f"{scen}_compliance", r.get("compliance_pct"), "%", "grid_sizing")

    # PSO vs grid comparison.
    if "pso_vs_grid" in tables:
        df = tables["pso_vs_grid"]
        for _, r in df.iterrows():
            scen = r.get("scenario", "")
            for col in ["p_bess_diff_mw", "e_bess_diff_mwh", "annualized_cost_diff_usd_per_year", "same_capacity"]:
                if col in df.columns:
                    add("PSO comparison", f"{scen}_{col}", r.get(col), "", "pso_vs_grid")

    # Environmental indicator.
    if "environmental_indicator" in tables:
        df = tables["environmental_indicator"]
        for _, r in df.iterrows():
            scen = r.get("scenario", "")
            add("Environmental", f"{scen}_co2eq", r.get("co2eq_ton_per_year"), "ton CO2/year", "environmental_indicator")

    # DIgSILENT metadata.
    if "digsilent_metadata" in tables:
        df = tables["digsilent_metadata"]
        for metric in [
            "selected_day",
            "selection_mode",
            "selected_day_raw_max_ramp_mw_per_min",
            "selected_day_raw_r5_violation_count",
            "selected_day_pv_energy_mwh",
            "selected_day_bess_discharge_mwh",
            "selected_day_bess_charge_mwh",
            "selected_day_soc_min",
            "selected_day_soc_max",
        ]:
            row = df[df["parameter"] == metric]
            if not row.empty:
                add("DIgSILENT", metric, row["value"].iloc[0], "", "digsilent_metadata")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build master summary workbook for Bab 4.")
    parser.add_argument("--base-dir", type=str, default=".", help="Project working directory containing output folders.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected file is missing.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    availability = file_availability(base_dir)
    missing = availability[~availability["exists"]]
    if args.strict and not missing.empty:
        raise FileNotFoundError("Missing expected files:\n" + missing[["label", "relative_path"]].to_string(index=False))

    tables: dict[str, pd.DataFrame] = {}
    load_errors: list[dict[str, str]] = []

    for label, rel in EXPECTED_FILES.items():
        p = base_dir / rel
        if not p.exists():
            continue
        try:
            tables[label] = read_table(p)
        except Exception as exc:
            load_errors.append({"label": label, "relative_path": rel, "error": str(exc)})

    key_metrics = build_key_metrics(tables)
    load_errors_df = pd.DataFrame(load_errors)

    xlsx_path = out_dir / "00_master_results_for_bab4.xlsx"
    key_csv_path = out_dir / "00_key_metrics_for_bab4.csv"
    availability_csv_path = out_dir / "00_file_availability.csv"

    key_metrics.to_csv(key_csv_path, index=False)
    availability.to_csv(availability_csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        availability.to_excel(writer, sheet_name="file_availability", index=False)
        key_metrics.to_excel(writer, sheet_name="key_metrics", index=False)
        if not load_errors_df.empty:
            load_errors_df.to_excel(writer, sheet_name="load_errors", index=False)

        for label, df in tables.items():
            sheet = safe_sheet_name(label)
            # Avoid Excel sheet row limit issues by clipping extremely large tables.
            if len(df) > 1_000_000:
                df.head(1_000_000).to_excel(writer, sheet_name=sheet, index=False)
            else:
                df.to_excel(writer, sheet_name=sheet, index=False)

    print("Saved master summary outputs:")
    print(f"  XLSX: {xlsx_path}")
    print(f"  Key metrics CSV: {key_csv_path}")
    print(f"  File availability CSV: {availability_csv_path}")

    if not missing.empty:
        print("\nMissing files recorded in file availability table:")
        print(missing[["label", "relative_path"]].to_string(index=False))

    if not load_errors_df.empty:
        print("\nFiles found but failed to load:")
        print(load_errors_df.to_string(index=False))

    print("\nKey metrics preview:")
    if key_metrics.empty:
        print("  No key metrics extracted yet. Run the upstream pipeline first.")
    else:
        print(key_metrics.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
