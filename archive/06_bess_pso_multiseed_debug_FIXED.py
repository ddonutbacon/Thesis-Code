# -*- coding: utf-8 -*-
"""
06_bess_pso_multiseed_debug.py

Multi-seed PSO robustness/debug runner for the thesis BESS sizing workflow.

Purpose
-------
Run the existing PSO implementation repeatedly using different random seeds,
then summarize whether PSO consistently reaches the same BESS sizing result as
Deterministic Grid Search.

This script DOES NOT change the BESS control logic, objective function, or PSO
algorithm in 05_bess_pso_all_ramp_scenarios.py. It only wraps repeated runs and
aggregates the outputs for defense/reporting.

Example
-------
python 06_bess_pso_multiseed_debug.py \
  --input soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv \
  --scenario ALL \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --particles 30 \
  --iters 60

Outputs
-------
- pso_multiseed_outputs/summary_all_runs.csv
- pso_multiseed_outputs/summary_by_scenario.csv
- pso_multiseed_outputs/pso_grid_deltas_all_runs.csv
- pso_multiseed_outputs/seed_<N>/... convergence and best time-series files
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_pso_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("pso_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import PSO script from {script_path}")
    module = importlib.util.module_from_spec(spec)

    # Important for Python 3.11+ / 3.13 dataclasses:
    # dataclass processing expects the imported module to already exist in sys.modules.
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)
    return module


def parse_seed_list(seed_text: str | None, seed_start: int, seed_count: int) -> list[int]:
    if seed_text:
        return [int(x.strip()) for x in seed_text.split(",") if x.strip()]
    return list(range(seed_start, seed_start + seed_count))


def summarize_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scen, g in df.groupby("scenario"):
        rows.append({
            "scenario": scen,
            "n_runs": len(g),
            "unique_p_bess_mw": ", ".join(map(str, sorted(g["p_bess_mw"].unique()))),
            "unique_e_bess_mwh": ", ".join(map(str, sorted(g["e_bess_mwh"].unique()))),
            "p_mean_mw": g["p_bess_mw"].mean(),
            "p_std_mw": g["p_bess_mw"].std(ddof=0),
            "p_min_mw": g["p_bess_mw"].min(),
            "p_max_mw": g["p_bess_mw"].max(),
            "e_mean_mwh": g["e_bess_mwh"].mean(),
            "e_std_mwh": g["e_bess_mwh"].std(ddof=0),
            "e_min_mwh": g["e_bess_mwh"].min(),
            "e_max_mwh": g["e_bess_mwh"].max(),
            "cost_mean_usd_per_year": g["annualized_cost_usd_per_year"].mean(),
            "cost_std_usd_per_year": g["annualized_cost_usd_per_year"].std(ddof=0),
            "cost_min_usd_per_year": g["annualized_cost_usd_per_year"].min(),
            "cost_max_usd_per_year": g["annualized_cost_usd_per_year"].max(),
            "all_feasible": bool(g["feasible"].all()),
            "max_violations_seen": int(g["violations"].max()),
            "mean_runtime_s": g["runtime_s"].mean(),
        })
    return pd.DataFrame(rows)


def make_grid_delta_table(all_runs: pd.DataFrame, grid_summary_path: Path) -> pd.DataFrame:
    if not grid_summary_path.exists():
        return pd.DataFrame()

    grid = pd.read_csv(grid_summary_path)
    rows = []
    for _, row in all_runs.iterrows():
        scen = row["scenario"]
        g = grid[grid["scenario"] == scen]
        if g.empty:
            continue
        g = g.iloc[0]
        rows.append({
            "scenario": scen,
            "seed": int(row["seed"]),
            "grid_p_bess_mw": float(g["p_bess_mw"]),
            "pso_p_bess_mw": float(row["p_bess_mw"]),
            "delta_p_mw": float(row["p_bess_mw"] - g["p_bess_mw"]),
            "grid_e_bess_mwh": float(g["e_bess_mwh"]),
            "pso_e_bess_mwh": float(row["e_bess_mwh"]),
            "delta_e_mwh": float(row["e_bess_mwh"] - g["e_bess_mwh"]),
            "grid_annualized_cost_usd_per_year": float(g["annualized_cost_usd_per_year"]),
            "pso_annualized_cost_usd_per_year": float(row["annualized_cost_usd_per_year"]),
            "delta_cost_usd_per_year": float(row["annualized_cost_usd_per_year"] - g["annualized_cost_usd_per_year"]),
            "pso_violations": int(row["violations"]),
            "pso_compliance_pct": float(row["compliance_pct"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed PSO robustness check.")
    parser.add_argument("--input", type=str, default="soda_final_outputs/synthetic_pv_soda_2020_FINAL.csv")
    parser.add_argument("--scenario", type=str, default="ALL", choices=["R20", "R10", "R5", "R3", "ALL"])
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 1,2,3,4,5")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--particles", type=int, default=30)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--output", type=str, default="pso_multiseed_outputs")
    parser.add_argument("--pso-script", type=str, default="05_bess_pso_all_ramp_scenarios.py")
    parser.add_argument("--grid-summary", type=str, default="bess_sizing_outputs/03_bess_sizing_ramp_scenarios.csv")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pso_script_path = Path(args.pso_script)
    if not pso_script_path.exists():
        # Allows running from /mnt/data while user is in another working directory.
        alt = Path(__file__).resolve().parent / args.pso_script
        if alt.exists():
            pso_script_path = alt
        else:
            raise FileNotFoundError(f"PSO script not found: {args.pso_script}")

    pso = load_pso_module(pso_script_path)
    seeds = parse_seed_list(args.seeds, args.seed_start, args.seed_count)
    scenarios = ["R20", "R10", "R5", "R3"] if args.scenario == "ALL" else [args.scenario]

    pv_series = pso.load_pv_profile(Path(args.input))
    pv = pv_series.values.astype(np.float64)

    print("=" * 80)
    print("MULTI-SEED PSO DEBUG RUN")
    print("=" * 80)
    print(f"Input       : {args.input}")
    print(f"Scenarios   : {scenarios}")
    print(f"Seeds       : {seeds}")
    print(f"Particles   : {args.particles}")
    print(f"Iterations  : {args.iters}")
    print(f"Output dir  : {out_dir.resolve()}")

    results = []
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed:04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for scen in scenarios:
            print("\n" + "-" * 80)
            print(f"Running scenario={scen}, seed={seed}")
            res = pso.run_pso_for_scenario(
                pv=pv,
                scenario=scen,
                n_particles=args.particles,
                n_iter=args.iters,
                seed=seed,
                out_dir=seed_dir,
                save_trace_index=pv_series.index,
            )
            row = asdict(res)
            row["run_output_dir"] = str(seed_dir)
            results.append(row)

    all_runs = pd.DataFrame(results)
    all_runs.to_csv(out_dir / "summary_all_runs.csv", index=False)

    by_scenario = summarize_by_scenario(all_runs)
    by_scenario.to_csv(out_dir / "summary_by_scenario.csv", index=False)

    deltas = make_grid_delta_table(all_runs, Path(args.grid_summary))
    if not deltas.empty:
        deltas.to_csv(out_dir / "pso_grid_deltas_all_runs.csv", index=False)

    print("\n" + "=" * 80)
    print("SUMMARY BY SCENARIO")
    print("=" * 80)
    print(by_scenario.to_string(index=False))

    print("\nSaved:")
    print(f"  {out_dir / 'summary_all_runs.csv'}")
    print(f"  {out_dir / 'summary_by_scenario.csv'}")
    if not deltas.empty:
        print(f"  {out_dir / 'pso_grid_deltas_all_runs.csv'}")


if __name__ == "__main__":
    main()
