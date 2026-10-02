# Thesis BESS-SoDa Workflow

This repository contains the Python scripts used for the thesis:

**Evaluation and Optimal Sizing of Battery Energy Storage Systems for Ramp-rate Control in a 100 MW Solar Power Plant Using Synthetic SoDa Power Profiles**

This repository is structured as a re-auditable code package. The main workflow aligns with the final thesis book: synthetic SoDa PV power profile generation, sanity check against NASA POWER, BESS capacity optimization via deterministic grid search, economic sensitivity analysis, environmental indicators, DIgSILENT input preparation, and a PSO baseline comparison using Deb's Feasibility Rules.

## Structure

```text
thesis-bess-soda-github-ready/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ scripts/
│  ├─ solar_data.py
│  ├─ 01_generate_soda_profile.py
│  ├─ 02_check_soda_nasa_consistency.py
│  ├─ 03_optimize_bess_grid_search.py
│  ├─ 04_economic_sensitivity_r5.py
│  ├─ 05_prepare_digsilent_inputs.py
│  ├─ 06_environmental_indicator.py
│  ├─ 07_optimize_bess_pso_deb_rules.py
│  └─ 08_master_summary_for_bab4.py
├─ docs/
│  └─ NOTE_PENGGUNAAN_KODE.md
└─ archive/
   ├─ 05_bess_pso_comparison_R5.py
   ├─ 05_bess_pso_all_ramp_scenarios.py
   ├─ 05_bess_pso_all_ramp_scenarios_CONTINUOUS_DEBUG.py
   ├─ 06_bess_pso_multiseed_debug_FIXED.py
   └─ 08_economic_sensitivity_rerun_optimization_grid_pso.py
```

## Main Scripts

1. `01_generate_soda_profile.py`  
   Generates the 1-minute resolution 2020 synthetic SoDa PV power profile.

2. `02_check_soda_nasa_consistency.py`  
   Performs a sanity check comparing monthly SoDa energy against NASA POWER.

3. `03_optimize_bess_grid_search.py`  
   Main script for BESS capacity optimization using deterministic grid search for R20, R10, R5, and R3 scenarios.

4. `04_economic_sensitivity_r5.py`  
   Economic sensitivity analysis for R5 without re-optimizing capacity.

5. `05_prepare_digsilent_inputs.py`  
   Prepares PV and BESS time characteristics for DIgSILENT based on R5 results.

6. `06_environmental_indicator.py`  
   Calculates indicative CO₂ equivalent indicators based on annual BESS discharge.

7. `07_optimize_bess_pso_deb_rules.py`  
   Final PSO comparison script incorporating Deb's Feasibility Rules, multi-seed execution, and 4-decimal candidate resolution.

8. `08_master_summary_for_bab4.py`  
   Helper script to aggregate primary outputs into the Chapter 4 summary.

## Security Notes

NSRDB/NREL API keys are not stored in the source code. Before running the SoDa generator, set the environment variable:

```bat
set NREL_API_KEY=YOUR_API_KEY_HERE
```

Pada PowerShell:

```powershell
$env:NREL_API_KEY="YOUR_API_KEY_HERE"
```

## Limitations

- SoDa profiles are synthetic profiles intended for pre-feasibility studies, not actual field measurement data.
- NASA POWER is utilized as a reference for monthly climatological patterns, not as ground truth for 1-minute resolution PV power.
- The BESS is evaluated strictly for ramp-rate smoothing—not for energy shifting, arbitrage, frequency regulation, spinning reserve, or peak shaving.
- CO2eq indicators are indicative only and do not constitute claims of actual emission reductions or carbon credits.
- Legacy/debug files are preserved in archive/ solely for historical tracking and are not part of the primary pipeline.

## Acknowledgments
The code in this repository is based on the methodology developed by Ignacio-Losada. I have used the codebase from the [SoDa repository](https://github.com/dpinney/SoDa), which serves as the reference implementation for this work.

- **Original Author:** Ignacio-Losada
- **Reference Repository:** [dpinney/SoDa](https://github.com/dpinney/SoDa)
- **License:** The referenced work is licensed under the [MIT License](https://github.com/dpinney/SoDa?tab=MIT-1-ov-file), a copy of which is included in this repository.
