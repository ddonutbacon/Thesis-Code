# Note Penggunaan Kode Tesis

Dokumen ini menjelaskan cara menjalankan skrip tesis melalui Anaconda Prompt di Windows.

## 1. Membuat environment

Buka **Anaconda Prompt**, lalu jalankan:

```bat
conda create -n thesis-bess-soda python=3.11 -y
conda activate thesis-bess-soda
pip install -r requirements.txt
```

Apabila `NREL-PySAM` gagal terpasang melalui `pip`, coba:

```bat
conda install -c conda-forge nrel-pysam -y
```

## 2. Menyiapkan API key NSRDB/NREL

Generator SoDa membutuhkan API key untuk mengambil data NSRDB/Himawari. API key tidak disimpan di kode agar aman untuk GitHub.

Di Anaconda Prompt:

```bat
set NREL_API_KEY=ISI_API_KEY_ANDA_DI_SINI
```

Di PowerShell:

```powershell
$env:NREL_API_KEY="ISI_API_KEY_ANDA_DI_SINI"
```

## 3. Urutan menjalankan pipeline utama

Jalankan dari folder utama repo.

### 3.1 Membuat profil SoDa 1 menit tahun 2020

```bat
python scripts\01_generate_soda_profile.py
```

Output utama:

```text
soda_final_outputs\synthetic_pv_soda_2020_FINAL.csv
soda_final_outputs\soda_final_summary.csv
soda_final_outputs\soda_monthly_energy.csv
soda_final_outputs\generation_parameters.json
```

### 3.2 Pemeriksaan kewajaran SoDa terhadap NASA POWER

Pastikan file NASA POWER harian berada di folder utama repo dengan nama yang sesuai default script:

```text
POWER_Point_Daily_20200101_20201231_006d58S_107d90E_LST.csv
```

Lalu jalankan:

```bat
python scripts\02_check_soda_nasa_consistency.py
```

Output utama:

```text
consistency_level1_outputs\03_level1_monthly_comparison.csv
consistency_level1_outputs\04_level1_metrics.csv
```

### 3.3 Optimasi kapasitas BESS dengan deterministic grid search

```bat
python scripts\03_optimize_bess_grid_search.py --input soda_final_outputs\synthetic_pv_soda_2020_FINAL.csv
```

Output utama:

```text
bess_sizing_outputs\01_pv_summary.csv
bess_sizing_outputs\02_ramp_statistics_before_bess.csv
bess_sizing_outputs\03_bess_sizing_ramp_scenarios.csv
bess_sizing_outputs\04_top_feasible_candidates.csv
bess_sizing_outputs\05_best_timeseries_R20.csv
bess_sizing_outputs\05_best_timeseries_R10.csv
bess_sizing_outputs\05_best_timeseries_R5.csv
bess_sizing_outputs\05_best_timeseries_R3.csv
```

### 3.4 Analisis sensitivitas ekonomi R5

```bat
python scripts\04_economic_sensitivity_r5.py
```

Output utama:

```text
economic_sensitivity_outputs\06_economic_sensitivity_R5.csv
```

Catatan: skrip ini tidak melakukan optimasi ulang. Skrip membaca kapasitas optimum R5 dari hasil grid search, lalu menghitung dampak perubahan CAPEX baterai dan discount rate.

### 3.5 Menyiapkan input DIgSILENT

```bat
python scripts\05_prepare_digsilent_inputs.py
```

Output utama:

```text
digsilent_time_characteristics\07_DIgSILENT_PV_time_characteristic_R5_*.csv
digsilent_time_characteristics\08_DIgSILENT_BESS_time_characteristic_R5_*.csv
digsilent_time_characteristics\09_DIgSILENT_combined_R5_*.csv
digsilent_time_characteristics\digsilent_selected_day_metadata.json
```

### 3.6 Menghitung indikator lingkungan indikatif

```bat
python scripts\06_environmental_indicator.py
```

Output utama:

```text
environmental_outputs\10_environmental_indicator.csv
environmental_outputs\10_environmental_indicator.xlsx
```

### 3.7 Pembanding PSO dengan Deb's Feasibility Rules

```bat
python scripts\07_optimize_bess_pso_deb_rules.py --input soda_final_outputs\synthetic_pv_soda_2020_FINAL.csv --scenario ALL --seeds 1,2,3,4,5,6,7,8,9,10 --particles 30 --iters 60 --grid-summary bess_sizing_outputs\03_bess_sizing_ramp_scenarios.csv --output pso_deb_multiseed_outputs
```

Output utama:

```text
pso_deb_multiseed_outputs\summary_all_runs.csv
pso_deb_multiseed_outputs\summary_by_scenario.csv
pso_deb_multiseed_outputs\best_run_by_scenario.csv
pso_deb_multiseed_outputs\pso_grid_deltas_all_runs.csv
pso_deb_multiseed_outputs\09_pso_deb_best_timeseries_R20.csv
pso_deb_multiseed_outputs\09_pso_deb_best_timeseries_R10.csv
pso_deb_multiseed_outputs\09_pso_deb_best_timeseries_R5.csv
pso_deb_multiseed_outputs\09_pso_deb_best_timeseries_R3.csv
```

### 3.8 Membuat master summary untuk Bab 4

```bat
python scripts\08_master_summary_for_bab4.py
```

Output utama:

```text
master_summary_outputs\00_master_results_for_bab4.xlsx
master_summary_outputs\00_key_metrics_for_bab4.csv
master_summary_outputs\00_file_availability.csv
```

## 4. Catatan konsistensi dengan buku tesis

Pipeline utama menggunakan parameter final berikut:

```text
PLTS                : 100 MWac
Lokasi              : Mekarwaru/Subang, sekitar -6.58558, 107.89692
Tahun profil         : 2020, non-kabisat, 365 hari
Resolusi profil      : 1 menit
Input NSRDB/Himawari : 30 menit
Rasio DC/AC          : 1.10
Tilt                 : 10 derajat
Azimuth              : 0 derajat
Efisiensi inverter   : 96%
Losses               : 14%
Seed SoDa            : 42
Ramp scenarios       : R20, R10, R5, R3
SOC                  : 20% - 90%
SOC target           : 55%
C-rate maksimum      : 1C
Durasi minimum       : 60 menit
Minimum daya BESS    : 10 MW, yaitu 10% dari kapasitas PLTS
Fixed investment cost komponen daya  : 427,420 USD/MW
Fixed investment cost komponen energi : 307,990 USD/MWh
Fixed O&M            : 2.07%/tahun
Discount rate        : 8%
Project life         : 20 tahun
PSO final            : Deb's Feasibility Rules, multi-seed, 4 desimal
```

## 5. File di archive

Folder `archive/` berisi skrip lama/debug/backup. File ini tidak digunakan sebagai pipeline utama karena sebagian masih memakai penalty objective PSO lama atau hanya dipakai untuk debugging.

Pipeline final menggunakan `scripts/07_optimize_bess_pso_deb_rules.py` sebagai pembanding PSO, bukan file PSO lama di `archive/`.
