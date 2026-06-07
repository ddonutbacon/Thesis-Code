# Thesis BESS-SoDa Workflow

Repository ini berisi skrip Python yang digunakan untuk tesis:

**Evaluasi dan Optimasi Kapasitas Battery Energy Storage System untuk Pengendalian Ramp-rate PLTS 100 MW Berbasis Profil Daya Sintetik SoDa**

Repositori ini disusun sebagai paket kode yang dapat diaudit ulang. Alur utama mengikuti buku tesis final: pembangkitan profil daya PV sintetik SoDa, pemeriksaan kewajaran terhadap NASA POWER, optimasi kapasitas BESS dengan deterministic grid search, analisis sensitivitas ekonomi, indikator lingkungan, persiapan input DIgSILENT, dan pembanding PSO berbasis Deb's Feasibility Rules.

## Struktur

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

## Skrip utama

1. `01_generate_soda_profile.py`  
   Membuat profil daya PV sintetik SoDa tahun 2020 resolusi 1 menit.

2. `02_check_soda_nasa_consistency.py`  
   Memeriksa kewajaran energi bulanan SoDa terhadap NASA POWER.

3. `03_optimize_bess_grid_search.py`  
   Skrip utama optimasi kapasitas BESS menggunakan deterministic grid search untuk R20, R10, R5, dan R3.

4. `04_economic_sensitivity_r5.py`  
   Analisis sensitivitas ekonomi R5 tanpa re-optimasi kapasitas.

5. `05_prepare_digsilent_inputs.py`  
   Menyiapkan time characteristic PV dan BESS untuk DIgSILENT berdasarkan hasil R5.

6. `06_environmental_indicator.py`  
   Menghitung indikator CO2 ekuivalen indikatif berdasarkan annual discharge BESS.

7. `07_optimize_bess_pso_deb_rules.py`  
   Pembanding PSO final berbasis Deb's Feasibility Rules, multi-seed, dan resolusi kandidat 4 desimal.

8. `08_master_summary_for_bab4.py`  
   Helper untuk mengumpulkan output utama menjadi ringkasan Bab 4.

## Catatan keamanan

API key NSRDB/NREL tidak disimpan di kode. Sebelum menjalankan generator SoDa, set environment variable:

```bat
set NREL_API_KEY=ISI_API_KEY_ANDA_DI_SINI
```

Pada PowerShell:

```powershell
$env:NREL_API_KEY="ISI_API_KEY_ANDA_DI_SINI"
```

## Batasan

- Profil SoDa adalah profil sintetik untuk studi pra-kelayakan, bukan data pengukuran aktual lapangan.
- NASA POWER digunakan sebagai referensi pola klimatologi bulanan, bukan ground truth daya PV resolusi satu menit.
- BESS digunakan hanya untuk ramp-rate smoothing, bukan energy shifting, arbitrase, frequency regulation, reserve, atau peak shaving.
- Indikator CO2eq bersifat indikatif, bukan klaim pengurangan emisi aktual atau carbon credit.
- File lama/debug disimpan di `archive/` hanya untuk rekam jejak, bukan untuk pipeline utama.
