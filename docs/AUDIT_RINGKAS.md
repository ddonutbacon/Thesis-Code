# Audit Ringkas Skrip

Skrip utama disusun dari file lokal terbaru yang diunggah pada sesi finalisasi.

Keputusan utama:

- `01_bess_sizing_ramp_scenarios.py` dipakai sebagai grid search final karena sudah memakai `DURATION_MIN = 60.0`, parameter ekonomi final, dan fine search 0.01 MW/MWh.
- `05c_bess_pso_deb_rules_multiseed_4dp.py` dipakai sebagai PSO final karena sesuai narasi buku: Deb's Feasibility Rules, multi-seed, 4 desimal, tanpa penalty besar.
- `solar_data.py` disanitasi: API key hardcoded diganti pembacaan environment variable `NREL_API_KEY`.
- Skrip PSO penalty lama, debug continuous, wrapper multiseed debug, dan re-optimization sensitivity dipindahkan ke `archive/`.

File `archive/` tidak perlu dijalankan untuk mereproduksi hasil utama tesis.
