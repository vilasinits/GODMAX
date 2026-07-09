# Changelog

All notable fixes and changes on the `tsv-dev-debug` branch are documented here,
one entry per commit. Newest first.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
Each entry lists the commit hash, date, and every fix in that commit.

## [Unreleased]

### (this commit) — 2026-07-09 — Add BUGS.md bug tracker

- **`BUGS.md`** — New local bug tracker for this branch (fork issues are disabled).
  Seeded with the two fixed bugs B1 (`Mclm_mat` zeros) and B2 (satellite profile
  normalization), both fixed in `2bbedc1`.

### 9ee5fc8 — 2026-07-09 — Add CHANGELOG documenting fixes on tsv-dev-debug

- **`CHANGELOG.md`** — New file. Introduces a per-commit changelog for this branch.

### 2bbedc1 — 2026-07-09 — Fix backreaction inconsistency in CLM mass and satellite profile

- **`src/get_radial_profiles.py`** — Clip the no-backreaction `Mclm_mat` to `1e-30`
  to match the floor applied in the backreaction path (`get_Mclm`). `get_Mnfw`
  underflows to exactly `0.0` at the innermost radius (`jr=0`), which zeroed the
  whole inner `(nz, nM)` slice and produced `inf`/`NaN` in downstream ratios.
- **`src/get_Pkzs.py`** — Renormalize the satellite Fourier profile so
  `u_sat(k_min) = 1` by construction. The backreaction `rho_clm` (a clipped
  numerical `dMclm/dr`) loses ~15% of its mass at `k -> 0` compared to the analytic
  no-backreaction profile, breaking the requirement that the large-scale galaxy bias
  be backreaction-independent. Renormalizing forces the `Pgg`/`Pgm`/`Pgy`/`Pge`
  ratios to `-> 1` at large scales.
- **`param_files/Pge/params_v2.yaml`**, **`notebooks/test/test_bary.ipynb`** —
  Supporting parameter and notebook updates for the above debugging.
