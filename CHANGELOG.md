# Changelog

All notable fixes and changes on the `tsv-dev-debug` branch are documented here,
one entry per commit. Newest first.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
Each entry lists the commit hash, date, and every fix in that commit.

## [Unreleased]

### (this commit) — 2026-07-09 — r-grid consistency: coverage warning (M3) and minr floor (M2)

- **`src/get_radial_profiles.py`** — B8/M3: warn at init when the FFTLog grid under-covers
  the profiles (`rmax < max(epsilon_rt·r200c, theta_ej·r200c)`), which truncates massive
  haloes on the grid (`_profile_grid_mass < Mtot`). Diagnostic only, no result change.
- **`src/get_radial_profiles.py`** — B9/M2: set the enclosed-mass inner floor to
  `min(min(5e-4, 0.5·rmin), 0.005·r200c)` so `minr < rmin` strictly. Previously `minr`
  could equal `rmin` (Pge / massive haloes) → zero-width integral at `jr=0` (the zeros B1
  clips). Default params (`rmin=0.005`) unchanged; removes the zero-width at the source.
- M1 (`0.01·r200c`) and M4 (`6·r200c`) left as-is: physical per-halo bounds, by design.

### ac6bcc6 — 2026-07-09 — Fix CLM-mass conservation (large-scale Pgg) and param-file grid reads

- **`src/get_radial_profiles.py`** — B7: compute `rho_clm` via a central-difference
  log-derivative `jnp.gradient(ln_Mclm, ln_r)` instead of `jax.grad` of a piecewise-linear
  `jnp.interp`. The old form zeroed the saturated outer tail of the truncated NFW and lost
  ~15% of the CLM mass in the backreaction run only, so `u_clm(k→0)≈0.85` there vs `≈0.98`
  in the no-backreaction run — breaking the large-scale `Pgg` (bary/no-bary) ratio. The
  central difference conserves mass to <2%, restoring the large-scale `Pgg` ratio to 1
  without disturbing the DMB/halofit matter ratio. Verified against both.
- **`src/get_Pkzs.py`** — reverted two earlier symptom-patches now subsumed by B7: the
  `uk_clm/uk_clm[0]` renorm (B2, corrupted the 1-halo ratio) and the low-k `uk`
  extrapolation (B6, drifted `uk_dmb(k→0)` off 1 and broke DMB/halofit). `get_uk_from_interp_Pk`
  is back to the plain log-space interp.
- **`src/base_class.py`** — B4: read the dedicated `num_points_gal_cal` key instead of
  mis-keying to `num_points_trapz_int` (param value was silently ignored).
- **`src/base_class.py`** — B5: parse `z_array_source`/`z_array_lens` `[start, stop, n]`
  shorthand consistently with `nbar_gal_comoving_zarray`, via new `_build_grid_1d` /
  `_build_pzs` helpers; scalar `nz<jb>` now broadcasts to a flat p(z). Previously only
  worked via an `except` fallback to a hardcoded grid.

### 019d13c — 2026-07-09 — Add BUGS.md bug tracker

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
