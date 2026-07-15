# Changelog

All notable fixes and changes on the `tsv-dev-debug` branch are documented here,
one entry per commit. Newest first.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
Each entry lists the commit hash, date, and every fix in that commit.

## [Unreleased]

### (working tree, uncommitted) — 2026-07-15 — tSZ toggle large-scale yy conservation (k-space rematch) + fstar_sat log(10) factor

- **`src/godmax/get_Pkzs.py`** — fix the `baryonification_tSZ` toggle so the gravity-only NFW
  baseline conserves the large-scale (k→0) tSZ auto power. The old path Y3D-matched the NFW
  baseline in **real space** (`run_pressure_calc_nfw`, k=0 monopole), assuming
  `uk_y_nfw(k→0) == uk_y(k→0)`. But the pipeline never reaches k=0: the lowest FFTLog bin is
  `k_mcfit[0] ≈ 0.046 h/Mpc` (for `rmax=16`) and `get_uk_from_interp_Pk` clamps `uk_y` flat
  below it, while large-scale yy (ℓ~10) probes k~0.008 ≪ k_mcfit[0]. Equal real-space Y3D does
  NOT give equal `uk_y` at k_mcfit[0] (finite-k Bessel weighting differs for extended-baryonified
  vs concentrated-NFW shapes) — a toy pair with identical Y3D gave `uk` ratio 1.16 → Pyy 1.35.
  Now rematch the NFW baseline in **k-space** at k_mcfit[0]: transform both `y3d_mat` and
  `y3d_nfw_mat`, scale the NFW `uk_y` by `uk_y_bary[0]/uk_y_nfw[0]` per (z, M). A constant
  real-space rescale scales `uk_y` uniformly at all k, so matching the lowest bin forces
  `uk_y_nfw == uk_y_bary` on the clamped floor while leaving the small-scale baryon signal
  untouched. Old real-space selection kept commented out with rationale.
  - Verified: 3D `Pyy` ratio (bary/nfw) is now `1.0000` for k ≤ 0.046 and deviates only at
    k ≳ 0.1 (0.71 at k=0.88). Note: the projected `Cl_yy` large-scale ratio does NOT reach 1
    (the 1/χ² Limber weight upweights low z, where even ℓ=10 samples k above the matched
    floor) — intrinsic to the yy projection, not a residual bug.
- **`src/godmax/get_radial_profiles.py`** — `run_pressure_calc_nfw`: note added that the Y3D
  rescale is superseded for the observable large-scale limit by the k-space rematch in
  `get_Pkzs`; it now only sets a sensible absolute amplitude for `y3d_nfw_mat`.
- **`src/godmax/get_radial_profiles.py`** — `get_fstar_sat`: multiply the satellite stellar-mass
  integral `val2` by `ln(10)` so the `d ln M` trapezoid carries the correct log-base factor.

### (working tree, uncommitted) — 2026-07-13 — Restructure into a src-layout `godmax` package

- **Repo layout** — move the flat `src/*.py` modules and the `helpers` / `mcfitjax`
  subpackages into a single importable package `src/godmax/` (src-layout). Add
  `__init__.py` for the package and both subpackages; the top-level `godmax` namespace
  re-exports the core chain (`Profiles`, `get_Pkz`, `get_Cl`, `get_cov`, `get_xi`).
- **Imports** — rewrite all intra-package imports to `godmax.`-absolute
  (`from get_Cls import ...` → `from godmax.get_Cls import ...`, `import helpers.constants`
  → `import godmax.helpers.constants`, `from mcfitjax.transforms import ...` →
  `from godmax.mcfitjax.transforms import ...`).
- **`pyproject.toml`** — adopt the `extract-yy` build config: setuptools src-layout
  (`package-dir "" = "src"`, `packages.find where = ["src"]`), pinned scientific deps
  (numpy/scipy/jax/…), Python ≥ 3.9. `pip install -e .` now exposes `import godmax`.
- **`archive/`** — relocate the old scratch code (`src/arxiv`) and notes
  (`src/context`) out of the package tree so `src/` contains only `godmax`.
- Note: `notebooks/` and `run_scripts/` still use flat `sys.path` imports and need
  updating to `from godmax.… import …` separately.

### (committed `bbae9dc`) — 2026-07-13 — tSZ baryonification toggle + Pyy/uy refactor

New `baryonification_tSZ` switch that turns the tSZ (pressure) sector between the full
DMB-HSE pressure and a gravity-only NFW baseline, plus a refactor that moves the tSZ auto
spectrum and l-space y-profile out of `get_covs` so they are available on plain `get_Cl`
instances.

- **`src/base_class.py`** — add `self.baryonification_tSZ` (default `True`). Accepts either
  spelling (`baryonification_tSZ` / `baryonification_tsz`) so a casing typo cannot silently
  leave the tSZ baryonification on.
- **`src/get_radial_profiles.py`** — add `run_pressure_calc_nfw` / `get_Ptot_nfw`: gravity-only
  HSE pressure with NFW on both legs (`M_nfw` gravity, `(Ob0/Om0)·rho_nfw` gas, `R_nt=0` →
  fully thermal), rescaled per `(z, M)` so the grid-integrated `Y3D = ∫4πr²y3d dr` over
  `r_array` matches the baryonified `y3d_mat`. This conserves Compton-y per halo, so
  `uk_y_nfw(k→0)=uk_y(k→0)` and the large-scale y-power ratio → 1; the toggle then isolates the
  profile-**shape** effect of baryons. `Mnfw_mat` is now precomputed on `r_array` (only when the
  toggle is off), `y3d_const_coeff` is cached, and `run_pressure_calc_nfw` runs after
  `run_pressure_calc` when `baryonification_tSZ` is False.
- **`src/get_radial_profiles.py`** — B10: extend the FFTLog coverage check (B8) to require the
  grid to reach `6·r200c`, since the tSZ pressure is integrated out to `6·r200c`; otherwise
  `Y3D` is silently truncated for the toggle's Y3D match. Diagnostic only.
- **`src/get_Pkzs.py`** — route the toggle through the y sector: `uk_y` is built from
  `y3d_mat` or `y3d_nfw_mat`; the `Pym` matter leg follows it (gravity-only pressure pairs with
  the NFW matter leg `bm_nfw` / 1-halo probe 1) so `Pym` stays a consistent cross-spectrum. The
  tSZ auto power (`Pyy_1h` / `Pyy_2h` / `Pyy_tot_kz_mat`) is now computed here (moved from
  `get_covs`) as the plain `1h+2h` sum, reproducing the prior covariance behaviour exactly.
- **`src/get_Cls.py`** — move the l-space y machinery here (`get_uyl`, `get_uyl_interp`,
  `get_uy_l_forcov`, and the `uyl_mat_tointp` / `uyl_mat` / `uy_l_for_cov` arrays) and compute the
  tSZ auto Cl here (`get_Pkyy_lz`, `get_Cl_y_y_tot`, `Cl_y_y_tot_mat`), so `Cl_y_y_tot_mat` is
  available on any `get_Cl` instance alongside the other Cls.
- **`src/get_covs.py`** — delete the now-duplicated Pyy/uy definitions; `get_cov` inherits
  `Pyy_tot_kz_mat`, `Pkyy_lz_mat`, `logPkyylz_2d_interp`, `Cl_y_y_tot_mat`, `uyl_mat_tointp`,
  `uyl_mat`, and `uy_l_for_cov` unchanged. No result change to the covariance.

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
