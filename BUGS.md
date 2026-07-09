# Bug Tracker — `tsv-dev-debug`

Lightweight bug log for this debugging branch (fork: `vilasinits/GODMAX`).
Identify bugs here, fix them, push to this branch. Each fix references its commit
and is also recorded in [CHANGELOG.md](CHANGELOG.md).

**Status legend:** `OPEN` · `IN-PROGRESS` · `FIXED` · `WONTFIX`

Each bug: id, title, category (`bug`/`enhancement`), where it lives, symptom,
root cause, fix commit.

---

## Open

_(none yet — add as identified)_

---

## Fixed

### B8 — FFTLog grid can under-cover the halo profiles (M3) — `enhancement` — FIXED (warning added)

- **Where:** `src/get_radial_profiles.py` `setup_main_calc`
- **Issue:** the FFTLog grid reaches only `rmax = r_array[-1]`, but profiles extend to
  `rt = epsilon_rt·r200c` (and gas to `~theta_ej·r200c`). For `params_default.yaml`
  (`rmax=8`, `lg10_Mmax=15.5`), `rt_max ≈ 12.4 Mpc > 8`, so heavy haloes are truncated on
  the grid → `_profile_grid_mass < Mtot` and `u(k)` shape distorted for them. Not a logic
  bug — silent param under-coverage.
- **Fix:** emit a `RuntimeWarning` at init if `rmax < max(rt, r_ej)`, reporting the needed
  `rmax`. Does not change results. (Physical inner/outer r200c-relative bounds M1 `0.01·r200c`
  and M4 `6·r200c` are by-design and left as-is.)

### B9 — enclosed-mass inner floor `minr` can coincide with `rmin` (M2, root of B1) — `bug` — FIXED

- **Where:** `src/get_radial_profiles.py` (all `minr = ...` enclosed-mass integrals)
- **Symptom:** for param sets with `rmin = 5e-4` (e.g. Pge) or massive haloes where
  `0.005·r200c ≥ 5e-4`, `minr` equalled `rmin`, so `get_Mnfw(r_array[0])` integrated a
  zero-width `[rmin, rmin]` interval → 0 → the `jr=0` zeros that B1 clips.
- **Fix:** `minr = min(min(5e-4, 0.5·rmin), 0.005·r200c)`, guaranteeing `minr ≤ 0.5·rmin < rmin`
  while preserving the `5e-4` floor for `rmin=0.005` (default unchanged). Removes the
  zero-width integral at the source; B1's clip is now defensive/redundant.

### B4 — `num_points_gal_cal` param silently ignored (wrong key) — `bug` — FIXED

- **Where:** `src/base_class.py` (grid/param setup)
- **Symptom:** the param-file value `num_points_gal_cal` had no effect; the attribute
  always mirrored `num_points_trapz_int`.
- **Root cause:** `self.num_points_gal_cal = analysis_dict.get('num_points_trapz_int', 32)`
  read the wrong dict key. Masked in defaults (both 32), diverges if a user sets them apart.
- **Fix:** read the dedicated `'num_points_gal_cal'` key.

### B5 — `nz_source_info_dict` / `nz_lens_info_dict` grid shorthand not parsed — `bug` — FIXED

- **Where:** `src/base_class.py` (source & lens n(z) setup)
- **Symptom:** `z_array_source`/`z_array_lens` given as `[start, stop, n]` were read
  literally as a 3-element array (e.g. `[0.01, 1.5, 128]` → three redshifts incl. z=128),
  and a scalar `nz<jb>` mismatched the grid length. It "worked" only via an `except`
  fallback to a hardcoded `linspace(0.01, 1.5, 128)`, inconsistent with the
  `nbar_gal_comoving_zarray` parsing which does support `[start, stop, n]`.
- **Fix:** added `_build_grid_1d` (`[start,stop,n]`→linspace, else explicit array) and
  `_build_pzs` (full-length array used as-is, scalar broadcast to a flat p(z)); rewired
  both source and lens blocks. Default params now parse by design, not by exception.

### B7 — `get_rho_clm` loses ~15% of CLM mass (root cause of large-scale Pgg ratio ≠ 1) — `bug` — FIXED (verified: large-scale Pgg ratio → 1, DMB/halofit unaffected)

- **Where:** `src/get_radial_profiles.py` `get_rho_clm`
- **Symptom:** `Pgg` (and `bg`/`P2h`) backreaction/no-backreaction ratio ≠ 1 at large
  scales (`max|bg-1|`≈0.069, `max|P2h-1|`≈0.13). Not fixable in `get_Pkzs` (B2/B6) because
  the mass is lost *before* FFTLog.
- **Root cause:** `rho_clm` was `jax.grad` of a piecewise-linear `jnp.interp` of
  `log M_clm(log r)`. In the saturated outer region (flat `M_clm` for the truncated NFW)
  the local slope went ≤0 and clipped to 0, collapsing the tail and losing ~15% of the
  enclosed mass. The backreaction run (uses `get_rho_clm`) then had `u_clm(k→0)≈0.85`,
  while the no-backreaction run (analytic `fclm·rho_nfw`) had `≈0.98` — different plateaus,
  so `bg` was not backreaction-independent.
- **Fix:** compute the log-derivative with a central difference `jnp.gradient(ln_Mclm, ln_r)`
  (r_array is log-spaced) — smooth, JAX-differentiable, mass-conserving to <2%, so
  `u_clm(k→0)→~1` for both runs and the large-scale `Pgg` ratio returns to 1. Root fix for
  B2/B6 symptoms. Identified (but left commented) on the `test_baryonification` branch.

### B1 — `Mclm_mat` zeros in no-backreaction branch — `bug` — FIXED (`2bbedc1`)

- **Where:** `src/get_radial_profiles.py` (no-backreaction path, `run_clm_calc`)
- **Symptom:** `Mclm_n / Mclm_b` ratio returned `inf`; 528 zeros (whole inner
  `(nz, nM)` slice at `jr=0`).
- **Root cause:** `get_Mnfw` underflows to exactly `0.0` at the innermost radius;
  the no-backreaction branch didn't clip, unlike the backreaction path (`get_Mclm`
  floors at `1e-30`).
- **Fix:** clip no-backreaction `Mclm_mat` to `1e-30` to match the backreaction floor.

### B2 — Satellite Fourier profile `u(k→0)≠1` at large scales — `bug` — SUPERSEDED by B6

- **Where:** `src/get_Pkzs.py` (`uk_clm` / low-k interpolation)
- **Symptom:** `Pgg` (and `Pgm`/`Pgy`/`Pge`) backreaction/no-backreaction ratio
  ≠ 1 at large scales (~10% off); `bg` ratio @ k0 ≈ 0.951.
- **First attempt (`2bbedc1`, REVERTED):** renormalized `uk_clm` by its low-k value
  (`uk_clm/uk_clm[0]`). WRONG — this rescales the whole profile by a run-dependent factor
  and corrupts the 1-halo ratio (`max|P1h-1|` blew up 0.13 → 1.21). Reverted.
- **Correct root cause (see B6):** the leak is in `get_uk_from_interp_Pk`, which clamped
  `uk` at `k_mcfit[0]≈0.06` for all lower k, pinning the large-scale band to `uk≈0.85`
  instead of `→1`. Real fix is low-k extrapolation, not a global rescale.

### B6 — low-k `uk` extrapolation in `get_uk_from_interp_Pk` — REVERTED (superseded by B7)

- **Where:** `src/get_Pkzs.py` `get_uk_from_interp_Pk`
- **Why tried:** with the mass-losing `get_rho_clm`, the clamped interp pinned `uk` to
  `uk(k_mcfit[0])≈0.85` at large scales. Extrapolation toward 1 was an attempt to patch it.
- **Why reverted:** it addressed a symptom of the B7 mass loss, not the cause. Once B7
  conserves the CLM mass, `uk(k_mcfit[0])≈1` for all profiles and the plain clamp is correct.
  Worse, the extrapolation read its slope from the first two FFTLog points, so after B7
  reshaped `rho_dmb`'s tail it drifted `uk_dmb(k→0)` off 1 and broke the DMB/halofit matter
  ratio (which is pinned to 1 only when `uk_dmb(k→0)=1`). Reverted to plain log-space interp.
