# Bug Tracker — `tsv-dev-debug`

Lightweight bug log for this debugging branch (fork: `vilasinits/GODMAX`).
Identify bugs here, fix them, push to this branch. Each fix references its commit
and is also recorded in [CHANGELOG.md](CHANGELOG.md).

**Status legend:** `OPEN` · `IN-PROGRESS` · `FIXED` · `WONTFIX`

Each bug: id, title, category (`bug`/`enhancement`), where it lives, symptom,
root cause, fix commit.

---

## Open

### B13 — `halofit_parameters` returns `n_eff`/`C` for only the first scale factor — `bug` — OPEN (latent)

- **Where:** `src/godmax/helpers/jax_cosmo_power.py` (`halofit_parameters`, return statement)
- **Symptom:** called with an array of `a` of length > 1, the function either raises
  `TypeError: Cannot concatenate arrays with different numbers of dimensions: got (1, N), (1,), (1,)`
  or, where shapes happen to broadcast, silently returns single-redshift `n_eff`/`C`
  alongside a full-length `k_nl`.
- **Root cause:** `k_nl` is `jax.vmap`-ed over `a` and so has shape `(na,)`, but the return
  statement is `np.array([k_nl, n_eff[0], C[0]])` — `n_eff` and `C` are computed for all `a`
  (shape `(na,)`) and then indexed to element 0.
- **Not affected:** the pipeline. `get_Pkzs.py` wraps the call in `vmap(..., (None, 0))`, so each
  invocation sees a scalar `a` (`na == 1`) and the `[0]` is a no-op. Only a direct caller passing
  an `a` array is bitten — which is the natural way to read the signature.
- **Fix:** not applied; out of scope of B12. Returning `np.stack([k_nl, n_eff, C])` would do it,
  but every call site assumes the current shape, so it needs changing together.

---

## Fixed

### B12 — halofit non-linear scale: 64-node linear-interp root find destroys the cosmology derivative — `bug` — FIXED (resolution raised + made configurable)

- **Where:** `src/godmax/helpers/jax_cosmo_power.py` (`halofit_parameters`), reached from
  `src/godmax/get_Pkzs.py` whenever `model_matter == 'halofit'`
- **Symptom:** halofit `C_ell` match a CCL reference to sub-percent in *amplitude*, but
  `d ln P_nl / d Omega_m` is a **staircase in z**: smooth to z≈0.15, then discrete jumps at
  z≈0.20, 0.70, 0.80, 1.05, of size O(0.1–0.2) — comparable to the CCL signal itself
  (~-0.15 at k = 1/Mpc). Central differences do not converge under step refinement (at z=0.30
  the value walks -0.085, -0.025, +0.095, +0.455, -0.126 as the step goes 2e-2 → 1e-3), while
  CCL returns -0.087 at every step.
- **Root cause:** the non-linear scale is located by *linear* interpolation to `sigma(R) = 1`
  on `num_points` log-spaced nodes (`root = interp(1.0, sigma, logr)`), with `num_points`
  hardcoded to **64** — upstream `jax_cosmo` uses 256. A linear interpolant makes the located
  root piecewise-linear in the node values, so `d R_nl / d(cosmology)` is piecewise-**constant**
  and jumps whenever the `sigma = 1` crossing slides across a node. The *value* converges fast
  (`k_nl` at 64 is within 0.9% of the 1024-node result, `max |k_nl - linear fit| / k_nl = 3.6e-5`)
  which is why amplitude checks pass; the *derivative* does not. At 64 nodes the derivative is
  wrong in **sign** at some redshifts — at z = 0.30: `dk_nl/dOm = -1.650` (converged -0.122),
  `dn_eff/dOm = -2.068` (converged +1.256), `dC/dOm = -5.258` (converged +0.545).
  `num_points` also sets the Simpson steps for `sigma`, `n_eff` and `C`.
- **Fix:** `num_points` is now read from `analysis_dict` as `num_points_halofit`
  (`base_class.read_all_input`) and passed through in `get_Pkzs`; the signature default is
  raised 64 → 512. Convergence is non-monotonic, as a staircase must be — at z = 0.30 `dC/dOm`
  goes 64 → -5.26, 128 → +0.502, 256 → +0.147, 512 → +0.548, 1024 → +0.545 — so upstream's 256
  is **not** sufficient and 512 is the smallest resolution tracking 1024 to ~1% at every
  pipeline redshift. Cost is ~51 ms per `P(k,z)` build (2 ms at 64, 18 ms at 256) against an
  ~11 s `C_ell` evaluation, i.e. ~0.5%.
- **Verified:** RMS of `d lnP_nl/dOm` minus CCL's, over k = 0.05–2 /Mpc and z = 0–2, drops
  **3.2950 → 0.0683** (typical |CCL `dlnP/dOm`| = 0.3184). The residual at 512 is a smooth
  ~-0.03 offset, i.e. the genuine jax_cosmo-Takahashi2012 vs CCL-halofit implementation
  difference, no longer a resolution artefact.
- **Downstream effect:** this was the cause of the wide, low-side-skewed `Omega_m` posterior in
  the dr1_cmbx GODMAX `ccl_mode` vs PyCCL comparison
  (`notebooks/test_integration/compare_theory_cls.ipynb`); the scan agrees with PyCCL after the
  fix. Note the earlier "unchanged by this fix" reading came from a kernel that had loaded a
  different GODMAX checkout and was running unpatched code — check `godmax.__file__` before
  concluding a source change had no effect.
- **Not a full cure:** piecewise-constant is piecewise-constant at any resolution — `dC/dOm` at
  z = 0 still shifts ~20% between 256/512/1024. z = 0 sits outside the pipeline grid
  (`zmin = 0.01`), so this does not bite here, but the real fix is the one the existing `TODO`
  asks for: a root solve carrying implicit-function derivatives
  (`d logR*/dtheta = -(dsigma/dtheta)/(dsigma/dlogR)`, e.g. `jax.lax.custom_root`), which would
  be exact at low `num_points` and cost less than 512 nodes.

### B10 — FFTLog coverage check ignored the tSZ pressure radius (`6·r200c`) — `enhancement` — FIXED (warning extended)

- **Where:** `src/get_radial_profiles.py` `setup_main_calc` (extends B8)
- **Issue:** the B8 coverage warning tested `rmax` only against the truncation/ejection radii
  (`rt`, `r_ej`). But the tSZ pressure (`get_Ptot` / `get_Ptot_nfw`) is integrated out to
  `6·r200c`, so `y3d` needs the grid to reach at least there. If `rmax < 6·r200c` the
  grid-integrated `Y3D = ∫4πr²y3d dr` is truncated — which silently biases the `baryonification_tSZ`
  Y3D match (`y3d_nfw` rescaled to a truncated `Y3D_bary`).
- **Fix:** include `rpress_max = 6·max(r200c)` in `r_needed` (only when `model_tSZ`). Diagnostic
  only; no result change.

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
