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

### B1 — `Mclm_mat` zeros in no-backreaction branch — `bug` — FIXED (`2bbedc1`)

- **Where:** `src/get_radial_profiles.py` (no-backreaction path, `run_clm_calc`)
- **Symptom:** `Mclm_n / Mclm_b` ratio returned `inf`; 528 zeros (whole inner
  `(nz, nM)` slice at `jr=0`).
- **Root cause:** `get_Mnfw` underflows to exactly `0.0` at the innermost radius;
  the no-backreaction branch didn't clip, unlike the backreaction path (`get_Mclm`
  floors at `1e-30`).
- **Fix:** clip no-backreaction `Mclm_mat` to `1e-30` to match the backreaction floor.

### B2 — Satellite Fourier profile not mass-normalized under backreaction — `bug` — FIXED (`2bbedc1`)

- **Where:** `src/get_Pkzs.py` (`uk_clm` construction)
- **Symptom:** `Pgg` (and `Pgm`/`Pgy`/`Pge`) backreaction/no-backreaction ratio
  ≠ 1 at large scales (~10% off); `bg` ratio @ k0 ≈ 0.951.
- **Root cause:** backreaction `rho_clm` (clipped numerical `dMclm/dr`) loses ~15%
  of its mass at `k→0` (`uk_clm[k0]` ≈ 0.84–0.90 vs ≈ 1.0 analytic), so the
  large-scale galaxy bias was not backreaction-independent as it must be.
- **Fix:** renormalize satellite Fourier profile so `u_sat(k_min)=1` by construction.
