"""CMB lensing, always computed alongside shear (probe 5 in get_Cl_tot, field 'c' in get_covs).

Checks:
  * null control: the CMB settings (z_source_cmb, nz_cmb_lensing_tail) leave every non-CMB output
    bit-identical;
  * the CMB kernel equals the independent numpy kernel used by the map code;
  * shear x CMB-kappa and the high-z halofit tail of the CMB auto equal independent numpy Limber
    integrals of the stored kernels and P(k,z);
  * the tail is converged in its redshift sampling;
  * the CMB covariance blocks follow the Gaussian Knox formula;
  * the removed is_cmb_lensing switch fails loudly.

Run:
    uv run --no-sync --with pytest python -m pytest -q tests/physics/test_cmb_lensing_joint.py
"""
import numpy as np
import pytest

from godmax.get_Cls import get_Cl
from godmax.get_covs import get_cov
from godmax.get_sim_maps import compute_cmb_lensing_efficiency

NON_CMB = ["Cl_kappa_kappa_tot_mat", "Cl_kappa_y_tot_mat", "Cl_gal_gal_tot_mat", "Cl_gal_kappa_tot_mat",
           "Cl_gal_y_tot_mat", "Cl_gal_tau_tot_mat", "Cl_y_y_signal_mat", "Wk_mat"]
# Same arithmetic, different code path: agreement is at round-off level (measured ~4e-16).
RTOL_SAME_INTEGRAL = 1e-12


def _a(x):
    return np.asarray(x, dtype=np.float64)


def test_cmb_settings_leave_other_outputs_bit_identical(model):
    ref = model("on", cls=get_Cl)
    for over in ({"z_source_cmb": 1089.0}, {"nz_cmb_lensing_tail": 32}):
        alt = model("on", cls=get_Cl, ana_over=over)
        assert [n for n in NON_CMB if not np.array_equal(_a(getattr(ref, n)), _a(getattr(alt, n)))] == []
    # ... while the CMB auto does respond to the source redshift
    alt = model("on", cls=get_Cl, ana_over={"z_source_cmb": 1089.0})
    assert not np.array_equal(_a(ref.Cl_kcmb_kcmb_tot_mat), _a(alt.Cl_kcmb_kcmb_tot_mat))


def test_cmb_kernel_matches_map_kernel(model):
    m = model("on", cls=get_Cl)
    for z, chi, W in ((m.z_array_for_Cls, m.chi_array_for_Cls, m.Wkcmb_array),
                      (m.z_array_cmb_tail, m.chi_array_cmb_tail, m.Wkcmb_tail_array)):
        ref = compute_cmb_lensing_efficiency(_a(z), _a(chi), _a(m.chi_CMB), float(m.cosmo_jax.Omega_m))
        np.testing.assert_allclose(_a(W), ref, rtol=RTOL_SAME_INTEGRAL, atol=1e-30)
    assert _a(m.Wkcmb_tail_array)[-1] == 0.0                      # kernel vanishes at the source
    assert float(m.z_array_cmb_tail[0]) == pytest.approx(float(m.z_array_for_Cls[-1]), rel=1e-12)


def test_shear_cross_cmb_independent_limber(model):
    m = model("on", cls=get_Cl)
    chi, dchi, z = _a(m.chi_array_for_Cls), _a(m.dchi_dz_array_for_Cls), _a(m.z_array_for_Cls)
    mb = _a(m.mult_shear_bias_array)[0]
    fx = (1 + mb) * (_a(m.Wk_mat[0]) / chi**2) * (_a(m.Wkcmb_array) / chi**2) * chi**2 * dchi \
        * _a(m.cached_power_spectra[0, 0])
    np.testing.assert_allclose(_a(m.Cl_kappa_kcmb_tot_mat[:, 0]), np.trapezoid(fx, x=z, axis=-1),
                               rtol=RTOL_SAME_INTEGRAL, atol=0)
    # Same P(k,z) for both fields, so Cauchy-Schwarz holds exactly for the low-z Limber integrand;
    # the tail only adds to the CMB auto, which keeps r below 1.
    r = _a(m.Cl_kappa_kcmb_tot_mat[:, 0]) / np.sqrt(_a(m.Cl_kappa_kappa_tot_mat[:, 0, 0]) * _a(m.Cl_kcmb_kcmb_tot_mat))
    assert np.all(r > 0) and np.all(r <= 1.0)


def test_cmb_auto_tail_independent_limber(model):
    m = model("on", cls=get_Cl)
    z, chi, dchi = _a(m.z_array_cmb_tail), _a(m.chi_array_cmb_tail), _a(m.dchi_dz_array_cmb_tail)
    W = compute_cmb_lensing_efficiency(z, chi, _a(m.chi_CMB), float(m.cosmo_jax.Omega_m))
    logk, logP = np.log(_a(m.kPk_array)), np.log(_a(m.phfit_kz_cmb_tail_mat))
    cl = []
    for ell in _a(m.ell_array):
        k_ell = (ell + 0.5) / np.clip(chi, 1.0, None)
        P = np.exp([np.interp(np.log(k_ell[i]), logk, logP[:, i]) for i in range(z.size)])
        cl.append(np.trapezoid(W**2 / chi**2 * dchi * P, x=z))
    np.testing.assert_allclose(_a(m.Cl_kcmb_kcmb_tail_mat), np.array(cl), rtol=1e-10, atol=0)
    # total = halo-model part on z_array_for_Cls + tail
    np.testing.assert_allclose(_a(m.Cl_kcmb_kcmb_tot_mat) - _a(m.Cl_kcmb_kcmb_tail_mat),
                               _a(m.get_Cl_tot(0, 0, 5, 5)), rtol=RTOL_SAME_INTEGRAL, atol=0)


def test_cmb_auto_tail_converged(model):
    # Measured (reduced grid, default fiducial): 64 vs 256 nodes 2.1e-3, 128 vs 256 nodes 4.1e-4.
    ref = model("on", cls=get_Cl, ana_over={"nz_cmb_lensing_tail": 512})
    m = model("on", cls=get_Cl)                                    # default 128 nodes
    dev = np.max(np.abs(_a(m.Cl_kcmb_kcmb_tot_mat) / _a(ref.Cl_kcmb_kcmb_tot_mat) - 1))
    assert dev < 1e-3, f"{dev:.3g}"


def test_two_source_bins_shapes(model):
    zs = np.linspace(0.01, 1.5, 128)
    nz = {"nbins": 2, "z_array_source": zs, "nz0": ((zs > 0.2) & (zs < 0.6)).astype(float),
          "nz1": ((zs > 0.6) & (zs < 1.2)).astype(float)}
    m = model("on", cls=get_Cl, ana_over={"nz_source_info_dict": nz})
    nell = int(m.ell_array.size)
    assert _a(m.Cl_kappa_kappa_tot_mat).shape == (nell, 2, 2)
    assert _a(m.Cl_kappa_kcmb_tot_mat).shape == (nell, 2)
    assert _a(m.Cl_kcmb_kcmb_tot_mat).shape == (nell,)


def test_is_cmb_lensing_removed(model):
    with pytest.raises(ValueError, match="is_cmb_lensing"):
        model("on", cls=get_Cl, ana_over={"is_cmb_lensing": True})


def test_is_cmb_lensing_attribute_on_seed_raises(model):
    """dr1_cmbx's TheoryGodmax sets is_cmb_lensing on a get_Pkz object and builds get_Cl from it."""
    import copy
    from godmax.get_Pkzs import get_Pkz
    seed = copy.copy(model("on", cls=get_Pkz))
    seed.is_cmb_lensing = True
    with pytest.raises(ValueError, match="is_cmb_lensing"):
        get_Cl(None, None, None, None, Pkz_obj=seed)


def test_cmb_covariance_gaussian_blocks(model):
    ana = dict(stats_for_cov=["kk", "kc", "cc"], dl_array_survey=np.ones(12), sigma_epsilon_SN_bins=[0.26],
               neff_arcmin2_SN_bins=[5.0])
    c = model("on", cls=get_cov, ana_over=ana)
    ell, fsky = _a(c.ell_array), 0.1
    Nk = 0.26**2 / (5.0 * (180 * 60 / np.pi) ** 2)
    Ckk = _a(c.Cl_kappa_kappa_tot_mat[:, 0, 0]) + Nk
    Ccc, Ckc = _a(c.Cl_kcmb_kcmb_tot_mat), _a(c.Cl_kappa_kcmb_tot_mat[:, 0])
    pref = 1.0 / (fsky * (2 * ell + 1))
    expected = {("cc_cc", "bin_0_0_0_0"): 2 * Ccc**2,
                ("kc_kc", "bin_1_0_1_0"): Ckk * Ccc + Ckc**2,
                ("kc_cc", "bin_1_0_0_0"): 2 * Ckc * Ccc,
                ("kk_kc", "bin_1_1_1_0"): 2 * Ckk * Ckc}
    for (stat, bk), ref in expected.items():
        np.testing.assert_allclose(np.diag(_a(c.covG_dict[stat][bk])), pref * ref, rtol=RTOL_SAME_INTEGRAL, atol=0)
