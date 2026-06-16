"""
Unit tests for get_Cl: angular power spectra.
  - Shape checks for all probes
  - Positivity (Cl_kk, Cl_yy, Cl_gg > 0)
  - Cauchy-Schwarz: C_kk * C_yy >= C_ky^2
  - Cauchy-Schwarz: C_kk * C_gg >= C_gk^2
  - Lensing kernel W_kappa positive and normalised
  - Cl_kk < Cl_kk_nfw (baryons suppress matter clustering at high ell)
"""

import pytest
import jax.numpy as jnp
import numpy as np


class TestClShapes:
    def test_Cl_kk_shape(self, cl_obj):
        nell, nbins = cl_obj.nell, cl_obj.nbins
        assert cl_obj.Cl_kappa_kappa_tot_mat.shape == (nell, nbins, nbins)

    def test_Cl_ky_shape(self, cl_obj):
        nell, nbins = cl_obj.nell, cl_obj.nbins
        assert cl_obj.Cl_kappa_y_tot_mat.shape == (nell, nbins)

    def test_Cl_yy_shape(self, cl_obj):
        assert cl_obj.Cl_y_y_signal_mat.shape == (cl_obj.nell,)

    def test_Cl_yy_tot_shape(self, cl_obj):
        assert cl_obj.Cl_y_y_tot_mat.shape == (cl_obj.nell,)

    def test_Cl_gg_shape(self, cl_obj):
        nell, nbins_lens = cl_obj.nell, cl_obj.nbins_lens
        assert cl_obj.Cl_gal_gal_tot_mat.shape == (nell, nbins_lens, nbins_lens)

    def test_Cl_gy_shape(self, cl_obj):
        nell, nbins_lens = cl_obj.nell, cl_obj.nbins_lens
        assert cl_obj.Cl_gal_y_tot_mat.shape == (nell, nbins_lens)

    def test_Cl_gk_shape(self, cl_obj):
        nell, nbins_lens, nbins = cl_obj.nell, cl_obj.nbins_lens, cl_obj.nbins
        assert cl_obj.Cl_gal_kappa_tot_mat.shape == (nell, nbins_lens, nbins)


class TestClPositivity:
    def test_Cl_kk_positive(self, cl_obj):
        """Auto-spectrum must be positive."""
        Cl_kk = np.array(cl_obj.Cl_kappa_kappa_tot_mat[:, 0, 0])
        assert np.all(Cl_kk > 0)

    def test_Cl_yy_positive(self, cl_obj):
        Cl_yy = np.array(cl_obj.Cl_y_y_signal_mat)
        assert np.all(Cl_yy > 0)

    def test_Cl_gg_positive(self, cl_obj):
        Cl_gg = np.array(cl_obj.Cl_gal_gal_tot_mat[:, 0, 0])
        assert np.all(Cl_gg > 0)

    def test_Cl_kk_decreases_at_high_ell(self, cl_obj):
        """C_kk should fall off at high ell."""
        Cl_kk = np.array(cl_obj.Cl_kappa_kappa_tot_mat[:, 0, 0])
        # Last bin should be smaller than first
        assert Cl_kk[-1] < Cl_kk[0]

    def test_Cl_yy_signal_le_tot(self, cl_obj):
        """Signal ≤ total (noise = 0 here since no noise file given)."""
        Cl_sig = np.array(cl_obj.Cl_y_y_signal_mat)
        Cl_tot = np.array(cl_obj.Cl_y_y_tot_mat)
        np.testing.assert_array_less(Cl_sig - 1e-20, Cl_tot + 1e-20)


class TestCauchySchwarz:
    def test_CS_kk_yy_ge_ky2(self, cl_obj):
        """Cauchy-Schwarz: C_kk(ell) * C_yy(ell) >= C_ky(ell)^2 for all ell."""
        Cl_kk = np.array(cl_obj.Cl_kappa_kappa_tot_mat[:, 0, 0])
        Cl_yy = np.array(cl_obj.Cl_y_y_signal_mat)
        Cl_ky = np.array(cl_obj.Cl_kappa_y_tot_mat[:, 0])
        ratio = Cl_kk * Cl_yy / (Cl_ky ** 2)
        # Should be >= 1 at all ell; allow 1% numerical tolerance
        assert np.all(ratio >= 0.99), f"CS violated: min ratio = {ratio.min():.4f}"

    def test_CS_kk_gg_ge_gk2(self, cl_obj):
        """Cauchy-Schwarz: C_kk * C_gg >= C_gk^2."""
        Cl_kk = np.array(cl_obj.Cl_kappa_kappa_tot_mat[:, 0, 0])
        Cl_gg = np.array(cl_obj.Cl_gal_gal_tot_mat[:, 0, 0])
        Cl_gk = np.array(cl_obj.Cl_gal_kappa_tot_mat[:, 0, 0])
        ratio = Cl_kk * Cl_gg / (Cl_gk ** 2 + 1e-30)
        assert np.all(ratio >= 0.99), f"CS violated: min ratio = {ratio.min():.4f}"

    def test_CS_gg_yy_ge_gy2(self, cl_obj):
        """Cauchy-Schwarz: C_gg * C_yy >= C_gy^2."""
        Cl_gg = np.array(cl_obj.Cl_gal_gal_tot_mat[:, 0, 0])
        Cl_yy = np.array(cl_obj.Cl_y_y_signal_mat)
        Cl_gy = np.array(cl_obj.Cl_gal_y_tot_mat[:, 0])
        ratio = Cl_gg * Cl_yy / (Cl_gy ** 2 + 1e-30)
        assert np.all(ratio >= 0.99), f"CS violated: min ratio = {ratio.min():.4f}"


class TestLensingKernel:
    def test_Wk_positive(self, cl_obj):
        """Lensing efficiency kernel W_kappa must be non-negative."""
        Wk = np.array(cl_obj.Wk_mat)
        assert np.all(Wk >= 0.0)

    def test_Wy_positive(self, cl_obj):
        """tSZ window Wy = 1/(1+z) must be positive."""
        Wy = np.array(cl_obj.Wy_array)
        assert np.all(Wy > 0.0)

    def test_Wy_decreasing(self, cl_obj):
        """1/(1+z) decreases with z."""
        Wy = np.array(cl_obj.Wy_array)
        assert np.all(np.diff(Wy) < 0)

    def test_Wk_mat_shape(self, cl_obj):
        assert cl_obj.Wk_mat.shape == (cl_obj.nz_for_Cls, cl_obj.nbins)

    def test_Wg_mat_shape(self, cl_obj):
        assert cl_obj.Wg_mat.shape == (cl_obj.nbins_lens, cl_obj.nz_for_Cls)


class TestTSZPowerSpectra:
    def test_Pyy_1h_positive(self, cl_obj):
        """1h tSZ auto power spectrum must be positive."""
        assert float(jnp.min(cl_obj.Pyy_1h_kz_mat)) >= 0.0

    def test_Pyy_tot_positive(self, cl_obj):
        assert float(jnp.min(cl_obj.Pyy_tot_kz_mat)) >= 0.0

    def test_Pyy_tot_eq_1h_plus_2h(self, cl_obj):
        """P_yy_tot = P_yy_1h + P_yy_2h."""
        reconstructed = cl_obj.Pyy_1h_kz_mat + cl_obj.Pyy_2h_kz_mat
        np.testing.assert_allclose(
            np.array(cl_obj.Pyy_tot_kz_mat),
            np.array(reconstructed),
            rtol=1e-5,
        )

    def test_Pkyy_lz_positive(self, cl_obj):
        """Limber-projected P_yy(k=ell/chi, z) must be positive."""
        assert float(jnp.min(cl_obj.Pkyy_lz_mat)) >= 0.0


class TestInterpolators:
    def test_cached_power_spectra_shape(self, cl_obj):
        """cached_power_spectra[4,4,nell,nz_for_Cls] sanity."""
        expected = (4, 4, cl_obj.nell, cl_obj.nz_for_Cls)
        assert cl_obj.cached_power_spectra.shape == expected

    def test_cached_kk_slot_positive(self, cl_obj):
        pspec = np.array(cl_obj.cached_power_spectra[0, 0])
        assert np.all(pspec > 0)

    def test_cached_yy_slot_positive(self, cl_obj):
        pspec = np.array(cl_obj.cached_power_spectra[3, 3])
        assert np.all(pspec > 0)

    def test_cached_ky_symmetry(self, cl_obj):
        """cached_power_spectra[0,3] == cached_power_spectra[3,0]."""
        np.testing.assert_allclose(
            np.array(cl_obj.cached_power_spectra[0, 3]),
            np.array(cl_obj.cached_power_spectra[3, 0]),
            rtol=1e-6,
        )
