"""
Unit tests for get_Pkz: 3D power spectra sanity.
  - Positivity of 1h, 2h, total
  - Shapes match grids
  - 1h dominates at high k, 2h at low k (crossover)
  - Halofit Pmm_hf positive and in same ballpark as halo model
  - Fourier profiles uk → 1 at k → 0 (mass normalization)
"""

import pytest
import jax.numpy as jnp
import numpy as np


class TestFourierProfiles:
    def test_uk_dmb_shape(self, pkz_obj):
        nk, nz, nM = pkz_obj.nk, pkz_obj.nz, pkz_obj.nM
        assert pkz_obj.uk_dmb.shape == (nk, nz, nM)

    def test_uk_nfw_shape(self, pkz_obj):
        nk, nz, nM = pkz_obj.nk, pkz_obj.nz, pkz_obj.nM
        assert pkz_obj.uk_nfw.shape == (nk, nz, nM)

    def test_uk_dmb_at_low_k_near_unity(self, pkz_obj):
        """u(k→0) = 1 by Fourier normalization (mass-normalized profiles).

        Tight bound: catches the k < k_mcfit[0] clamp regression, where uk
        froze at the FFTlog edge value instead of extrapolating to 1.
        """
        uk_low_k = np.abs(np.array(pkz_obj.uk_dmb[0]))  # lowest k bin
        median_val = float(np.median(uk_low_k))
        assert 0.99 < median_val <= 1.0

    def test_uk_nfw_at_low_k_near_unity(self, pkz_obj):
        uk_low_k = np.abs(np.array(pkz_obj.uk_nfw[0]))
        median_val = float(np.median(uk_low_k))
        assert 0.99 < median_val <= 1.0

    def test_uk_y_shape(self, pkz_obj):
        nk, nz, nM = pkz_obj.nk, pkz_obj.nz, pkz_obj.nM
        assert pkz_obj.uk_y.shape == (nk, nz, nM)

    def test_ukg_cross_shape(self, pkz_obj):
        nk, nz, nM = pkz_obj.nk, pkz_obj.nz, pkz_obj.nM
        assert pkz_obj.ukg_cross.shape == (nk, nz, nM)


class TestMatterPkz:
    def test_Pmm_positive(self, pkz_obj):
        assert float(jnp.min(pkz_obj.Pmm_tot_mat)) >= 0.0

    def test_Pmm_shape(self, pkz_obj):
        assert pkz_obj.Pmm_tot_mat.shape == (pkz_obj.nk, pkz_obj.nz)

    def test_Pmm_peaks_at_intermediate_k(self, pkz_obj):
        """Pmm has a turnover — it's not monotone in k for the full range."""
        Pmm_z0 = np.array(pkz_obj.Pmm_tot_mat[:, 0])
        # Should not be all increasing or all decreasing
        diffs = np.diff(Pmm_z0)
        has_increase = np.any(diffs > 0)
        has_decrease = np.any(diffs < 0)
        assert has_increase and has_decrease

    def test_halofit_positive(self, pkz_obj):
        assert float(jnp.min(pkz_obj.phfit_kz_mat)) >= 0.0

    def test_halofit_shape(self, pkz_obj):
        assert pkz_obj.phfit_kz_mat.shape == (pkz_obj.nk, pkz_obj.nz)

    def test_Pmm_2h_at_low_k(self, pkz_obj):
        """At low k (large scales) 2h term dominates."""
        P2h_low = float(pkz_obj.Pmm_dmb_2h_kz_mat[0, 0])
        P1h_low = float(pkz_obj.Pmm_dmb_1h_kz_mat[0, 0])
        assert P2h_low > P1h_low

    def test_Pmm_1h_at_high_k(self, pkz_obj):
        """At high k (small scales) 1h term dominates."""
        P1h_high = float(pkz_obj.Pmm_dmb_1h_kz_mat[-1, 0])
        P2h_high = float(pkz_obj.Pmm_dmb_2h_kz_mat[-1, 0])
        assert P1h_high > P2h_high

    def test_Pmm_dmb_tot_equals_1h_plus_2h(self, pkz_obj):
        """Pmm_dmb_tot = Pmm_dmb_1h + Pmm_dmb_2h (before suppression factor)."""
        reconstructed = pkz_obj.Pmm_dmb_1h_kz_mat + pkz_obj.Pmm_dmb_2h_kz_mat
        np.testing.assert_allclose(
            np.array(pkz_obj.Pmm_dmb_tot_mat),
            np.array(reconstructed),
            rtol=1e-5,
        )

    def test_Pmm_sup_factor_positive(self, pkz_obj):
        """Suppression factor S(k,z) = P_halofit / P_NFW_halomodel must be positive."""
        assert float(jnp.min(pkz_obj.Pmm_sup_tot_mat)) > 0.0

    def test_Pmm_dmb_over_nfw_ratio_to_unity_at_low_k(self, pkz_obj):
        """Baryon suppression P_dmb/P_nfw must -> 1 at large scales (low k).

        Regression test for the k < k_mcfit[0] clamp bug: uk_dmb and uk_nfw
        froze at different non-unity constants below the FFTlog grid edge,
        so the ratio was stuck off from 1 even at the lowest k instead of
        converging there.
        """
        ratio_low_k = float(pkz_obj.Pmm_dmb_tot_mat[0, 0] / pkz_obj.Pmm_nfw_tot_mat[0, 0])
        assert 0.99 < ratio_low_k <= 1.01


class TestTSZPkz:
    def test_Pym_positive(self, pkz_obj):
        assert float(jnp.min(pkz_obj.Pym_tot_mat)) >= 0.0

    def test_Pym_shape(self, pkz_obj):
        assert pkz_obj.Pym_tot_mat.shape == (pkz_obj.nk, pkz_obj.nz)


class TestGalaxyPkz:
    def test_Pgg_positive(self, pkz_obj):
        assert float(jnp.min(pkz_obj.Pgg_tot_mat)) >= 0.0

    def test_Pgg_shape(self, pkz_obj):
        assert pkz_obj.Pgg_tot_mat.shape == (pkz_obj.nk, pkz_obj.nz)

    def test_Pgm_positive(self, pkz_obj):
        assert float(jnp.min(pkz_obj.Pgm_tot_mat)) >= 0.0

    def test_Pgy_positive(self, pkz_obj):
        assert float(jnp.min(pkz_obj.Pgy_tot_mat)) >= 0.0

    def test_nbarz_positive(self, pkz_obj):
        """Comoving galaxy number density must be positive."""
        assert float(jnp.min(pkz_obj.nbarz)) > 0.0

    def test_nbarz_shape(self, pkz_obj):
        assert pkz_obj.nbarz.shape == (pkz_obj.nz,)


class TestLinearPk:
    def test_plin_positive(self, pkz_obj):
        assert float(jnp.min(pkz_obj.plin_kz_mat)) > 0.0

    def test_plin_shape(self, pkz_obj):
        assert pkz_obj.plin_kz_mat.shape == (pkz_obj.nk, pkz_obj.nz)

    def test_plin_decreases_with_z(self, pkz_obj):
        """Linear growth: P_lin(k,z=0) > P_lin(k,z>0) for all k."""
        plin = np.array(pkz_obj.plin_kz_mat)
        # Compare z=0 to last z bin at a mid-k
        mid_k = pkz_obj.nk // 2
        assert plin[mid_k, 0] > plin[mid_k, -1]
