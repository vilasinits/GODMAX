"""
Unit tests for Profiles class:
  - HMF positivity
  - NFW mass normalization
  - Baryon budget (fgas + fstar ≤ fb)
  - Concentration–mass relation monotonicity
  - Gas + CLM + CGA = DMB (profile decomposition)
  - Pressure profile positivity (when model_tSZ)
"""

import pytest
import jax.numpy as jnp
import numpy as np
from .conftest import SIM_PARAMS


FB = SIM_PARAMS["cosmo"]["Ob0"] / SIM_PARAMS["cosmo"]["Om0"]  # baryon fraction


class TestHMF:
    def test_hmf_positive(self, profiles_obj):
        """dn/dM must be non-negative everywhere."""
        assert float(jnp.min(profiles_obj.hmf_Mz_mat)) >= 0.0

    def test_hmf_shape(self, profiles_obj):
        nz = profiles_obj.nz
        nM = profiles_obj.nM
        assert profiles_obj.hmf_Mz_mat.shape == (nz, nM)

    def test_hmf_peaks_at_intermediate_mass(self, profiles_obj):
        """HMF peaks at some intermediate mass, not at extremes."""
        hmf_at_z0 = np.array(profiles_obj.hmf_Mz_mat[0])
        peak_idx = int(np.argmax(hmf_at_z0))
        nM = profiles_obj.nM
        assert 0 < peak_idx < nM - 1

    def test_sigma_positive(self, profiles_obj):
        assert float(jnp.min(profiles_obj.sigma_Mz_mat)) > 0.0

    def test_nu_positive(self, profiles_obj):
        """nu = delta_c / sigma > 0."""
        assert float(jnp.min(profiles_obj.nu_Mz_mat)) > 0.0


class TestConcentration:
    def test_conc_positive(self, profiles_obj):
        assert float(jnp.min(profiles_obj.conc_Mz_mat)) > 0.0

    def test_conc_decreases_with_mass(self, profiles_obj):
        """Duffy08: c decreases with mass at fixed z."""
        conc_z0 = np.array(profiles_obj.conc_Mz_mat[0])
        diffs = np.diff(conc_z0)
        # Allow small numerical noise — majority of differences should be negative
        frac_negative = np.mean(diffs < 0)
        assert frac_negative > 0.7

    def test_conc_range(self, profiles_obj):
        """Typical halo concentrations 2–30."""
        c_min = float(jnp.min(profiles_obj.conc_Mz_mat))
        c_max = float(jnp.max(profiles_obj.conc_Mz_mat))
        assert 1.0 < c_min
        assert c_max < 60.0


class TestNFWProfile:
    def test_rho_nfw_positive(self, profiles_obj):
        assert float(jnp.min(profiles_obj.rho_nfw_mat)) >= 0.0

    def test_rho_nfw_shape(self, profiles_obj):
        nr, nz, nM = profiles_obj.nr, profiles_obj.nz, profiles_obj.nM
        assert profiles_obj.rho_nfw_mat.shape == (nr, nz, nM)

    def test_rho_nfw_decreases_radially(self, profiles_obj):
        """NFW decreases monotonically with radius at most (z, M) combinations."""
        nfw = np.array(profiles_obj.rho_nfw_mat)  # (nr, nz, nM)
        diffs = np.diff(nfw, axis=0)
        frac_decreasing = np.mean(diffs < 0)
        assert frac_decreasing > 0.85

    def test_Mtot_positive(self, profiles_obj):
        assert float(jnp.min(profiles_obj.Mtot_mat)) > 0.0

    def test_r200c_positive(self, profiles_obj):
        assert float(jnp.min(profiles_obj.r200c_mat)) > 0.0

    def test_r200c_increases_with_mass(self, profiles_obj):
        """r200c ~ M^(1/3): should increase with halo mass at fixed z."""
        r200c_z0 = np.array(profiles_obj.r200c_mat[0])
        diffs = np.diff(r200c_z0)
        assert np.all(diffs > 0)


class TestBaryonBudget:
    def test_fgas_non_negative(self, profiles_obj):
        """Gas fraction can't be negative."""
        assert float(jnp.min(profiles_obj.fgas_mat)) >= 0.0

    def test_fgas_below_cosmic(self, profiles_obj):
        """Gas fraction ≤ cosmic baryon fraction at every grid point."""
        assert float(jnp.max(profiles_obj.fgas_mat)) <= FB * 1.05  # 5% tolerance for interp

    def test_fstar_non_negative(self, profiles_obj):
        assert float(jnp.min(profiles_obj.fstar_tot_mat)) >= 0.0

    def test_fbar_tot_le_cosmic(self, profiles_obj):
        """fgas + fstar ≤ fb everywhere (mass is conserved)."""
        fbar = profiles_obj.fgas_mat + profiles_obj.fstar_tot_mat
        assert float(jnp.max(fbar)) <= FB * 1.05

    def test_fclm_consistent(self, profiles_obj):
        """fclm = (1 - fb) + fstar_sat, should be < 1."""
        assert float(jnp.max(profiles_obj.fclm_mat)) < 1.0
        assert float(jnp.min(profiles_obj.fclm_mat)) > 0.0


class TestDMBDecomposition:
    def test_rho_dmb_positive(self, profiles_obj):
        """Total baryonified matter profile must be positive."""
        assert float(jnp.min(profiles_obj.rho_dmb_mat)) >= 0.0

    def test_dmb_decomposition(self, profiles_obj):
        """rho_dmb = rho_gas + rho_clm + rho_cga."""
        reconstructed = (profiles_obj.rho_gas_mat
                         + profiles_obj.rho_clm_mat
                         + profiles_obj.rho_cga_mat)
        np.testing.assert_allclose(
            np.array(profiles_obj.rho_dmb_mat),
            np.array(reconstructed),
            rtol=1e-5,
        )

    def test_rho_gas_positive(self, profiles_obj):
        assert float(jnp.min(profiles_obj.rho_gas_mat)) >= 0.0

    def test_rho_clm_positive(self, profiles_obj):
        assert float(jnp.min(profiles_obj.rho_clm_mat)) >= 0.0

    def test_rho_cga_positive(self, profiles_obj):
        assert float(jnp.min(profiles_obj.rho_cga_mat)) >= 0.0


class TestPressureProfile:
    def test_y3d_positive(self, profiles_obj):
        """Compton-y 3D profile must be non-negative."""
        assert float(jnp.min(profiles_obj.y3d_mat)) >= 0.0

    def test_y3d_shape(self, profiles_obj):
        nr, nz, nM = profiles_obj.nr, profiles_obj.nz, profiles_obj.nM
        assert profiles_obj.y3d_mat.shape == (nr, nz, nM)

    def test_y3d_decreases_radially(self, profiles_obj):
        """Pressure profile decreases with radius."""
        y3d = np.array(profiles_obj.y3d_mat)
        diffs = np.diff(y3d, axis=0)
        frac_decreasing = np.mean(diffs < 0)
        assert frac_decreasing > 0.85
