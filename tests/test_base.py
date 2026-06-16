"""
Unit tests for base_class: grid shapes, cosmology params, vmapped helpers.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from godmax.base_class import get_vmapped_func, get_vmapped_func_warg
from .conftest import HALO_PARAMS, SIM_PARAMS


class TestGridShapes:
    def test_r_array_shape(self, base_obj):
        assert base_obj.r_array.shape == (HALO_PARAMS["nr"],)

    def test_z_array_shape(self, base_obj):
        assert base_obj.z_array.shape == (HALO_PARAMS["nz"],)

    def test_M_array_shape(self, base_obj):
        assert base_obj.M_array.shape == (HALO_PARAMS["nM"],)

    def test_kPk_array_shape(self, base_obj):
        assert base_obj.kPk_array.shape == (HALO_PARAMS["nk"],)

    def test_ell_array_shape(self, base_obj):
        assert base_obj.ell_array.shape == (HALO_PARAMS["nell"],)

    def test_r_array_logspaced(self, base_obj):
        """r_array should be log-spaced: ratio of consecutive elements constant."""
        ratios = base_obj.r_array[1:] / base_obj.r_array[:-1]
        assert float(jnp.std(ratios) / jnp.mean(ratios)) < 1e-5

    def test_k_array_logspaced(self, base_obj):
        ratios = base_obj.kPk_array[1:] / base_obj.kPk_array[:-1]
        assert float(jnp.std(ratios) / jnp.mean(ratios)) < 1e-5

    def test_ell_array_logspaced(self, base_obj):
        ratios = base_obj.ell_array[1:] / base_obj.ell_array[:-1]
        assert float(jnp.std(ratios) / jnp.mean(ratios)) < 1e-5

    def test_M_array_bounds(self, base_obj):
        lg_min = HALO_PARAMS["lg10_Mmin"]
        lg_max = HALO_PARAMS["lg10_Mmax"]
        assert float(jnp.log10(base_obj.M_array[0])) == pytest.approx(lg_min, abs=0.01)
        assert float(jnp.log10(base_obj.M_array[-1])) == pytest.approx(lg_max, abs=0.01)

    def test_scale_fac_from_z(self, base_obj):
        """a = 1/(1+z) consistency."""
        expected = 1.0 / (1.0 + base_obj.z_array)
        np.testing.assert_allclose(
            np.array(base_obj.scale_fac_a_array),
            np.array(expected), rtol=1e-6
        )


class TestCosmologyParams:
    def test_h_value(self, base_obj):
        assert base_obj.h == pytest.approx(SIM_PARAMS["cosmo"]["H0"] / 100.0, rel=1e-6)

    def test_Om0_value(self, base_obj):
        assert base_obj.Om0 == pytest.approx(SIM_PARAMS["cosmo"]["Om0"], rel=1e-6)

    def test_rho_m_bar_positive(self, base_obj):
        assert float(base_obj.rho_m_bar) > 0

    def test_rho_m_bar_scale(self, base_obj):
        """Mean matter density ~1e11 M_sun/Mpc^3 order."""
        rho = float(base_obj.rho_m_bar)
        assert 1e10 < rho < 1e12

    def test_cosmo_jax_initialized(self, base_obj):
        """jax_cosmo Cosmology object must exist."""
        from jax_cosmo import Cosmology
        assert isinstance(base_obj.cosmo_jax, Cosmology)


class TestVmapHelpers:
    """get_vmapped_func and get_vmapped_func_warg return callables."""

    def test_get_vmapped_func_2args(self):
        # Outer vmap over arg-1 (ys), inner over arg-0 (xs) → shape (ny, nx)
        def f(x, y): return x + y
        vf = get_vmapped_func(f, 2)
        xs = jnp.arange(3, dtype=float)
        ys = jnp.arange(4, dtype=float)
        out = vf(xs, ys)
        assert out.shape == (4, 3)

    def test_get_vmapped_func_3args(self):
        # Nested vmap: outermost over arg-2 → shape (nz, ny, nx)
        def f(x, y, z): return x * y * z
        vf = get_vmapped_func(f, 3)
        xs = jnp.arange(2, dtype=float)
        ys = jnp.arange(3, dtype=float)
        zs = jnp.arange(4, dtype=float)
        out = vf(xs, ys, zs)
        assert out.shape == (4, 3, 2)
