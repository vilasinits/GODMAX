"""A CloelibCosmology passed as analysis['cosmology_obj'] drives GODMAX.

The injected object wraps the caller's own cloelib JAX objects (built from As). It
must select the cloelib backend, supply GODMAX's cosmological parameters, and give
the same background and linear P(k) as the built-in 'cloelib' backend run with the
matching parameters.
"""
import jax.numpy as jnp
import pytest

pytest.importorskip("cloelib")

from cloelib.cosmology.jax_cosmology import (
    JAXBackground,
    JAXLinearPerturbations,
    JAXNonLinearPerturbations,
)

from godmax.base_class import base_class
from godmax.helpers.cloelib_cosmology import CloelibCosmology

HALO = dict(z_array=[0.1, 0.5, 1.0], nk=16, nM=6, nr=16, ell_array=jnp.array([10.0, 100.0]))


@pytest.fixture(scope="module")
def cosmology():
    background = JAXBackground(
        H0=67.2, Omega_b0=0.049, Omega_cdm0=0.261, Omega_k0=0.0, As=2.1e-9, ns=0.96,
        mnu=0.0, w0=-0.9, wa=0.0, gamma_MG=0.55, N_mnu=0,
    )
    return CloelibCosmology(
        background, JAXLinearPerturbations(background), JAXNonLinearPerturbations(background)
    )


@pytest.fixture(scope="module")
def injected(cosmology):
    # The 'cosmo' entry is deliberately wrong: the injected object must win.
    sim = {'cosmo': {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.04, 'sigma8': 0.7, 'ns': 1.0, 'w0': -1.0}}
    return base_class(sim, HALO, {'cosmology_obj': cosmology}, {})


def test_injected_object_sets_backend_and_parameters(injected, cosmology):
    background = cosmology.background
    assert injected.cosmology_backend == 'cloelib'
    assert injected.cosmo_cloelib is cosmology
    assert injected.h == pytest.approx(background.h)
    assert injected.Om0 == pytest.approx(background.Omega_b0 + background.Omega_cdm0)
    assert injected.Ob0 == pytest.approx(background.Omega_b0)
    assert float(injected.cosmo_params['sigma8']) == pytest.approx(float(cosmology.linear.sigma8_0()))
    assert injected.cosmo_params['w0'] == background.w0


def test_injected_matches_parameter_built_backend(injected, cosmology):
    sim = {'cosmo': cosmology.godmax_params()}
    from_params = base_class(sim, HALO, {'cosmology_backend': 'cloelib'}, {})
    for name in ('chi_array', 'DA_array', 'dchi_dz_array', 'growth_array', 'plin_kz_mat'):
        assert jnp.allclose(getattr(injected, name), getattr(from_params, name), rtol=1e-10), name


def test_cosmology_obj_conflicts_with_jax_cosmo_backend(cosmology):
    with pytest.raises(ValueError, match="cosmology_obj requires"):
        base_class({}, HALO, {'cosmology_obj': cosmology, 'cosmology_backend': 'jax_cosmo'}, {})
