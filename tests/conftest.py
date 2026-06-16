"""
Shared session-scoped fixtures for GODMAX unit tests.

Uses a minimal grid (small nz, nM, nr, nk, nell) so the full pipeline
builds in ~60s total — one cold-JIT warmup per test session.
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Minimal 4-dict config — fast grids, all physics on
# ---------------------------------------------------------------------------

SIM_PARAMS = {
    "cosmo": {
        "flat": True, "H0": 67.2, "Om0": 0.31, "Ob0": 0.049,
        "sigma8": 0.81, "ns": 0.95, "w0": -1.0,
    },
    "init_power": True,
    "nfw_trunc": True,
    "epsilon_rt": 4.0,
    # BCMP gas params (defaults)
    "theta_ej_0": 2.0, "log10_Mstar0_theta_ej": 16.0,
    "nu_theta_ej_M": 0.0, "nu_theta_ej_z": 0.0,
    "theta_co_0": 0.05, "log10_Mstar0_theta_co": 16.0,
    "nu_theta_co_M": 0.0, "nu_theta_co_z": 0.0,
    "mu_beta": 0.6, "log10_Mstar0": 14.0, "log10_Mc0": 14.83,
    "nu_z": 0.0, "nu_M": 0.0,
    "gamma_rhogas": 2.0, "delta_rhogas": 7.0,
    "a_zeta": 0.3, "n_zeta": 2.0,
    "alpha_nt": 0.18, "beta_nt": 0.5, "n_nt": 0.3,
    # HOD / SHMR
    "log10M1_fshmr": 12.35, "log10M1_a_fshmr": 0.28,
    "log10Mstar0_fshmr": 10.72, "log10Mstar0_a_fshmr": 0.55,
    "beta_fshmr": 0.44, "beta_a_fshmr": 0.18,
    "delta_fshmr": 0.57, "delta_a_fshmr": 0.17,
    "gamma_fshmr": 1.56, "gamma_a_fshmr": 2.51,
    "siglogMstar_Ncen": 0.25,
    "alphasat_Nsat": 1.0, "Bcut_Nsat": 1.69,
    "Bsat_Nsat": 9.01, "betacut_Nsat": 0.6, "betasat_Nsat": 0.74,
    "eta_star": 0.3, "eta_cga": 0.6, "A_starcga": 0.09, "log10_M1_starcga": 11.4,
}

HALO_PARAMS = {
    # Tiny grids for fast tests
    "rmin": 0.005, "rmax": 8.0, "nr": 12,
    "zmin": 0.05, "zmax": 1.5, "nz": 6,
    "lg10_Mmin": 11.5, "lg10_Mmax": 15.5, "nM": 8,
    "kmin": 0.001, "kmax": 100.0, "nk": 14,
    "ellmin": 50.0, "ellmax": 5000.0, "nell": 8,
    "mdef_Delta": 200,
    "conc_model": "Duffy08",
    "hmf_model": "T10",
    "do_corr_2h_mm": True,
}

ANALYSIS = {
    "backreaction": False,          # faster; doesn't change sanity invariants
    "baryonification": True,
    "model_galaxies": True,
    "model_tSZ": True,
    "num_points_trapz_int": 16,
    "num_points_gal_cal": 16,
    "calc_nfw_only": True,
    "beam_fwhm_arcmin": 1.6,
    "verbose_time": False,
    "tSZ_transition_model": "poweradd",
    "nbar_gal_comoving_zarray": [0.01, 2.0, 16],
    "nbar_gal_comoving_val": 5e-4,
    "nz_source_info_dict": {
        "nbins": 1,
        "z_array_source": [0.05, 1.5, 32],
        "nz0": [1.0],
    },
    "nz_lens_info_dict": {
        "nbins_lens": 1,
        "z_array_lens": [0.05, 1.5, 32],
        "nz0": [1.0],
    },
    "zmin_for_Cls": 0.05,
    "zmax_for_Cls": 1.5,
    "nz_for_Cls": 32,
    "angles_data_array": [2.5, 250, 10],
    "fsky_yy": 0.4, "fsky_ky": 0.437, "fsky_kk": 0.437,
    "fsky_yg": 0.34, "fsky_kg": 0.1, "fsky_gg": 0.34,
    "stats_for_cov": ["ky", "kk", "gy", "gg", "gk"],
}

OTHER_PARAMS = {
    "A_IA": 0.1, "eta_IA": 0.1, "z0_IA": 0.62, "C1_rhocrit": 0.0134,
    "alpha_ky": 1.0, "alpha_gy": 1.0,
    "Delta_z_bias_array": [0.0],
    "mult_shear_bias_array": [0.0],
}


# ---------------------------------------------------------------------------
# Fixtures — session scope so JIT warmup happens once per test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_obj():
    from godmax.base_class import base_class
    return base_class(SIM_PARAMS, HALO_PARAMS, ANALYSIS, OTHER_PARAMS)


@pytest.fixture(scope="session")
def profiles_obj(base_obj):
    from godmax.get_radial_profiles import Profiles
    return Profiles(SIM_PARAMS, HALO_PARAMS, ANALYSIS, OTHER_PARAMS,
                    base_class_obj=base_obj)


@pytest.fixture(scope="session")
def pkz_obj(profiles_obj):
    from godmax.get_Pkzs import get_Pkz
    return get_Pkz(SIM_PARAMS, HALO_PARAMS, ANALYSIS, OTHER_PARAMS,
                   Profiles_obj=profiles_obj)


@pytest.fixture(scope="session")
def cl_obj(pkz_obj):
    from godmax.get_Cls import get_Cl
    return get_Cl(SIM_PARAMS, HALO_PARAMS, ANALYSIS, OTHER_PARAMS,
                  Pkz_obj=pkz_obj)


@pytest.fixture(scope="session")
def cov_obj(cl_obj):
    from godmax.get_covs import get_cov
    return get_cov(SIM_PARAMS, HALO_PARAMS, ANALYSIS, OTHER_PARAMS,
                   Cl_obj=cl_obj)
