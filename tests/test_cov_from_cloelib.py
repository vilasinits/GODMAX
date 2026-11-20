"""get_cov_from_cloelib: GODMAX's covariance on an existing halo model and given C_ell.

The main check: given GODMAX's own C_ell, it must reproduce a standalone
get_cov run exactly, for both the Gaussian and the non-Gaussian part.
"""
import copy
import pathlib

import numpy as np
import pytest
import yaml

from godmax.cov_from_cloelib import get_cov_from_cloelib
from godmax.get_covs import get_cov
from godmax.get_Pkzs import get_Pkz

PARAMS_PATH = pathlib.Path(__file__).resolve().parents[1] / "param_files" / "params_default.yaml"

# All C_ell that get_cov reads, with galaxies.
CL_NAMES = (
    'Cl_kappa_kappa_tot_mat', 'Cl_kappa_y_tot_mat', 'Cl_y_y_signal_mat',
    'Cl_kcmb_kcmb_tot_mat', 'Cl_kappa_kcmb_tot_mat', 'Cl_kcmb_y_tot_mat',
    'Cl_gal_gal_tot_mat', 'Cl_gal_kappa_tot_mat', 'Cl_gal_y_tot_mat', 'Cl_gal_kcmb_tot_mat',
)


def load_config():
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)
    sim = copy.deepcopy(params["sim_params"])
    halo = copy.deepcopy(params["halo_params"])
    ana = copy.deepcopy(params["analysis"])
    other = copy.deepcopy(params["other_params"])
    # Small grids: this checks that the two routes agree, not accuracy.
    halo.update(nr=16, nz=4, nM=6, nk=16, nell=6)
    ana.update(model_galaxies=True, model_tSZ=True, nz_for_Cls=32, nbar_gal_comoving_val=5.0e-4)
    ana["num_points_trapz_int"] = 16
    ana["stats_for_cov"] = ['yy', 'ky', 'kk', 'gy', 'gg', 'gk', 'cc']
    ana["dl_array_survey"] = np.full(6, 100.0)
    ana["sigma_epsilon_SN_bins"] = [0.26]
    ana["neff_arcmin2_SN_bins"] = [5.0]
    ana["nbar_lens_bins"] = [0.3]
    return sim, halo, ana, other


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def standalone(config):
    """The reference: GODMAX's own pipeline, halo model + C_ell + covariance."""
    sim, halo, ana, other = copy.deepcopy(config)
    return get_cov(sim, halo, ana, other)


@pytest.fixture(scope="module")
def halo_model(config):
    """A separate halo-model run, as cloelib would provide it."""
    sim, halo, ana, other = copy.deepcopy(config)
    return get_Pkz(sim, halo, ana, other)


def godmax_cls(cov_obj):
    return {name: getattr(cov_obj, name) for name in CL_NAMES}


def assert_same_covariances(actual, expected):
    """Every block of covG_dict and covNG_dict must agree."""
    for dict_name in ('covG_dict', 'covNG_dict'):
        actual_dict = getattr(actual, dict_name)
        expected_dict = getattr(expected, dict_name)
        assert set(actual_dict) == set(expected_dict)
        for pair in expected_dict:
            for bin_key, block in expected_dict[pair].items():
                if bin_key == 'bins_comb':
                    assert actual_dict[pair][bin_key] == block
                    continue
                np.testing.assert_allclose(
                    actual_dict[pair][bin_key], block, rtol=1e-10, atol=0.0,
                    err_msg=f"{dict_name} {pair} {bin_key}",
                )


def test_matches_standalone_get_cov(config, standalone, halo_model):
    """Given GODMAX's own C_ell, the result is the standalone covariance."""
    sim, halo, ana, other = copy.deepcopy(config)
    result = get_cov_from_cloelib(halo_model, godmax_cls(standalone), sim, halo, ana, other)
    assert_same_covariances(result, standalone)


def test_cls_enter_only_the_gaussian_part(config, standalone, halo_model):
    """Null control: changing a C_ell changes the Gaussian part, not the non-Gaussian one."""
    sim, halo, ana, other = copy.deepcopy(config)
    cls = godmax_cls(standalone)
    cls['Cl_kappa_kappa_tot_mat'] = 2.0 * cls['Cl_kappa_kappa_tot_mat']
    result = get_cov_from_cloelib(halo_model, cls, sim, halo, ana, other)

    covG_new = result.covG_dict['kk_kk']['bin_1_1_1_1']
    covG_old = standalone.covG_dict['kk_kk']['bin_1_1_1_1']
    assert not np.allclose(covG_new, covG_old, rtol=1e-3, atol=0.0)
    for pair in standalone.covNG_dict:
        for bin_key, block in standalone.covNG_dict[pair].items():
            if bin_key != 'bins_comb':
                np.testing.assert_allclose(result.covNG_dict[pair][bin_key], block, rtol=1e-10, atol=0.0)


def test_missing_cl_raises(config, standalone, halo_model):
    sim, halo, ana, other = copy.deepcopy(config)
    cls = godmax_cls(standalone)
    del cls['Cl_gal_y_tot_mat']
    with pytest.raises(ValueError, match="cls is missing .Cl_gal_y_tot_mat."):
        get_cov_from_cloelib(halo_model, cls, sim, halo, ana, other)


def test_wrong_cl_shape_raises(config, standalone, halo_model):
    sim, halo, ana, other = copy.deepcopy(config)
    cls = godmax_cls(standalone)
    cls['Cl_kcmb_y_tot_mat'] = cls['Cl_kcmb_y_tot_mat'][:-1]
    with pytest.raises(ValueError, match="has shape"):
        get_cov_from_cloelib(halo_model, cls, sim, halo, ana, other)


def test_different_mass_grid_raises(config, standalone, halo_model):
    sim, halo, ana, other = copy.deepcopy(config)
    halo['nM'] = 8
    with pytest.raises(ValueError, match="M_array differs"):
        get_cov_from_cloelib(halo_model, godmax_cls(standalone), sim, halo, ana, other)


def test_different_cosmology_raises(config, standalone, halo_model):
    sim, halo, ana, other = copy.deepcopy(config)
    sim['cosmo'] = dict(sim['cosmo'], sigma8=0.7)
    with pytest.raises(ValueError, match="'sigma8'"):
        get_cov_from_cloelib(halo_model, godmax_cls(standalone), sim, halo, ana, other)


def test_probe_subset_without_tsz(config, standalone):
    """gg and gk without tSZ: only the gg, gk and kk C_ell are passed, and the
    blocks equal those of the full standalone run."""
    sim, halo, ana, other = copy.deepcopy(config)
    ana['model_tSZ'] = False
    ana['stats_for_cov'] = ['kk', 'gg', 'gk']
    halo_model_no_tsz = get_Pkz(*copy.deepcopy((sim, halo, ana, other)))
    cls = {
        name: getattr(standalone, name)
        for name in ('Cl_kappa_kappa_tot_mat', 'Cl_gal_gal_tot_mat', 'Cl_gal_kappa_tot_mat')
    }
    result = get_cov_from_cloelib(halo_model_no_tsz, cls, sim, halo, ana, other)

    assert set(result.covG_dict) == {'kk_kk', 'kk_gg', 'kk_gk', 'gg_gg', 'gg_gk', 'gk_gk'}
    for dict_name in ('covG_dict', 'covNG_dict'):
        for pair, blocks in getattr(result, dict_name).items():
            for bin_key, block in blocks.items():
                if bin_key != 'bins_comb':
                    np.testing.assert_allclose(
                        block, getattr(standalone, dict_name)[pair][bin_key], rtol=1e-10, atol=0.0,
                        err_msg=f"{dict_name} {pair} {bin_key}",
                    )


def test_probe_with_y_needs_tsz(config, standalone):
    sim, halo, ana, other = copy.deepcopy(config)
    ana['model_tSZ'] = False
    ana['stats_for_cov'] = ['gy']
    halo_model_no_tsz = get_Pkz(*copy.deepcopy((sim, halo, ana, other)))
    with pytest.raises(ValueError, match="contains y"):
        get_cov_from_cloelib(halo_model_no_tsz, godmax_cls(standalone), sim, halo, ana, other)


def test_more_lens_bins_than_source_bins():
    """Regression: get_cov built one galaxy profile per SOURCE bin, so with more
    lens bins than source bins every extra lens bin silently reused the last
    profile (the non-Gaussian gg blocks of bins 1 and 3 were identical and 100%
    correlated). Bins 1 and 3 barely overlap: their 1-halo cross term must vanish."""
    sim, halo, ana, other = load_config()
    z = np.linspace(0.01, 1.6, 64)
    lens = {"nbins_lens": 3, "z_array_lens": z}
    for i, mean in enumerate([0.3, 0.6, 0.9]):
        lens[f"nz{i}"] = np.exp(-0.5 * ((z - mean) / 0.08) ** 2)
    ana["nz_lens_info_dict"] = lens          # 3 lens bins; the source n(z) has 1 bin
    ana["nbar_lens_bins"] = [0.3, 0.3, 0.3]
    ana["stats_for_cov"] = ['gg']
    cov = get_cov(sim, halo, ana, other)

    ng = cov.covNG_dict['gg_gg']
    auto_1 = np.diag(ng['bin_1_1_1_1'])
    auto_3 = np.diag(ng['bin_3_3_3_3'])
    cross_13 = np.diag(ng['bin_1_1_3_3'])
    assert not np.allclose(auto_3, auto_1, rtol=1e-3, atol=0.0)
    assert np.all(np.abs(cross_13) / np.sqrt(auto_1 * auto_3) < 1e-6)
