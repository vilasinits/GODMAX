"""GODMAX covariance from a halo model and C_ell that were already computed.

GODMAX's covariance (`get_cov` in `get_covs.py`) has two parts:

- the Gaussian part, which needs the C_ell of every pair of fields, the noise
  and the sky fraction;
- the non-Gaussian part (1-halo trispectrum), which needs the halo profiles
  projected with the kernel of each field.

Run on its own, `get_cov` first runs the whole GODMAX pipeline: halo model,
then its own C_ell, then the covariance. When cloelib has already run the GODMAX
halo model and computed the C_ell, doing that again doubles the run time.
`get_cov_from_cloelib` avoids this:

1. The C_ell-side settings (n(z), ell grid, redshift grid of the C_ell
   integrals, intrinsic alignment, photo-z shifts, beam, CMB source redshift,
   noise, sky fractions) are read by GODMAX from the four config dicts, exactly
   as in a standalone run.
2. The halo-model profiles are taken from the GODMAX object that cloelib ran
   (`GODMAXNonLinearPerturbations.godmax`). They are not recomputed.
3. The kernels of each field are built with GODMAX's own methods.
4. The C_ell are the ones passed in (for example from cloelib).
5. `get_cov` then runs unchanged, so the covariance formulas are exactly
   GODMAX's.

Only the C_ell integrals of GODMAX are skipped.

Fields and probes
-----------------
The fields are named by one letter, as in `get_cov`:

- `k`: shear (one per source bin)
- `g`: galaxies (one per lens bin)
- `y`: Compton-y
- `c`: CMB lensing

A probe is a pair of fields, e.g. `gk` (galaxy-galaxy lensing). The probes are
listed in `analysis['stats_for_cov']`; the covariance is computed for every pair
of listed probes, e.g. `['gg', 'gk']` gives the blocks gg-gg, gg-gk and gk-gk.

Only the C_ell between the fields of the listed probes are needed (for
`['gg', 'gk']`: gg, gk and kk). The other C_ell may be left out: `get_cov` still
stores them, so they are set to zero, but no listed block uses them. In the
same way, y is only needed if a listed probe contains y.

Units are GODMAX's (h-units). The C_ell are dimensionless and need no
conversion, but they must describe the same data model as the covariance: the
y spectra include the beam, and the shear spectra include the intrinsic
alignment and multiplicative bias, as in GODMAX.
"""

import math

import interpax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from godmax.base_class import base_class, get_vmapped_func
from godmax.get_Cls import get_Cl
from godmax.get_covs import get_cov

# Halo-model arrays taken from the GODMAX object that cloelib ran.
# Shapes are (n_k, n_z, n_M) for the profiles and (n_z, n_M) for the others.
MATTER_HALO_MODEL_ARRAYS = (
    'uk_dmb',       # matter profile (baryonified) in Fourier space
    'Mtot_mat',     # total mass of each halo
    'hmf_Mz_mat',   # halo mass function dn/dlnM
)
TSZ_HALO_MODEL_ARRAYS = (
    'uk_y',         # Compton-y profile in Fourier space
)
GALAXY_HALO_MODEL_ARRAYS = (
    'ukg_cross',    # galaxy profile (HOD) in Fourier space
)

# Halo-model grids that must be the same in the GODMAX object and in the config.
SHARED_GRIDS = ('z_array', 'M_array', 'kPk_array')

# GODMAX's name of the C_ell of each pair of fields (see "Fields and probes").
CL_NAMES = {
    ('k', 'k'): 'Cl_kappa_kappa_tot_mat',
    ('k', 'y'): 'Cl_kappa_y_tot_mat',
    ('y', 'y'): 'Cl_y_y_signal_mat',
    ('c', 'c'): 'Cl_kcmb_kcmb_tot_mat',
    ('k', 'c'): 'Cl_kappa_kcmb_tot_mat',
    ('c', 'y'): 'Cl_kcmb_y_tot_mat',
    ('g', 'g'): 'Cl_gal_gal_tot_mat',
    ('g', 'k'): 'Cl_gal_kappa_tot_mat',
    ('g', 'y'): 'Cl_gal_y_tot_mat',
    ('g', 'c'): 'Cl_gal_kcmb_tot_mat',
}


def get_cov_from_cloelib(
    pkz_obj,
    cls,
    sim_params_dict,
    halo_params_dict,
    analysis_dict,
    other_params_dict,
):
    """Compute GODMAX's covariance from an existing halo model and given C_ell.

    Args:
        pkz_obj: The GODMAX object holding the halo model, e.g.
            `GODMAXNonLinearPerturbations(...).godmax` from cloelib. It must
            include tSZ if a listed probe contains y, and galaxies if a listed
            probe contains g.
        cls: Dict of C_ell arrays on GODMAX's `ell_array`, keyed by GODMAX's
            names (`CL_NAMES`). Only the pairs of fields used by the listed
            probes are needed (see the module docstring). n_ell is the length of
            `ell_array`; n_source and n_lens are the numbers of shear and galaxy
            bins of the n(z) in the config.

            - `Cl_kappa_kappa_tot_mat`: shear x shear, (n_ell, n_source, n_source)
            - `Cl_kappa_y_tot_mat`: shear x y, (n_ell, n_source)
            - `Cl_y_y_signal_mat`: y x y, signal only, (n_ell,); the noise is
              added here from the config
            - `Cl_kcmb_kcmb_tot_mat`: CMB lensing x CMB lensing, (n_ell,)
            - `Cl_kappa_kcmb_tot_mat`: shear x CMB lensing, (n_ell, n_source)
            - `Cl_kcmb_y_tot_mat`: CMB lensing x y, (n_ell,)
            - `Cl_gal_gal_tot_mat`: galaxy x galaxy, (n_ell, n_lens, n_lens)
            - `Cl_gal_kappa_tot_mat`: galaxy x shear, (n_ell, n_lens, n_source)
            - `Cl_gal_y_tot_mat`: galaxy x y, (n_ell, n_lens)
            - `Cl_gal_kcmb_tot_mat`: galaxy x CMB lensing, (n_ell, n_lens)
        sim_params_dict, halo_params_dict, analysis_dict, other_params_dict:
            The GODMAX config. It sets the probes (`stats_for_cov`), the n(z),
            ell grid, C_ell redshift grid, intrinsic alignment, photo-z shifts,
            beam, CMB source redshift, noise and sky fractions (`fsky_*`,
            `dl_array_survey`, ...). The halo-model grids (redshifts, masses,
            wavenumbers) and the cosmology are those of `pkz_obj`.

    Returns:
        The `get_cov` object. The covariances are in its `covG_dict` (Gaussian),
        `covNG_dict` (non-Gaussian) and `covtot_dict` (sum), as in a standalone
        `get_cov` run.
    """
    # --- Step 1: read the C_ell-side settings from the config. ---------------
    # The halo model was computed on pkz_obj's redshifts, so use the same ones.
    halo_params_dict = dict(halo_params_dict)
    halo_params_dict['z_array'] = pkz_obj.z_array

    # Use the same cosmology as pkz_obj. On the cloelib path, pkz_obj holds the
    # cloelib cosmology object, so pass that same object on.
    analysis_dict = dict(analysis_dict)
    if pkz_obj.cosmology_backend == 'cloelib':
        analysis_dict['cosmology_backend'] = 'cloelib'
        analysis_dict['cosmology_obj'] = pkz_obj.cosmo_cloelib

    # base_class reads every setting and computes the background on the grids.
    # It does not compute any halo profile.
    settings = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)

    # --- Step 2: check that the config matches the halo model. ---------------
    _check_same_cosmology(settings, pkz_obj)
    for name in SHARED_GRIDS:
        _check_same_grid(settings, pkz_obj, name)
    for switch in ('model_tSZ', 'model_galaxies'):
        if getattr(settings, switch) != getattr(pkz_obj, switch):
            raise ValueError(
                f"{switch} is {getattr(settings, switch)} in the config but "
                f"{getattr(pkz_obj, switch)} in the halo model; they must be the same."
            )

    fields = _fields_of_probes(analysis_dict['stats_for_cov'])
    if 'y' in fields and not settings.model_tSZ:
        raise ValueError("A listed probe contains y: the halo model needs model_tSZ=True.")
    if 'g' in fields and not settings.model_galaxies:
        raise ValueError("A listed probe contains g: the halo model needs galaxies.")

    # --- Step 3: an object with the settings and the halo model. -------------
    # get_Cl.__new__ creates an empty get_Cl object without running its __init__,
    # which is the part that computes GODMAX's own C_ell.
    cl_obj = get_Cl.__new__(get_Cl)
    cl_obj.__dict__.update(settings.__dict__)

    halo_model_names = list(MATTER_HALO_MODEL_ARRAYS)
    if settings.model_tSZ:
        halo_model_names = halo_model_names + list(TSZ_HALO_MODEL_ARRAYS)
    if settings.model_galaxies:
        halo_model_names = halo_model_names + list(GALAXY_HALO_MODEL_ARRAYS)
    for name in halo_model_names:
        setattr(cl_obj, name, getattr(pkz_obj, name))

    # --- Step 4: the kernels, with GODMAX's own methods. ---------------------
    _add_kernels(cl_obj)

    # --- Step 5: the C_ell passed in, and the y noise. -----------------------
    _add_cls(cl_obj, cls, fields)
    _add_yy_noise(cl_obj, analysis_dict)

    # --- Step 6: GODMAX's covariance, unchanged. -----------------------------
    return get_cov(
        sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Cl_obj=cl_obj
    )


def _fields_of_probes(probes):
    """Return the set of fields in a list of probes, e.g. ['gg', 'gk'] -> {'g', 'k'}."""
    fields = set()
    for probe in probes:
        for field in probe:
            if field not in ('k', 'g', 'y', 'c'):
                raise ValueError(
                    f"Unknown field {field!r} in probe {probe!r}; the fields are "
                    "k (shear), g (galaxies), y (Compton-y) and c (CMB lensing)."
                )
            fields.add(field)
    return fields


def _check_same_cosmology(settings, pkz_obj):
    """Raise if the config and the halo model use different cosmological parameters."""
    for name in pkz_obj.cosmo_params:
        if name == 'flat':
            continue
        value_config = float(settings.cosmo_params[name])
        value_halo_model = float(pkz_obj.cosmo_params[name])
        if not np.isclose(value_config, value_halo_model, rtol=1e-12, atol=0.0):
            raise ValueError(
                f"Cosmological parameter {name!r} is {value_config} in the config but "
                f"{value_halo_model} in the halo model; they must be the same."
            )


def _check_same_grid(settings, pkz_obj, name):
    """Raise if a halo-model grid differs between the config and the halo model."""
    grid_config = np.asarray(getattr(settings, name))
    grid_halo_model = np.asarray(getattr(pkz_obj, name))
    same = grid_config.shape == grid_halo_model.shape and np.allclose(
        grid_config, grid_halo_model, rtol=1e-12, atol=0.0
    )
    if not same:
        raise ValueError(
            f"{name} differs between the config and the halo model; use the same "
            "halo-model settings (grids) for both."
        )


def _add_kernels(cl_obj):
    """Build the kernels of each field and the y profile in ell space.

    These are the same lines as in `get_Cl.__init__` (get_Cls.py, "Get the window
    functions for different probes" and the tSZ y profile below it).
    """
    source_bins = jnp.arange(cl_obj.nbins)
    z_indices_for_Cls = jnp.arange(cl_obj.nz_for_Cls)

    # Shear: n(z) with the photo-z shift, lensing kernel and NLA kernel.
    cl_obj.pzs_inp_mat = vmap(cl_obj.get_photoz_biased_nz)(source_bins)
    cl_obj.Wk_gravonly_mat = get_vmapped_func(cl_obj.get_weak_lensing_kernel, 2)(
        source_bins, z_indices_for_Cls
    ).T
    cl_obj.nla_mat = get_vmapped_func(cl_obj.get_nla_kernel, 2)(
        source_bins, z_indices_for_Cls
    ).T
    cl_obj.Wk_mat = cl_obj.Wk_gravonly_mat + cl_obj.nla_mat

    # CMB lensing and Compton-y.
    cl_obj.Wkcmb_array = vmap(cl_obj.get_cmb_lensing_kernel)(
        cl_obj.z_array_for_Cls, cl_obj.chi_array_for_Cls
    )
    cl_obj.Wy_array = 1.0 / (1.0 + cl_obj.z_array_for_Cls)

    # Galaxies: the normalised lens n(z).
    if cl_obj.model_galaxies:
        cl_obj.Wg_mat = vmap(cl_obj.get_nz_lens_interp)(jnp.arange(cl_obj.nbins_lens))
    else:
        cl_obj.Wg_mat = jnp.zeros((1, 1))

    # Compton-y profile in ell space (beam included), as used by the covariance.
    if cl_obj.model_tSZ:
        ell_indices = jnp.arange(cl_obj.nell)
        cl_obj.uyl_mat_tointp = get_vmapped_func(cl_obj.get_uyl, 3)(
            ell_indices, jnp.arange(cl_obj.nz), jnp.arange(cl_obj.nM)
        ).T
        cl_obj.uyl_mat = get_vmapped_func(cl_obj.get_uyl_interp, 2)(
            ell_indices, jnp.arange(cl_obj.nM)
        ).T
        cl_obj.uyl_mat = jnp.moveaxis(cl_obj.uyl_mat, 0, 1)
        cl_obj.uy_l_for_cov = cl_obj.get_uy_l_forcov()
    else:
        # get_cov always stores the y profile, even when no probe uses it.
        # Without tSZ there is none, so store zeros; no block reads them
        # (a probe with y raises earlier).
        cl_obj.uy_l_for_cov = jnp.zeros((cl_obj.nell, cl_obj.nz_for_Cls, cl_obj.nM))


def _add_cls(cl_obj, cls, fields):
    """Check the C_ell passed in and store them under GODMAX's names.

    The C_ell of every pair of `fields` must be given. The others are set to
    zero: `get_cov` stores them, but no listed block uses them.
    """
    n_ell = cl_obj.nell
    n_bins = {'k': cl_obj.nbins, 'g': cl_obj.nbins_lens, 'y': 1, 'c': 1}

    for pair, name in CL_NAMES.items():
        field1, field2 = pair

        # Expected shape: n_ell, then one axis per field with several bins.
        expected_shape = [n_ell]
        if field1 in ('k', 'g'):
            expected_shape.append(n_bins[field1])
        if field2 in ('k', 'g'):
            expected_shape.append(n_bins[field2])
        expected_shape = tuple(expected_shape)

        needed = field1 in fields and field2 in fields
        if name in cls:
            value = jnp.asarray(cls[name])
            if value.shape != expected_shape:
                raise ValueError(
                    f"cls[{name!r}] has shape {value.shape}, expected {expected_shape} "
                    f"(n_ell={n_ell}, n_source={n_bins['k']}, n_lens={n_bins['g']})."
                )
        elif needed:
            raise ValueError(
                f"cls is missing {name!r} ({field1}{field2}), which the listed probes "
                "need; see the docstring of get_cov_from_cloelib."
            )
        else:
            value = jnp.zeros(expected_shape)
        setattr(cl_obj, name, value)


def _add_yy_noise(cl_obj, analysis_dict):
    """Set the y noise and the y signal + noise, as `get_Cl.__init__` does.

    The noise comes from `yy_total_ell_fname` (total spectrum; noise = total minus
    signal) or from `yy_noise_ell_fname` (noise spectrum). Without either, the
    noise is zero. Same lines as in get_Cls.py ("yy naming convention").
    """
    signal = cl_obj.Cl_y_y_signal_mat
    yy_total_ell_fname = analysis_dict.get('yy_total_ell_fname', None)
    yy_noise_ell_fname = analysis_dict.get('yy_noise_ell_fname', None)

    if yy_total_ell_fname is not None:
        ell_yy_f, Cl_yy_f = np.loadtxt(yy_total_ell_fname, unpack=True, usecols=(0, 1))
        log_Cl_yy_tot_interp = interpax.Interpolator1D(
            jnp.log(jnp.array(ell_yy_f)),
            jnp.log(jnp.array(Cl_yy_f) + 1e-25),
            extrap=(math.log(Cl_yy_f[0]), math.log(Cl_yy_f[-1])),
        )
        cl_obj.Cl_y_y_tot_mat = jnp.exp(log_Cl_yy_tot_interp(jnp.log(cl_obj.ell_array)))
        cl_obj.Cl_y_y_noise_mat = jnp.maximum(cl_obj.Cl_y_y_tot_mat - signal, 0.0)
        print('Loaded yy total from file into Cl_y_y_tot_mat')
    elif yy_noise_ell_fname is not None:
        ell_yy_n, Cl_yy_n = np.loadtxt(yy_noise_ell_fname, unpack=True, usecols=(0, 1))
        log_Cl_yy_noise_interp = interpax.Interpolator1D(
            jnp.log(jnp.array(ell_yy_n)),
            jnp.log(jnp.abs(jnp.array(Cl_yy_n)) + 1e-25),
            extrap=True,
        )
        cl_obj.Cl_y_y_noise_mat = jnp.exp(log_Cl_yy_noise_interp(jnp.log(cl_obj.ell_array)))
        cl_obj.Cl_y_y_tot_mat = signal + cl_obj.Cl_y_y_noise_mat
        print('Loaded yy noise from file; Cl_y_y_tot_mat = signal + noise')
    else:
        print('Warning: no yy-total or yy-noise file provided; Cl_y_y_tot_mat = signal only')
        cl_obj.Cl_y_y_noise_mat = jnp.zeros_like(signal)
        cl_obj.Cl_y_y_tot_mat = signal
