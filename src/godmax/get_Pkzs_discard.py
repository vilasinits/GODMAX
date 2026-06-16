from functools import partial

import jax.numpy as jnp
import jax.scipy.integrate as jsi
from jax import jit, vmap
 
from .base_class import get_vmapped_func, get_vmapped_func_warg
from .get_radial_profiles import Profiles
from .helpers import constants
from .helpers.jax_cosmo_power import halofit_parameters, nonlinear_matter_power
from .matter_pk_symbolic import *
from .mcfitjax.cosmology_jax import xi2P


class get_Pkz(Profiles):
    """
    Compute 3D power spectra over k and z for matter, tSZ, and galaxy fields.

    This version separates two effects that were mixed in the original code:

    1. Baryonification / matter-profile choice
       - DMB profile: probe 0
       - NFW profile: probe 1

    2. Halo-model-to-Halofit normalisation
       - P_hfit / P_mm^NFW,HM

    The true baryonic matter suppression used in the paper is

        S_bary(k, z) = P_mm^DMB(k, z) / P_mm^NFW(k, z).

    The previous variable Pmm_sup_tot_mat was not this quantity. It was

        P_hfit(k, z) / P_mm^NFW,HM(k, z),

    i.e. a halo-model-to-Halofit normalisation. For backward compatibility,
    Pmm_sup_tot_mat is still defined, but clearer names are also provided.

    Expected optional attributes/settings
    -------------------------------------
    use_baryonification : bool, optional
        Global switch for using DMB instead of NFW in matter-containing spectra.
        If absent, defaults to ``model_matter != 'halofit'`` to mimic the old
        matter-spectrum behaviour.

    apply_hm_to_halofit_norm : bool, optional
        Whether to multiply halo-model spectra by P_hfit / P_mm^NFW,HM.
        Defaults to True to remain close to the old behaviour.

    apply_bary_response_to_gg : bool, optional
        Whether to apply S_bary as an effective response to P_gg. Defaults to False,
        because P_gg does not directly contain the matter profile.

    apply_bary_response_to_ge : bool, optional
        Whether to apply S_bary as an effective response to P_ge. Defaults to False.

    apply_bary_response_to_tsz_cross : bool, optional
        Whether to apply S_bary as an effective response to tSZ cross-spectra
        that do not directly contain the matter profile, e.g. P_gy. Defaults to False.

    Notes
    -----
    - P_mm, P_gm, and P_ym are profile-level baryonified when
      use_baryonification=True because the matter profile switches from NFW to DMB.
    - P_gg, P_ge, and P_gy do not directly use probe 0/1 matter profiles in their
      definitions, so applying S_bary to them is an additional effective-response
      modelling choice controlled by separate flags.
    """

    PROBE_DMB = 0
    PROBE_NFW = 1
    PROBE_GALAXY = 2
    PROBE_Y = 3
    PROBE_ELECTRON = 4

    def __init__(
        self,
        sim_params_dict: dict,
        halo_params_dict: dict,
        analysis_dict: dict,
        other_params_dict: dict,
        Profiles_obj=None,
    ):
        if Profiles_obj is None:
            super().__init__(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
        else:
            self.__dict__.update(Profiles_obj.__dict__)

        self._initialise_baryonification_options()

        # ------------------------------------------------------------------
        # FFTLog transform of real-space profiles
        # ------------------------------------------------------------------
        xi2P_obj = xi2P(self.r_array, nx=self.nr, lowring=True)

        self.k_mcfit, uk_dmb = xi2P_obj(
            self.rho_dmb_mat / self.Mtot_mat[None, :, :], axis=0, extrap=False
        )
        self.uk_dmb_tointp = jnp.array(uk_dmb)

        self.k_mcfit, uk_nfw = xi2P_obj(
            self.rho_nfw_mat / self.Mtot_mat[None, :, :], axis=0, extrap=False
        )
        self.uk_nfw_tointp = jnp.array(uk_nfw)

        if self.model_galaxies:
            self.k_mcfit, uk_clm = xi2P_obj(
                self.rho_clm_mat / self.Mclm_mat[-1, :, :][None, :, :],
                axis=0,
                extrap=False,
            )
            self.uk_clm_tointp = jnp.array(uk_clm)

            self.k_mcfit, uk_ne = xi2P_obj(
                self.ne_mat / self.ne_mat_norm[-1, :, :][None, :, :],
                axis=0,
                extrap=False,
            )
            self.uk_ne_tointp = jnp.array(uk_ne)
        else:
            self.uk_clm_tointp = jnp.zeros((1, 1, 1))
            self.uk_ne_tointp = jnp.zeros((1, 1, 1))

        if self.model_tSZ:
            self.k_mcfit, uk_y = xi2P_obj(self.y3d_mat, axis=0, extrap=False)
            self.uk_y_tointp = jnp.array(uk_y)
        else:
            self.uk_y_tointp = jnp.zeros((1, 1, 1))

        # ------------------------------------------------------------------
        # Interpolate Fourier profiles on the target k grid
        # ------------------------------------------------------------------
        vmapped_uk_interp = get_vmapped_func_warg(self.get_uk_from_interp_Pk, 2, 3)

        self.uk_dmb = vmapped_uk_interp(
            jnp.arange(self.nz), jnp.arange(self.nM), self.PROBE_DMB
        ).T
        self.uk_nfw = vmapped_uk_interp(
            jnp.arange(self.nz), jnp.arange(self.nM), self.PROBE_NFW
        ).T

        if self.model_tSZ:
            self.uk_y = vmapped_uk_interp(
                jnp.arange(self.nz), jnp.arange(self.nM), self.PROBE_Y
            ).T
        else:
            self.uk_y = jnp.zeros((1, 1, 1))

        if self.model_galaxies:
            self.uk_clm = vmapped_uk_interp(
                jnp.arange(self.nz), jnp.arange(self.nM), self.PROBE_GALAXY
            ).T
            self.nbarz = jsi.trapezoid(
                self.hmf_Mz_mat * (self.Ncen_mat + self.Nsat_mat),
                jnp.log(self.M_array),
                axis=-1,
            )
            self.ukg_cross = (
                self.Ncen_mat[None, :, :] + self.Nsat_mat[None, :, :] * self.uk_clm
            ) / self.nbarz[None, :, None]

            ukg_auto_arg = jnp.clip(
                jnp.nan_to_num(
                    2.0 * self.Ncen_mat[None, :, :] * self.Nsat_mat[None, :, :] * self.uk_clm
                    + (self.Nsat_mat[None, :, :] * self.uk_clm) ** 2
                ),
                1e-10,
                2e4,
            )
            self.ukg_auto_sqr = ukg_auto_arg / (self.nbarz[None, :, None] ** 2)

            self.uk_ne = vmapped_uk_interp(
                jnp.arange(self.nz), jnp.arange(self.nM), self.PROBE_ELECTRON
            ).T
        else:
            self.uk_clm = jnp.zeros((1, 1, 1))
            self.ukg_cross = jnp.zeros((1, 1, 1))
            self.ukg_auto_sqr = jnp.zeros((1, 1, 1))
            self.uk_ne = jnp.zeros((1, 1, 1))

        # ------------------------------------------------------------------
        # Halofit nonlinear matter power
        # ------------------------------------------------------------------
        if self.symbolic_pk:
            vmap_func = vmap(symbolic_pkhalofit, (None, None, None, None, None, None, None, None, 0))
            self.phfit_kz_mat = vmap_func(
                self.kPk_array,
                self.plin_kz_mat,
                self.Om0,
                self.cosmo_params["Ob0"],
                self.h,
                self.cosmo_params["ns"],
                self.cosmo_params["sigma8"],
                self.z_array,
                jnp.arange(self.nz),
            ).T
        else:
            hfit_params = vmap(halofit_parameters, (None, 0))(
                self.cosmo_jax, self.scale_fac_a_array
            ).T
            self.phfit_kz_mat = vmap(nonlinear_matter_power, (None, None, 0, None, None, None))(
                self.cosmo_jax,
                self.kPk_array,
                self.scale_fac_a_array,
                self.plin_kz_mat,
                hfit_params,
                self.scale_fac_a_array,
            ).T

        # ------------------------------------------------------------------
        # Large-scale effective biases
        # ------------------------------------------------------------------
        self.bias_Mz_mat = get_vmapped_func(self.get_bias_Mz, 2)(
            jnp.arange(self.nz), jnp.arange(self.nM)
        ).T

        vmapped_b_2h = get_vmapped_func_warg(self.get_b_2h, 2, 3)

        if self.do_corr_2h_mm:
            bm_largescales_2h = vmap(self.get_bm_largescales_2h)(jnp.arange(self.nz))
            bm_largescales_2h_mat = jnp.tile(bm_largescales_2h, (self.nk, 1))

            self.bm_dmb_2h = vmapped_b_2h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_DMB
            ).T
            self.bm_largescales_2h_mat_lt_Mmin = 1.0 - bm_largescales_2h_mat
            self.bm_dmb_kz_mat = self.bm_dmb_2h + self.bm_largescales_2h_mat_lt_Mmin

            self.bm_nfw_2h = vmapped_b_2h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_NFW
            ).T
            self.bm_nfw_kz_mat = self.bm_nfw_2h + self.bm_largescales_2h_mat_lt_Mmin
        else:
            self.bm_dmb_kz_mat = jnp.ones((self.nk, self.nz))
            self.bm_nfw_kz_mat = jnp.ones((self.nk, self.nz))

        if self.model_tSZ:
            self.by_kz_mat = vmapped_b_2h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_Y
            ).T
        else:
            self.by_kz_mat = None

        if self.model_galaxies:
            self.bg_kz_mat = vmapped_b_2h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_GALAXY
            ).T
            self.be_kz_mat = vmapped_b_2h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_ELECTRON
            ).T
            if self.do_corr_2h_mm:
                self.be_kz_mat = self.be_kz_mat + self.bm_largescales_2h_mat_lt_Mmin
        else:
            self.bg_kz_mat = None
            self.be_kz_mat = None

        # ------------------------------------------------------------------
        # Consistent matter-profile selector
        # ------------------------------------------------------------------
        if self.use_baryonification:
            active_matter_probe = self.PROBE_DMB
            active_bm_kz_mat = self.bm_dmb_kz_mat
        else:
            active_matter_probe = self.PROBE_NFW
            active_bm_kz_mat = self.bm_nfw_kz_mat

        # ------------------------------------------------------------------
        # 2-halo power spectra
        # ------------------------------------------------------------------
        self.Pmm_dmb_2h_kz_mat = self.bm_dmb_kz_mat * self.bm_dmb_kz_mat * self.plin_kz_mat
        self.Pmm_nfw_2h_kz_mat = self.bm_nfw_kz_mat * self.bm_nfw_kz_mat * self.plin_kz_mat
        self.Pmm_active_2h_kz_mat = active_bm_kz_mat * active_bm_kz_mat * self.plin_kz_mat

        if self.model_tSZ:
            self.Pym_2h_kz_mat = active_bm_kz_mat * self.by_kz_mat * self.plin_kz_mat
            self.Pym_dmb_2h_kz_mat = self.bm_dmb_kz_mat * self.by_kz_mat * self.plin_kz_mat
            self.Pym_nfw_2h_kz_mat = self.bm_nfw_kz_mat * self.by_kz_mat * self.plin_kz_mat

        if self.model_galaxies:
            self.Pge_2h_kz_mat = self.bg_kz_mat * self.be_kz_mat * self.plin_kz_mat
            self.Pgm_2h_kz_mat = self.bg_kz_mat * active_bm_kz_mat * self.plin_kz_mat
            self.Pgm_dmb_2h_kz_mat = self.bg_kz_mat * self.bm_dmb_kz_mat * self.plin_kz_mat
            self.Pgm_nfw_2h_kz_mat = self.bg_kz_mat * self.bm_nfw_kz_mat * self.plin_kz_mat
            self.Pgy_2h_kz_mat = self.by_kz_mat * self.bg_kz_mat * self.plin_kz_mat
            self.Pgg_2h_kz_mat = self.bg_kz_mat * self.bg_kz_mat * self.plin_kz_mat

        # ------------------------------------------------------------------
        # 1-halo power spectra
        # ------------------------------------------------------------------
        vmapped_P_1h = get_vmapped_func_warg(self.get_P_1h, 2, 4)

        self.Pmm_dmb_1h_kz_mat = vmapped_P_1h(
            jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_DMB, self.PROBE_DMB
        ).T
        self.Pmm_nfw_1h_kz_mat = vmapped_P_1h(
            jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_NFW, self.PROBE_NFW
        ).T
        self.Pmm_active_1h_kz_mat = vmapped_P_1h(
            jnp.arange(self.nk), jnp.arange(self.nz), active_matter_probe, active_matter_probe
        ).T

        if self.model_tSZ:
            self.Pym_1h_kz_mat = vmapped_P_1h(
                jnp.arange(self.nk), jnp.arange(self.nz), active_matter_probe, self.PROBE_Y
            ).T
            self.Pym_dmb_1h_kz_mat = vmapped_P_1h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_DMB, self.PROBE_Y
            ).T
            self.Pym_nfw_1h_kz_mat = vmapped_P_1h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_NFW, self.PROBE_Y
            ).T

        if self.model_galaxies:
            self.Pge_1h_kz_mat = vmapped_P_1h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_GALAXY, self.PROBE_ELECTRON
            ).T
            self.Pgm_1h_kz_mat = vmapped_P_1h(
                jnp.arange(self.nk), jnp.arange(self.nz), active_matter_probe, self.PROBE_GALAXY
            ).T
            self.Pgm_dmb_1h_kz_mat = vmapped_P_1h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_DMB, self.PROBE_GALAXY
            ).T
            self.Pgm_nfw_1h_kz_mat = vmapped_P_1h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_NFW, self.PROBE_GALAXY
            ).T
            self.Pgy_1h_kz_mat = vmapped_P_1h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_Y, self.PROBE_GALAXY
            ).T
            self.Pgg_1h_kz_mat = vmapped_P_1h(
                jnp.arange(self.nk), jnp.arange(self.nz), self.PROBE_GALAXY, self.PROBE_GALAXY
            ).T

        # ------------------------------------------------------------------
        # Total power spectra and response factors
        # ------------------------------------------------------------------
        self.Pmm_nfw_tot_mat = self.Pmm_nfw_1h_kz_mat + self.Pmm_nfw_2h_kz_mat
        self.Pmm_dmb_tot_mat = self.Pmm_dmb_1h_kz_mat + self.Pmm_dmb_2h_kz_mat
        self.Pmm_active_hm_tot_mat = self.Pmm_active_1h_kz_mat + self.Pmm_active_2h_kz_mat

        eps = 1e-30

        # This is the paper's baryonic matter response/suppression.
        self.Pmm_bary_supp_tot_mat = self.Pmm_dmb_tot_mat / jnp.clip(
            self.Pmm_nfw_tot_mat, eps, jnp.inf
        )

        # This is the old ``Pmm_sup_tot_mat`` quantity: not baryonic suppression,
        # but a halo-model-to-Halofit normalisation.
        self.Pmm_hm_to_halofit_norm_mat = self.phfit_kz_mat / jnp.clip(
            self.Pmm_nfw_tot_mat, eps, jnp.inf
        )
        self.Pmm_sup_tot_mat = self.Pmm_hm_to_halofit_norm_mat  # backward compatibility

        if self.use_baryonification:
            self.Pmm_bary_response_mat = self.Pmm_bary_supp_tot_mat
        else:
            self.Pmm_bary_response_mat = jnp.ones_like(self.Pmm_nfw_tot_mat)

        if self.apply_hm_to_halofit_norm:
            hm_norm_mat = self.Pmm_hm_to_halofit_norm_mat
        else:
            hm_norm_mat = jnp.ones_like(self.Pmm_nfw_tot_mat)

        # Matter power.
        # If the requested matter baseline is Halofit, use Halofit and optionally
        # apply S_bary. Otherwise use the active halo-model matter spectrum, with
        # optional halo-model-to-Halofit normalisation.
        if self.model_matter == "halofit":
            self.Pmm_tot_mat = self.phfit_kz_mat * self.Pmm_bary_response_mat
        else:
            self.Pmm_tot_mat = self.Pmm_active_hm_tot_mat * hm_norm_mat

        # tSZ-matter cross spectrum. This directly contains matter, so the DMB/NFW
        # choice has already been applied through active_matter_probe/active_bm_kz_mat.
        if self.model_tSZ:
            self.Pym_hm_tot_mat = self._combine_1h_2h_transition(
                self.Pym_1h_kz_mat, self.Pym_2h_kz_mat, self.alpha_ky
            )
            self.Pym_tot_mat = self.Pym_hm_tot_mat * hm_norm_mat

            # Optional additional effective response. Off by default to avoid
            # double-counting matter-profile baryonification in P_ym.
            if self.apply_bary_response_to_ym:
                self.Pym_tot_mat = self.Pym_tot_mat * self.Pmm_bary_response_mat

        if self.model_galaxies:
            # Galaxy-electron. This does not directly switch DMB/NFW matter profiles.
            ge_response = self.Pmm_bary_response_mat if self.apply_bary_response_to_ge else 1.0
            self.Pge_hm_tot_mat = self.Pge_1h_kz_mat + self.Pge_2h_kz_mat
            self.Pge_tot_mat = self.Pge_hm_tot_mat * hm_norm_mat * ge_response

            # Galaxy-matter. This directly contains matter, so the profile switch is
            # already included in Pgm_1h/Pgm_2h.
            self.Pgm_hm_tot_mat = self.Pgm_1h_kz_mat + self.Pgm_2h_kz_mat
            self.Pgm_tot_mat = self.Pgm_hm_tot_mat * hm_norm_mat

            # Keep an explicit NFW galaxy-matter baseline for diagnostics.
            self.Pgm_nfw_hm_tot_mat = self.Pgm_nfw_1h_kz_mat + self.Pgm_nfw_2h_kz_mat
            self.Pgm_nfw_tot_mat = self.Pgm_nfw_hm_tot_mat * hm_norm_mat

            # Galaxy-y. This does not directly use matter probe 0/1, so S_bary is an
            # optional effective response.
            gy_response = self.Pmm_bary_response_mat if self.apply_bary_response_to_gy else 1.0
            self.Pgy_hm_tot_mat = self._combine_1h_2h_transition(
                self.Pgy_1h_kz_mat, self.Pgy_2h_kz_mat, self.alpha_gy
            )
            self.Pgy_tot_mat = self.Pgy_hm_tot_mat * hm_norm_mat * gy_response

            # Galaxy-galaxy. This is HOD/galaxy-profile based. Applying S_bary here
            # is an effective modelling choice, disabled by default.
            gg_response = self.Pmm_bary_response_mat if self.apply_bary_response_to_gg else 1.0
            self.Pgg_hm_tot_mat = self.Pgg_1h_kz_mat + self.Pgg_2h_kz_mat
            self.Pgg_tot_mat = self.Pgg_hm_tot_mat * hm_norm_mat * gg_response

    # ----------------------------------------------------------------------
    # Configuration helpers
    # ----------------------------------------------------------------------
    def _initialise_baryonification_options(self):
        """Initialise baryonification and normalisation switches."""

        # Backward-compatible default:
        # - old model_matter == 'halofit' behaved as no baryonic correction in P_mm
        # - old model_matter != 'halofit' behaved as Halofit * DMB/NFW in P_mm
        default_use_baryonification = self.model_matter != "halofit"

        self.use_baryonification = bool(
            getattr(self, "use_baryonification", default_use_baryonification)
        )

        # Keeps the old behaviour where many halo-model spectra were multiplied by
        # phfit / Pmm_nfw_hm. Set False if you want pure halo-model spectra.
        self.apply_hm_to_halofit_norm = bool(
            getattr(self, "apply_hm_to_halofit_norm", True)
        )

        # These are intentionally separate because these spectra do not all contain
        # the matter profile directly. Defaults are conservative to avoid accidental
        # double-counting.
        self.apply_bary_response_to_ym = bool(
            getattr(self, "apply_bary_response_to_ym", False)
        )
        self.apply_bary_response_to_ge = bool(
            getattr(self, "apply_bary_response_to_ge", False)
        )
        self.apply_bary_response_to_gy = bool(
            getattr(self, "apply_bary_response_to_gy", False)
        )
        self.apply_bary_response_to_gg = bool(
            getattr(self, "apply_bary_response_to_gg", False)
        )

    @staticmethod
    def _combine_1h_2h_transition(P_1h, P_2h, alpha):
        """Smoothly combine 1-halo and 2-halo terms."""
        return (P_1h**alpha + P_2h**alpha) ** (1.0 / alpha)

    # ----------------------------------------------------------------------
    # JITted halo-model kernels
    # ----------------------------------------------------------------------
    @partial(jit, static_argnums=(0,))
    def get_uk_from_interp_Pk(self, jz, jM, probe):
        """Compute interpolated u(k) for a given probe."""

        conditions = [
            (probe == self.PROBE_DMB, jnp.clip(self.uk_dmb_tointp[:, jz, jM], 1e-30, 1.0)),
            (probe == self.PROBE_NFW, jnp.clip(self.uk_nfw_tointp[:, jz, jM], 1e-30, 1.0)),
            (probe == self.PROBE_GALAXY, jnp.clip(self.uk_clm_tointp[:, jz, jM], 1e-30, 1.0)),
            (probe == self.PROBE_Y, self.uk_y_tointp[:, jz, jM]),
            (probe == self.PROBE_ELECTRON, self.uk_ne_tointp[:, jz, jM]),
        ]

        uk_val = jnp.nan
        for condition, value in conditions:
            uk_val = jnp.where(condition, value, uk_val)

        return jnp.exp(
            jnp.interp(
                jnp.log(self.kPk_array),
                jnp.log(self.k_mcfit),
                jnp.log(jnp.clip(uk_val, 1e-30, jnp.inf)),
            )
        )

    @partial(jit, static_argnums=(0,))
    def get_bias_Mz(self, jz, jM, mdef_delta=200):
        """Tinker et al. halo-bias function."""
        sigma = self.sigma_Mz_mat[jz, jM]
        delta_c = constants.DELTA_COLLAPSE
        nu = delta_c / sigma

        z = self.z_array[jz]
        rho_threshold = mdef_delta * self.get_rho_c(z)
        Delta = rho_threshold / self.get_rho_m(z)
        y = jnp.log10(Delta)

        A = 1.0 + 0.24 * y * jnp.exp(-1.0 * (4.0 / y) ** 4)
        a = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * jnp.exp(-1.0 * (4.0 / y) ** 4)
        c = 2.4

        bias = (
            1.0
            - A * nu**a / (nu**a + constants.DELTA_COLLAPSE**a)
            + B * nu**b
            + C * nu**c
        )
        return bias

    @partial(jit, static_argnums=(0,))
    def compute_ukz(self, jk, jz, probe):
        """
        Compute the field-specific integrand weight u_X(k, M, z).

        Probe IDs:
            0 -> DMB matter profile
            1 -> NFW matter profile
            2 -> galaxies
            3 -> Compton-y
            4 -> electron number density
        """
        conditions = [
            (
                probe == self.PROBE_DMB,
                (self.Mtot_mat[jz, :] * self.uk_dmb[jk, jz, :]) / self.rhom_0,
            ),
            (
                probe == self.PROBE_NFW,
                (self.Mtot_mat[jz, :] * self.uk_nfw[jk, jz, :]) / self.rhom_0,
            ),
            (probe == self.PROBE_GALAXY, self.ukg_cross[jk, jz, :]),
            (probe == self.PROBE_Y, self.uk_y[jk, jz, :]),
            (
                probe == self.PROBE_ELECTRON,
                (self.Mtot_mat[jz, :] * self.uk_ne[jk, jz, :]) / self.rhom_0,
            ),
        ]

        ukz = jnp.nan
        for condition, value in conditions:
            ukz = jnp.where(condition, value, ukz)
        return ukz

    @partial(jit, static_argnums=(0,))
    def get_b_2h(self, jk, jz, probe):
        """Compute the 2-halo effective bias for a field."""
        ukz = self.compute_ukz(jk, jz, probe)
        dndlnM_z = self.hmf_Mz_mat[jz, :]
        fx = ukz * dndlnM_z * self.bias_Mz_mat[jz, :]
        return jsi.trapezoid(fx, x=jnp.log(self.M_array))

    @partial(jit, static_argnums=(0,))
    def get_bm_largescales_2h(self, jz):
        """Compute the large-scale limit of the matter 2-halo integral."""
        ukz_intc = self.Mtot_mat[jz, :]
        dndlnM_z = self.hmf_Mz_mat[jz, :]
        fx = ukz_intc * dndlnM_z * self.bias_Mz_mat[jz, :] / self.rhom_0
        return jsi.trapezoid(fx, x=jnp.log(self.M_array))

    @partial(jit, static_argnums=(0,))
    def get_P_1h(self, jk, jz, probe1, probe2):
        """Compute the 1-halo power spectrum for two probes."""
        ukz1 = self.compute_ukz(jk, jz, probe1)
        ukz2 = self.compute_ukz(jk, jz, probe2)

        ukz_sqr = jnp.where(
            jnp.logical_and(probe1 == self.PROBE_GALAXY, probe2 == self.PROBE_GALAXY),
            self.ukg_auto_sqr[jk, jz, :],
            ukz1 * ukz2,
        )

        dndlnM_z = self.hmf_Mz_mat[jz, :]
        return jsi.trapezoid(ukz_sqr * dndlnM_z, x=jnp.log(self.M_array))