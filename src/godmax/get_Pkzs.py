from .get_radial_profiles import Profiles
from .base_class import get_vmapped_func, get_vmapped_func_warg
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from .helpers.jax_cosmo_power import halofit_parameters, nonlinear_matter_power
import jax.scipy.integrate as jsi
from .helpers import constants
from .mcfitjax.cosmology_jax import xi2P
from .matter_pk_symbolic import *


class get_Pkz(Profiles):
    """
    Compute the total power spectra for matter, tSZ, and galaxy contributions over k and z.
    Sets the class attributes (e.g., Pmm_tot_mat, Pym_tot_mat, Pgg_tot_mat) depending on
    the chosen models (matter, tSZ, galaxy). Includes 1-halo and 2-halo terms, and optionally
    applies 1-halo to 2-halo transition regime corrections.
    
    Returns:
        None
    """    
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

        # Do the FFTlog transform of the real-space profiles:
        xi2P_obj = (xi2P(self.r_array, nx=self.nr,lowring=True))
        # Grid-consistent normalisation: the profiles are normalised on a per-halo
        # [0.01,16]*r200c window, but FFTLog integrates them on the FIXED r_array grid
        # [rmin,rmax] Mpc. When 16*r200c != rmax the grid integral != Mtot, which makes
        # u(k->0) != 1 (e.g. low-M/low-z haloes leak the gas tail and reach u(0)~1.8).
        # Dividing by the mass actually enclosed within the FFT grid forces u(k->0)=1,
        # exactly as uk_clm is already normalised by Mclm_mat[-1] below.
        _fourpi_r2 = 4.0 * jnp.pi * self.r_array[:, None, None] ** 2
        self.Mdmb_grid = jnp.trapezoid(_fourpi_r2 * self.rho_dmb_mat, self.r_array, axis=0)
        self.Mnfw_grid = jnp.trapezoid(_fourpi_r2 * self.rho_nfw_mat, self.r_array, axis=0)
        self.k_mcfit, uk_dmb = xi2P_obj(self.rho_dmb_mat / self.Mdmb_grid[None, :, :], axis=0, extrap=False)
        self.uk_dmb_tointp = jnp.array(uk_dmb)
        self.k_mcfit, uk_nfw = xi2P_obj(self.rho_nfw_mat / self.Mnfw_grid[None, :, :], axis=0, extrap=False)
        self.uk_nfw_tointp = jnp.array(uk_nfw)

        if self.model_galaxies:
            self.k_mcfit, uk_clm = xi2P_obj(self.rho_clm_mat / self.Mclm_mat[-1, :, :][None, :, :], axis=0, extrap=False)
            self.uk_clm_tointp = jnp.array(uk_clm)
            self.k_mcfit, uk_ne = xi2P_obj(self.ne_mat / self.ne_mat_norm[-1, :, :][None, :, :], axis=0, extrap=False)
            # self.k_mcfit, uk_ne = xi2P_obj(self.ne_mat, axis=0, extrap=False)
            self.uk_ne_tointp = jnp.array(uk_ne)
        else: self.uk_clm_tointp, self.uk_ne_tointp = jnp.zeros((1,1,1)), jnp.zeros((1,1,1))
                        
        if self.model_tSZ:
            self.k_mcfit, uk_y = xi2P_obj(self.y3d_mat, axis=0, extrap=False)
            self.uk_y_tointp = jnp.array(uk_y)
        else: self.uk_y_tointp = jnp.zeros((1,1,1)) 
                       

        # Get the Fourier profiles uk's in the interpolated k array:
        vmapped_func = get_vmapped_func_warg(self.get_uk_from_interp_Pk, 2, 3)
        self.uk_dmb = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 0).T
        self.uk_nfw = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 1).T
        if self.model_tSZ:
            self.uk_y = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 3).T
        # else: self.uk_y = jnp.zeros((self.nk, self.nz, self.nM))
        else: self.uk_y = jnp.zeros((1,1,1))
        if self.model_galaxies:
            self.uk_clm = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 2).T
            self.nbarz = jsi.trapezoid(self.hmf_Mz_mat * (self.Ncen_mat + self.Nsat_mat), jnp.log(self.M_array), axis=-1)
            self.ukg_cross = (self.Ncen_mat[None,:,:] + self.Nsat_mat[None,:,:] * self.uk_clm)/self.nbarz[None,:,None]
            ukg_auto_arg = jnp.clip(jnp.nan_to_num(2 * self.Ncen_mat[None,:,:] * self.Nsat_mat[None,:,:] * self.uk_clm + (self.Nsat_mat[None,:,:] * self.uk_clm)**2), 1e-10, 2e4)
            self.ukg_auto_sqr = (ukg_auto_arg)/(self.nbarz[None,:,None] ** 2)
            self.uk_ne = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 4).T
        # else: self.uk_clm, self.ukg_cross, self.ukg_auto_sqr, self.uk_ne = jnp.zeros((self.nk, self.nz, self.nM)), jnp.zeros((self.nk, self.nz, self.nM)), jnp.zeros((self.nk, self.nz, self.nM)), jnp.zeros((self.nk, self.nz, self.nM))
        else: self.uk_clm, self.ukg_cross, self.ukg_auto_sqr, self.uk_ne = jnp.zeros((1,1,1)), jnp.zeros((1,1,1)), jnp.zeros((1,1,1)), jnp.zeros((1,1,1))

        # Get the halofit power spectra:
        if self.symbolic_pk:
            vmap_func = vmap(symbolic_pkhalofit,(None, None, None, None, None, None, None, None, 0))
            self.phfit_kz_mat = vmap_func(self.kPk_array, self.plin_kz_mat, self.Om0, self.cosmo_params['Ob0'], self.h, self.cosmo_params['ns'], self.cosmo_params['sigma8'], self.z_array, jnp.arange(self.nz)).T
        else:
            hfit_params = vmap(halofit_parameters,(None, 0))(self.cosmo_jax, self.scale_fac_a_array).T
            self.phfit_kz_mat = vmap(nonlinear_matter_power,(None, None, 0, None, None, None))(self.cosmo_jax, self.kPk_array, self.scale_fac_a_array, self.plin_kz_mat, hfit_params, self.scale_fac_a_array).T

        # Get the large-scale bias of the fields:
        self.bias_Mz_mat = get_vmapped_func(self.get_bias_Mz, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T

        vmapped_func = get_vmapped_func_warg(self.get_b_2h, 2, 3)
        if self.do_corr_2h_mm:
            bm_largescales_2h = vmap(self.get_bm_largescales_2h)(jnp.arange((self.nz)))
            bm_largescales_2h_mat = jnp.tile(bm_largescales_2h, ((self.nk), 1))

            self.bm_dmb_2h = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 0).T
            self.bm_largescales_2h_mat_lt_Mmin = 1. - bm_largescales_2h_mat
            self.bm_dmb_kz_mat = self.bm_dmb_2h + self.bm_largescales_2h_mat_lt_Mmin

            bm_nfw_2h = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 1).T
            self.bm_nfw_kz_mat = bm_nfw_2h + self.bm_largescales_2h_mat_lt_Mmin   

        else:
            self.bm_dmb_kz_mat = jnp.ones((len(self.nk), self.nz))
            self.bm_nfw_kz_mat = jnp.ones((len(self.nk), self.nz))

        if self.model_tSZ:
            self.by_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 3).T
        else: self.by_kz_mat = None
        if self.model_galaxies:
            self.bg_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 2).T
            self.be_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 4).T
            if self.do_corr_2h_mm:
                self.be_kz_mat = self.be_kz_mat + self.bm_largescales_2h_mat_lt_Mmin
        else: self.bg_kz_mat, self.be_kz_mat = None, None

        # Get the 2-halo power:
        self.Pmm_dmb_2h_kz_mat = self.bm_dmb_kz_mat * self.bm_dmb_kz_mat * self.plin_kz_mat
        self.Pmm_nfw_2h_kz_mat = self.bm_nfw_kz_mat * self.bm_nfw_kz_mat * self.plin_kz_mat
        if self.model_tSZ:
            self.Pym_2h_kz_mat = self.bm_dmb_kz_mat * self.by_kz_mat * self.plin_kz_mat
        if self.model_galaxies:
            self.Pge_2h_kz_mat = self.bg_kz_mat * self.be_kz_mat * self.plin_kz_mat
            self.Pgm_2h_kz_mat = self.bg_kz_mat * self.bm_dmb_kz_mat * self.plin_kz_mat
            self.Pgm_nfw_2h_kz_mat = self.bg_kz_mat * self.bm_nfw_kz_mat * self.plin_kz_mat
            self.Pgy_2h_kz_mat = self.by_kz_mat * self.bg_kz_mat * self.plin_kz_mat
            self.Pgg_2h_kz_mat = self.bg_kz_mat * self.bg_kz_mat * self.plin_kz_mat

        # Get the 1-halo power:
        vmapped_func = get_vmapped_func_warg(self.get_P_1h, 2, 4)
        self.Pmm_dmb_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 0, 0).T
        self.Pmm_nfw_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 1, 1).T
        if self.model_tSZ:
            self.Pym_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 0, 3).T
        if self.model_galaxies:
            self.Pge_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 2, 4).T
            self.Pgm_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 0, 2).T
            self.Pgm_nfw_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 1, 2).T
            self.Pgy_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 3, 2).T
            self.Pgg_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 2, 2).T

        # Get the total power:
        self.Pmm_nfw_tot_mat = self.Pmm_nfw_1h_kz_mat + self.Pmm_nfw_2h_kz_mat
        self.Pmm_dmb_tot_mat = self.Pmm_dmb_1h_kz_mat + self.Pmm_dmb_2h_kz_mat        
        self.Pmm_sup_tot_mat = self.phfit_kz_mat / self.Pmm_nfw_tot_mat
        if self.model_matter == 'halofit':
            self.Pmm_tot_mat = self.phfit_kz_mat
        else:
            self.Pmm_tot_mat = (self.Pmm_dmb_tot_mat) * self.Pmm_sup_tot_mat
        if self.model_tSZ:
            self.Pym_tot_mat = ((self.Pym_1h_kz_mat)**(self.alpha_ky) + (self.Pym_2h_kz_mat)**(self.alpha_ky))**(1/self.alpha_ky)
            if self.tSZ_transition_model == 'response':
                self.Pym_tot_mat = self.Pym_tot_mat * self.Pmm_sup_tot_mat
        if self.model_galaxies:
            self.Pge_tot_mat = (self.Pge_1h_kz_mat + self.Pge_2h_kz_mat) * self.Pmm_sup_tot_mat
            self.Pgm_tot_mat = (self.Pgm_1h_kz_mat + self.Pgm_2h_kz_mat) * self.Pmm_sup_tot_mat
            self.Pgm_nfw_tot_mat = (self.Pgm_nfw_1h_kz_mat + self.Pgm_nfw_2h_kz_mat) * self.Pmm_sup_tot_mat
            self.Pgy_tot_mat = ((self.Pgy_1h_kz_mat)**(self.alpha_gy) + (self.Pgy_2h_kz_mat)**(self.alpha_gy))**(1/self.alpha_gy)
            if self.tSZ_transition_model == 'response':
                self.Pgy_tot_mat = self.Pgy_tot_mat * self.Pmm_sup_tot_mat
            self.Pgg_tot_mat = (self.Pgg_1h_kz_mat + self.Pgg_2h_kz_mat) * self.Pmm_sup_tot_mat

        # --- no-baryonification: start ---
        # When baryonification=False, replace baryonified power spectra with NFW-only baselines:
        #   gκ   : matter leg  rho_dmb -> rho_nfw  (Pgm_nfw already computed above)
        #   κy/gy/yy: pressure leg  rho_gas,(M_dmb) -> (Ob/Om)*rho_nfw, M_nfw  (via y3d_nfw_mat)
        if not self.baryonification:
            # gκ: swap to NFW matter leg (no tSZ dependence)
            if self.model_galaxies:
                self.Pgm_tot_mat = self.Pgm_nfw_tot_mat
                self.Pgm_1h_kz_mat = self.Pgm_nfw_1h_kz_mat
                self.Pgm_2h_kz_mat = self.Pgm_nfw_2h_kz_mat

            # κy / gy / yy: FFTlog the NFW-pressure profile and rebuild all y power spectra
            if self.model_tSZ:
                _, uk_y_nfw_raw = xi2P_obj(self.y3d_nfw_mat, axis=0, extrap=False)
                uk_y_nfw_tointp = jnp.array(uk_y_nfw_raw)

                # Interpolate from k_mcfit grid to kPk_array for all (jz, jM)
                log_k = jnp.log(self.k_mcfit)
                log_kPk = jnp.log(self.kPk_array)
                # uk_y_nfw_tointp shape: [nk_mcfit, nz, nM]
                # .T -> [nM, nz, nk_mcfit]; vmap(vmap(f)) -> [nM, nz, nk]; .transpose -> [nk, nz, nM]
                def _interp_uk(uk_vec):
                    # Old: clamp at k_mcfit[0] → spurious low-k ratio offset
                    # return jnp.exp(jnp.interp(log_kPk, log_k, jnp.log(jnp.clip(uk_vec, 1e-30, jnp.inf))))
                    log_uv        = jnp.log(jnp.clip(uk_vec, 1e-30, jnp.inf))
                    log_ui        = jnp.interp(log_kPk, log_k, log_uv)
                    slope_left    = (log_uv[1] - log_uv[0]) / (log_k[1] - log_k[0])
                    log_ue        = jnp.minimum(log_uv[0] + slope_left * (log_kPk - log_k[0]), 0.0)
                    return jnp.exp(jnp.where(log_kPk < log_k[0], log_ue, log_ui))
                uk_y_nfw = vmap(vmap(_interp_uk))(uk_y_nfw_tointp.T).transpose(2, 1, 0)  # [nk, nz, nM]

                # 2h bias for NFW-pressure y
                def _by_nfw(jk, jz):
                    return jsi.trapezoid(
                        uk_y_nfw[jk, jz, :] * self.hmf_Mz_mat[jz, :] * self.bias_Mz_mat[jz, :],
                        x=jnp.log(self.M_array)
                    )
                by_nfw_kz_mat = vmap(vmap(_by_nfw, in_axes=(None, 0)), in_axes=(0, None))(
                    jnp.arange(self.nk), jnp.arange(self.nz)
                )  # [nk, nz]

                # 1h integrals
                def _P1h_ym_nfw(jk, jz):
                    ukz_m = (self.Mtot_mat[jz, :] * self.uk_nfw[jk, jz, :]) / self.rhom_0
                    return jsi.trapezoid(ukz_m * uk_y_nfw[jk, jz, :] * self.hmf_Mz_mat[jz, :], x=jnp.log(self.M_array))
                Pym_nfw_1h = vmap(vmap(_P1h_ym_nfw, in_axes=(None, 0)), in_axes=(0, None))(
                    jnp.arange(self.nk), jnp.arange(self.nz)
                )  # [nk, nz]

                # 2h term
                Pym_nfw_2h = self.bm_nfw_kz_mat * by_nfw_kz_mat * self.plin_kz_mat
                # (Pyy 1h/2h are NOT computed here — get_Cls recomputes them via
                #  the overridden self.uk_y and self.by_kz_mat below)

                # Total (same transition model as baryonified)
                Pym_nfw_tot = ((Pym_nfw_1h)**(self.alpha_ky) + (Pym_nfw_2h)**(self.alpha_ky))**(1 / self.alpha_ky)
                if self.tSZ_transition_model == 'response':
                    Pym_nfw_tot = Pym_nfw_tot * self.Pmm_sup_tot_mat

                # Override κy, yy attributes — get_Cls picks these up automatically
                self.Pym_tot_mat = Pym_nfw_tot
                self.Pym_1h_kz_mat = Pym_nfw_1h
                self.Pym_2h_kz_mat = Pym_nfw_2h
                self.uk_y = uk_y_nfw          # used by get_Cls for Pyy 1h via get_P_1h(3,3)
                self.by_kz_mat = by_nfw_kz_mat  # used by get_Cls for Pyy 2h

                if self.model_galaxies:
                    def _P1h_gy_nfw(jk, jz):
                        return jsi.trapezoid(
                            self.ukg_cross[jk, jz, :] * uk_y_nfw[jk, jz, :] * self.hmf_Mz_mat[jz, :],
                            x=jnp.log(self.M_array)
                        )
                    Pgy_nfw_1h = vmap(vmap(_P1h_gy_nfw, in_axes=(None, 0)), in_axes=(0, None))(
                        jnp.arange(self.nk), jnp.arange(self.nz)
                    )  # [nk, nz]
                    Pgy_nfw_2h = self.bg_kz_mat * by_nfw_kz_mat * self.plin_kz_mat
                    Pgy_nfw_tot = ((Pgy_nfw_1h)**(self.alpha_gy) + (Pgy_nfw_2h)**(self.alpha_gy))**(1 / self.alpha_gy)
                    if self.tSZ_transition_model == 'response':
                        Pgy_nfw_tot = Pgy_nfw_tot * self.Pmm_sup_tot_mat
                    self.Pgy_tot_mat = Pgy_nfw_tot
                    self.Pgy_1h_kz_mat = Pgy_nfw_1h
                    self.Pgy_2h_kz_mat = Pgy_nfw_2h
        # --- no-baryonification: end ---


    @partial(jit, static_argnums=(0,))
    def get_uk_from_interp_Pk(self, jz, jM, probe):
        '''Compute uk values based on the probe and interpolate over kPk_array.'''
        
        # Helper function to select uk_val based on the probe
        def compute_uk_val(probe):
            conditions = [
                (probe == 0, jnp.clip(self.uk_dmb_tointp[:, jz, jM], 1e-30, 1)),
                (probe == 1, jnp.clip(self.uk_nfw_tointp[:, jz, jM], 1e-30, 1)),
                (probe == 2, jnp.clip(self.uk_clm_tointp[:, jz, jM], 1e-30, 1)),
                (probe == 3, self.uk_y_tointp[:, jz, jM]),
                (probe == 4, self.uk_ne_tointp[:, jz, jM]),
            ]
            
            # Default value if no condition matches
            uk_val = jnp.nan
            for condition, value in conditions:
                uk_val = jnp.where(condition, value, uk_val)
            return uk_val

        # Compute uk_val based on the probe
        uk_val = compute_uk_val(probe)

        # Interpolate in log-log space; power-law extrapolate for k < k_mcfit[0]
        # toward uk=1 (correct k→0 limit for any normalised profile).
        # Old behaviour: clamp at k_mcfit[0] (caused spurious ratio≠1 at low k/ell)
        # return jnp.exp(
        #     jnp.interp(
        #         jnp.log(self.kPk_array),
        #         jnp.log(self.k_mcfit),
        #         jnp.log(jnp.clip(uk_val, 1e-30, jnp.inf))
        #     )
        # )
        log_k    = jnp.log(self.k_mcfit)
        log_kPk  = jnp.log(self.kPk_array)
        log_uk   = jnp.log(jnp.clip(uk_val, 1e-30, jnp.inf))

        log_uk_interp = jnp.interp(log_kPk, log_k, log_uk)

        # slope from first two FFTlog k-points; cap at 0 so uk ≤ 1
        slope_left    = (log_uk[1] - log_uk[0]) / (log_k[1] - log_k[0])
        log_uk_extrap = jnp.minimum(log_uk[0] + slope_left * (log_kPk - log_k[0]), 0.0)

        # return jnp.exp(jnp.where(log_kPk < log_k[0], log_uk_extrap, log_uk_interp))
        # Smooth blend around k_mcfit[0]
        # width is in log(k); 0.15 means roughly a 15%–20% transition width
        width = 0.15

        w = 0.5 * (1.0 + jnp.tanh((log_kPk - log_k[0]) / width))

        log_uk_matched = (1.0 - w) * log_uk_extrap + w * log_uk_interp

        return jnp.exp(log_uk_matched)
    
    @partial(jit, static_argnums=(0,))
    def get_bias_Mz(self, jz, jM, mdef_delta=200):
        '''Tinker 2010 bias function'''
        sigma = self.sigma_Mz_mat[jz, jM]
        delta_c = constants.DELTA_COLLAPSE
        nu = delta_c / sigma

        z = self.z_array[jz]    
        rho_treshold = mdef_delta * self.get_rho_c(z)
        Delta = rho_treshold / self.get_rho_m(z)
        y = jnp.log10(Delta)

        A = 1.0 + 0.24 * y * jnp.exp(-1.0 * (4.0 / y)**4)
        a = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * jnp.exp(-1.0 * (4.0 / y)**4)
        c = 2.4
        
        bias = 1.0 - A * nu**a / (nu**a + constants.DELTA_COLLAPSE**a) + B * nu**b + C * nu**c
        return bias

    @partial(jit, static_argnums=(0,))
    def compute_ukz(self, jk, jz, probe):
        """
        Compute the field-specific uk values for a given k and redshift bin.

        Depending on the probe type:
            0 -> dark matter-baryon (dmb)
            1 -> NFW halo model
            2 -> galaxies
            3 -> Compton y-parameter
            4 -> electron number density

        Args:
            jk (int): Index for the wavenumber (k).
            jz (int): Index for the redshift bin (z).
            probe (int): Integer code selecting which field to compute uk for.

        Returns:
            ukz (jax.numpy.DeviceArray): The dimensionless clustering amplitude or weighting function
                for the specified probe, evaluated at the given k and z.
        """        
        conditions = [
            (probe == 0, (self.Mtot_mat[jz, :] * self.uk_dmb[jk, jz, :]) / self.rhom_0),
            (probe == 1, (self.Mtot_mat[jz, :] * self.uk_nfw[jk, jz, :]) / self.rhom_0),
            (probe == 2, self.ukg_cross[jk, jz, :]),
            (probe == 3, self.uk_y[jk, jz, :]),
            (probe == 4, (self.Mtot_mat[jz, :] * self.uk_ne[jk, jz, :]) / self.rhom_0),
        ]
        
        # Default value if no condition matches
        ukz = jnp.nan
        for condition, value in conditions:
            ukz = jnp.where(condition, value, ukz)
        return ukz

    
    @partial(jit, static_argnums=(0,))
    def get_b_2h(self, jk, jz, probe):
        '''Function getting the 2halo effective bias of the fields'''
        ukz = self.compute_ukz(jk, jz, probe)
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        fx = ukz * dndlnM_z * self.bias_Mz_mat[jz, :]
        b_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))        
        return b_2h

    
    @partial(jit, static_argnums=(0,))
    def get_bm_largescales_2h(self, jz):
        '''Get the large scale limit of the above 2halo integral'''
        ukz_intc = self.Mtot_mat[jz, :]
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        fx = ukz_intc * dndlnM_z * self.bias_Mz_mat[jz,:] * (1/self.rhom_0)
        bmm_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return bmm_2h

    @partial(jit, static_argnums=(0,))
    def get_P_1h(self, jk, jz, probe1, probe2):
        """
        Compute the 1-halo power spectrum for the specified probes.

        This function calculates ukz values for two probes, handles the special case
        when both probes are galaxies (auto-squared), and integrates over the halo mass
        function to obtain the 1-halo contribution.

        Args:
            jk (int): Index for the wavenumber (k).
            jz (int): Index for the redshift bin (z).
            probe1 (int): Integer code selecting the first field.
            probe2 (int): Integer code selecting the second field.

        Returns:
            jax.numpy.DeviceArray: The 1-halo power spectrum for the selected probes.
        """

        # Compute ukz1 and ukz2
        ukz1 = self.compute_ukz(jk, jz, probe1)
        ukz2 = self.compute_ukz(jk, jz, probe2)
        # Handle the special case for both probes being 2 (auto-squared case)
        ukz_sqr = jnp.where(
            jnp.logical_and(probe1 == 2, probe2 == 2), self.ukg_auto_sqr[jk, jz, :], ukz1 * ukz2
        )
        # Compute P_1h using trapezoid integration
        dndlnM_z = self.hmf_Mz_mat[jz, :]
        P_1h = jsi.trapezoid(ukz_sqr * dndlnM_z, x=jnp.log(self.M_array))
        return P_1h