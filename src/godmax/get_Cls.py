from .get_Pkzs import get_Pkz
from .base_class import get_vmapped_func, get_vmapped_func_warg, EmptyCallable
import jax.numpy as jnp
from jax import jit, vmap
import jax.scipy.integrate as jsi
from functools import partial
from astropy import constants as const
import interpax
from jax_cosmo.scipy.integrate import simps
import time
import numpy as _np
import math as _math

class get_Cl(get_Pkz):
    """
    Class that extends the get_Pkz object to compute angular power spectra C(ℓ).

    This class uses the 3D power spectra calculated in get_Pkz and transforms
    them to angular multipole space. It manages beam factors, optionally applies
    smoothing, and stores 2D interpolators for quick evaluation of the angular
    power at given ℓ and redshift.

    Attributes:
        sim_params_dict (dict): Dictionary containing the simulation parameters.
        halo_params_dict (dict): Dictionary containing the halo model parameters.
        analysis_dict (dict): Dictionary containing the analysis parameters.
        other_params_dict (dict): Dictionary containing other parameters.
        Pkz_obj (get_Pkz): If provided, the attributes from this object are
            copied into the get_Cl instance.
    """    
    def __init__(
                self,
                sim_params_dict: dict,
                halo_params_dict: dict,
                analysis_dict: dict,     
                other_params_dict: dict,
                Pkz_obj=None
            ):    
        if Pkz_obj is None:
            super().__init__(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
        else:
            self.__dict__.update(Pkz_obj.__dict__)

        # Convert the 3D power spectra multipole space Pkz_ell:
        vmapped_func = get_vmapped_func_warg(self.get_P_lz, 2, 3)
        self.Pkmm_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pmm_tot_mat).T
        self.Pkmm_nfw_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.phfit_kz_mat).T 
        if self.model_tSZ:
            self.Pkym_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pym_tot_mat).T
            Bl_array = jnp.exp(-1. * self.ell_array * (self.ell_array + 1) * (self.sig_beam ** 2) / 2.)
            self.Bl_mat = Bl_array[:, None]
            self.Pkym_lz_mat = self.Pkym_lz_mat * self.Bl_mat
            # tSZ auto 3D power spectrum P_yy(k, z)
            vmapped_func_yy = get_vmapped_func_warg(self.get_P_1h, 2, 4)
            self.Pyy_1h_kz_mat = vmapped_func_yy(jnp.arange(self.nk), jnp.arange(self.nz), 3, 3).T
            self.Pyy_2h_kz_mat = self.by_kz_mat * self.by_kz_mat * self.plin_kz_mat
            self.Pyy_tot_kz_mat = self.Pyy_1h_kz_mat + self.Pyy_2h_kz_mat
        if self.model_galaxies:
            self.Pkge_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pge_tot_mat).T
            self.Pkgm_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pgm_tot_mat).T
            self.Pkgm_nfw_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pgm_nfw_tot_mat).T
            self.Pkgy_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pgy_tot_mat).T
            self.Pkgy_lz_mat = self.Pkgy_lz_mat * self.Bl_mat
            self.Pkgg_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pgg_tot_mat).T

        # Get the interpolators:
        self.cached_power_spectra = jnp.zeros((4, 4, self.nell, self.nz_for_Cls))
        log_ell = jnp.log(self.ell_array)
        self.logPkmmlz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkmm_lz_mat), extrap=True)        
        self.cached_power_spectra = self.cached_power_spectra.at[0,0].set(vmap(lambda l: jnp.exp(self.logPkmmlz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
        self.logPkmm_nfw_lz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkmm_nfw_lz_mat), extrap=True)        
        self.cached_power_spectra = self.cached_power_spectra.at[1,1].set(vmap(lambda l: jnp.exp(self.logPkmm_nfw_lz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
        if self.model_tSZ:
            self.logPkymlz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkym_lz_mat), extrap=True)
            self.cached_power_spectra = self.cached_power_spectra.at[0,3].set(vmap(lambda l: jnp.exp(self.logPkymlz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
            self.cached_power_spectra = self.cached_power_spectra.at[3,0].set(self.cached_power_spectra[0,3])
            # P_yy Limber-projected interpolator + cache
            self.Pkyy_lz_mat = get_vmapped_func(self.get_Pkyy_lz, 2)(jnp.arange(self.nell), jnp.arange(self.nz)).T
            self.logPkyylz_2d_interp = interpax.Interpolator2D(
                jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkyy_lz_mat), extrap=True
            )
            self.cached_power_spectra = self.cached_power_spectra.at[3,3].set(vmap(lambda l: jnp.exp(self.logPkyylz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
        else: self.logPkymlz_2d_interp = EmptyCallable()
        if self.model_galaxies:
            self.logPkgmlz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkgm_lz_mat), extrap=True)        
            self.cached_power_spectra = self.cached_power_spectra.at[2,0].set(vmap(lambda l: jnp.exp(self.logPkgmlz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
            self.cached_power_spectra = self.cached_power_spectra.at[0,2].set(self.cached_power_spectra[2,0])
            self.logPkgglz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkgg_lz_mat), extrap=True)        
            self.cached_power_spectra = self.cached_power_spectra.at[2,2].set(vmap(lambda l: jnp.exp(self.logPkgglz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
            self.logPkgylz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkgy_lz_mat), extrap=True)         
            self.cached_power_spectra = self.cached_power_spectra.at[2,3].set(vmap(lambda l: jnp.exp(self.logPkgylz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
            self.cached_power_spectra = self.cached_power_spectra.at[3,2].set(self.cached_power_spectra[2,3])
            self.logPkgm_nfw_lz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkgm_nfw_lz_mat), extrap=True)        
            self.cached_power_spectra = self.cached_power_spectra.at[2,1].set(vmap(lambda l: jnp.exp(self.logPkgm_nfw_lz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
            self.cached_power_spectra = self.cached_power_spectra.at[1,2].set(self.cached_power_spectra[2,1])
        else: self.logPkgmlz_2d_interp, self.logPkgglz_2d_interp, self.logPkgylz_2d_interp, self.logPkgm_nfw_lz_2d_interp = EmptyCallable(), EmptyCallable(), EmptyCallable(), EmptyCallable()

        # Get the window functions for different probes:
        self.pzs_inp_mat = vmap(self.get_photoz_biased_nz)(jnp.arange(self.nbins))
        self.Wk_gravonly_mat = get_vmapped_func(self.get_weak_lensing_kernel, 2)(jnp.arange(self.nbins), jnp.arange(self.nz_for_Cls)).T
        self.nla_mat = get_vmapped_func(self.get_nla_kernel, 2)(jnp.arange(self.nbins), jnp.arange(self.nz_for_Cls)).T        
        self.Wk_mat = self.Wk_gravonly_mat + self.nla_mat
        self.Wy_array = (1.0 / (1.0 + self.z_array_for_Cls))
        if self.model_galaxies:
            self.Wg_mat = vmap(self.get_nz_lens_interp)(jnp.arange(self.nbins_lens))
        else: self.Wg_mat = jnp.zeros((1,1))

        if self.ENABLE_TIMING:
            ti = time.time()
        # Get the Cls:
        # vmapped_func = get_vmapped_func_warg(self.get_Cl_tot, 3, 5)
        # self.Cl_kappa_kappa_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins), jnp.arange(self.nbins), 0, 0).T

        vmapped_func = get_vmapped_func_warg(self.get_Cl_tot, 2, 4)
        self.Cl_kappa_kappa_tot_mat = vmapped_func(jnp.arange(self.nbins), jnp.arange(self.nbins), 0, 0).T


        # self.Cl_kappa_kappa_nfw_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins), jnp.arange(self.nbins), 1, 1).T
        if self.ENABLE_TIMING:
            print("Time to compute the kappa kappa: ", time.time() - ti)
            ti = time.time()

        if self.model_galaxies:
            # self.Cl_gal_gal_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins_lens), jnp.arange(self.nbins_lens), 2, 2).T
            self.Cl_gal_gal_tot_mat = vmapped_func(jnp.arange(self.nbins_lens), jnp.arange(self.nbins_lens), 2, 2).T
            if self.ENABLE_TIMING:
                print("Time to compute the gal gal: ", time.time() - ti)
                ti = time.time()
            
            # self.Cl_gal_kappa_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins_lens), jnp.arange(self.nbins), 2, 0).T
            self.Cl_gal_kappa_tot_mat = vmapped_func(jnp.arange(self.nbins_lens), jnp.arange(self.nbins), 2, 0).T
            if self.ENABLE_TIMING:
                print("Time to compute the kappa gal: ", time.time() - ti)
                ti = time.time()
            # self.Cl_gal_kappa_nfw_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins_lens), jnp.arange(self.nbins), 2, 1).T
            # Get the Pge in the given k-array:
            self.Pge_zarray = vmap(self.get_Pge_interpz)(jnp.arange(self.nk))
            self.Pge_tot_mat = vmap(self.get_Pge_tot_ks)(jnp.arange(self.nbins_lens))
            if self.ENABLE_TIMING:
                print("Time to compute the Pge in the k-array: ", time.time() - ti)
                ti = time.time()
        if self.model_tSZ:
            # vmapped_func = get_vmapped_func_warg(self.get_Cl_tot, 2, 5)
            vmapped_func = get_vmapped_func_warg(self.get_Cl_tot, 1, 4)

            # self.Cl_kappa_y_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins), 0, 0, 3).T
            self.Cl_kappa_y_tot_mat = vmapped_func(jnp.arange(self.nbins), 0, 0, 3).T
            if self.ENABLE_TIMING:
                print("Time to compute the kappa y: ", time.time() - ti)
                ti = time.time()
            if self.model_galaxies:
                # self.Cl_gal_y_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins_lens), 0, 2, 3).T
                self.Cl_gal_y_tot_mat = vmapped_func(jnp.arange(self.nbins_lens), 0, 2, 3).T
                if self.ENABLE_TIMING:
                    print("Time to compute the gal y: ", time.time() - ti)

            # tSZ auto angular power spectrum C(ℓ)_yy.
            # Always store pure theory first; noise and total follow from the file.
            self.Cl_y_y_signal_mat = self.get_Cl_tot(0, 0, 3, 3)

            yy_total_ell_fname = analysis_dict.get('yy_total_ell_fname', None)
            yy_noise_ell_fname = analysis_dict.get('yy_noise_ell_fname', None)
            if yy_total_ell_fname is not None:
                ell_yy_f, Cl_yy_f = _np.loadtxt(yy_total_ell_fname, unpack=True, usecols=(0, 1))
                log_interp = interpax.Interpolator1D(
                    jnp.log(jnp.array(ell_yy_f)),
                    jnp.log(jnp.array(Cl_yy_f) + 1e-25),
                    extrap=(_math.log(Cl_yy_f[0]), _math.log(Cl_yy_f[-1]))
                )
                self.Cl_y_y_tot_mat   = jnp.exp(log_interp(jnp.log(self.ell_array)))
                self.Cl_y_y_noise_mat = self.Cl_y_y_tot_mat - self.Cl_y_y_signal_mat
                print('Loaded yy total from file into Cl_y_y_tot_mat')
            elif yy_noise_ell_fname is not None:
                ell_yy_n, Cl_yy_n = _np.loadtxt(yy_noise_ell_fname, unpack=True, usecols=(0, 1))
                log_noise_interp = interpax.Interpolator1D(
                    jnp.log(jnp.array(ell_yy_n)),
                    jnp.log(jnp.abs(jnp.array(Cl_yy_n)) + 1e-25),
                    extrap=True
                )
                self.Cl_y_y_noise_mat = jnp.exp(log_noise_interp(jnp.log(self.ell_array)))
                self.Cl_y_y_tot_mat   = self.Cl_y_y_signal_mat + self.Cl_y_y_noise_mat
                print('Loaded yy noise from file; Cl_y_y_tot_mat = theory + noise')
            else:
                print('Warning: no yy-total or yy-noise file provided; Cl_y_y_tot_mat = theory only')
                self.Cl_y_y_noise_mat = jnp.zeros_like(self.Cl_y_y_signal_mat)
                self.Cl_y_y_tot_mat   = self.Cl_y_y_signal_mat

        self._build_cls_1h2h_dict()



    @partial(jit, static_argnums=(0,))
    def get_P_lz(self, jl, jz, Pk_mat):
        """
        Compute the projected angular power spectrum for a given ℓ and redshift index.

        This method takes the 3D power spectrum (Pk_mat) at a given redshift bin jz,
        converts the target multipole ℓ into a corresponding wavenumber k, and then
        interpolates the 3D power spectrum at that k to get the angular power value.

        Args:
            jl (int): The index of the multipole ℓ in self.ell_array.
            jz (int): The index of the redshift in self.chi_array.
            Pk_mat (jax.numpy.DeviceArray): The 3D power spectrum array, with shape
                [n_k, n_z].

        Returns:
            jax.numpy.DeviceArray: The projected angular power spectrum at the
            specified multipole ℓ and redshift jz.
        """        
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(Pk_mat[:,jz])))
        return Pkz_ell

    @partial(jit, static_argnums=(0,))
    def get_P_lz_for_Cls(self, jl, jz, Pk_interp_obj):
        value = jnp.exp(Pk_interp_obj(jnp.log(self.ell_array[jl]), self.z_array_for_Cls[jz]))        
        return value 
    

    @partial(jit, static_argnums=(0,))
    def get_photoz_biased_nz(self, jb):
        """
        Returns a photo-z shift parameter biased n(z)
        """
        val_biased = jnp.interp(self.z_array_nz - self.Delta_z_bias_array[jb], self.z_array_nz, self.pzs_inp_mat_inp[jb, :])
        norm_val = jsi.trapezoid(val_biased, x=self.z_array_nz)
        return val_biased / norm_val


    @partial(jit, static_argnums=(0,))
    def get_nz_lens_interp(self, jb):
        nz_jb = self.pzs_inp_mat_inp_lens[jb,:]
        nz_interp = jnp.interp(self.z_array_for_Cls, self.z_array_nz_lens, nz_jb)
        norm_val = jsi.trapezoid(nz_interp, x=self.z_array_for_Cls)
        return nz_interp / norm_val


    @partial(jit, static_argnums=(0,))
    def get_weak_lensing_kernel(self, jb, jz):
        """
        Returns a weak lensing kernel

        Note: this function handles differently nzs that correspond to extended redshift
        distribution, and delta functions.
        """
        z = self.z_array_for_Cls[jz]
        chi = self.chi_array_for_Cls[jz]

        @vmap
        def integrand(z_prime):
            chi_prime = jnp.exp(jnp.interp(z_prime, self.z_array, jnp.log(self.chi_array)))
            dndz = (jnp.interp(z_prime, self.z_array_nz, self.pzs_inp_mat[jb, :]))
            return dndz * jnp.clip(chi_prime - chi, 0) / jnp.clip(chi_prime, 0.1)

        radial_kernel = simps(integrand, z, self.zmax, 128) * (1.0 + z) * chi

        constant_factor = 3.0 * (100.)**2 * self.cosmo_jax.Omega_m / (2.0 * ((const.c.value * 1e-3)**2))
        return constant_factor * radial_kernel

    @partial(jit, static_argnums=(0,))
    def get_nla_kernel(self, jb, jz):
        """
        Computes the NLA IA kernel
        """
        z = self.z_array_for_Cls[jz]
        Dz = self.growth_array_for_Cls[jz]
        # Az_IA = -1. * self.A_IA * self.rho_m_bar * self.C1_bar * (1. / Dz) * ((1. + z) / (1. + self.z0_IA))**self.eta_IA
        Az_IA = -1. * self.A_IA * self.C1_rho_m_bar * (1. / Dz) * ((1. + z) / (1. + self.z0_IA))**self.eta_IA        
        # dchi_dz = (const.c.to(u.km / u.s)).value / (bkgrd.H(self.cosmo_jax, z2a(z)))
        dchi_dz = self.dchi_dz_array_for_Cls[jz]
        dndz = (jnp.interp(z, self.z_array_nz, self.pzs_inp_mat[jb, :]))
        return Az_IA * dndz / dchi_dz

    @partial(jit, static_argnums=(0,))
    def get_Cl_tot(self, jb1, jb2, probe1, probe2): 
        """
        Compute the total angular power spectrum C(ℓ) for given multipole index jl,
        lens/source bins jb1, jb2, and probe types probe1, probe2.

        This function determines prefactors based on the probe type, selects the
        appropriate 2D interpolator for the power spectrum, and applies the
        line-of-sight integration or weighting factors as needed.

        Args:
            jl (int): Index of the multipole ℓ in self.ell_array.
            jb1 (int): First index for lens/source bin.
            jb2 (int): Second index for lens/source bin.
            probe1 (int): Integer code selecting the first field (e.g., shear, galaxies, tSZ).
            probe2 (int): Integer code selecting the second field.

        Returns:
            jax.numpy.DeviceArray: The total angular power spectrum C(ℓ) for the
            specified multipole index, bins, and probes.
        """        

        # Handle the first probe condition
        @jit
        def compute_prefac(probe, jb):
            conditions = [
                (probe == 0, (1. + self.mult_shear_bias_array[jb]) * (self.Wk_mat[jb] / (self.chi_array_for_Cls**2))),
                (probe == 1, (1. + self.mult_shear_bias_array[jb]) * (self.Wk_mat[jb] / (self.chi_array_for_Cls**2))),
                (probe == 2, self.Wg_mat[jb] / (self.dchi_dz_array_for_Cls * self.chi_array_for_Cls**2)),
                (probe == 3, self.Wy_array / (self.chi_array_for_Cls**2)),
            ]
            
            # Default value if no condition matches
            prefac = jnp.nan
            for condition, value in conditions:
                prefac = jnp.where(condition, value, prefac)
            return prefac

        # Compute prefactors for probe1 and probe2
        prefac_for_uk1 = compute_prefac(probe1, jb1)
        prefac_for_uk2 = compute_prefac(probe2, jb2)        

        # Define the conditions and corresponding functions
        # conditions = [
        #     (jnp.logical_and(probe1 == 0, probe2 == 0), self.logPkmmlz_2d_interp),
        #     (jnp.logical_and(probe1 == 1, probe2 == 1), self.logPkmm_nfw_lz_2d_interp),
        #     (jnp.logical_and(probe1 == 2, probe2 == 2), self.logPkgglz_2d_interp),
        #     (jnp.logical_and(probe1 == 2, probe2 == 0), self.logPkgmlz_2d_interp),
        #     (jnp.logical_and(probe1 == 0, probe2 == 2), self.logPkgmlz_2d_interp),
        #     (jnp.logical_and(probe1 == 2, probe2 == 1), self.logPkgm_nfw_lz_2d_interp),
        #     (jnp.logical_and(probe1 == 1, probe2 == 2), self.logPkgm_nfw_lz_2d_interp),
        #     (jnp.logical_and(probe1 == 2, probe2 == 3), self.logPkgylz_2d_interp),
        #     (jnp.logical_and(probe1 == 3, probe2 == 2), self.logPkgylz_2d_interp),
        #     (jnp.logical_and(probe1 == 0, probe2 == 3), self.logPkymlz_2d_interp),
        #     (jnp.logical_and(probe1 == 3, probe2 == 0), self.logPkymlz_2d_interp),
        # ]

        # Compute Pk
        # Pk = jnp.nan  # Default value if no condition matches
        # for condition, func in conditions:
        #     Pk = jnp.where(condition, jnp.exp(func(jnp.log(self.ell_array[jl]), self.z_array_for_Cls)), Pk)
        Pk = self.cached_power_spectra[probe1, probe2]
        fx = prefac_for_uk1 * prefac_for_uk2  * (self.chi_array_for_Cls ** 2) * self.dchi_dz_array_for_Cls * Pk
        return jsi.trapezoid(fx, x=self.z_array_for_Cls)     

    @partial(jit, static_argnums=(0,))
    def get_Pkyy_lz(self, jl, jz):
        """
        Limber-projected tSZ auto power spectrum at multipole index jl and redshift index jz,
        including the beam suppression factor B(ℓ)^2.
        """
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        Bl = jnp.exp(-1. * ell * (ell + 1) * (self.sig_beam ** 2) / 2.)
        k_ell = (ell + 0.5) / jnp.clip(chi_z, 1.0)
        Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.Pyy_tot_kz_mat[:, jz])))
        return (Bl ** 2) * Pkz_ell

    @partial(jit, static_argnums=(0,))
    def get_Pge_interpz(self, jk):
        return jnp.interp(self.z_array_for_Cls, self.z_array, self.Pge_tot_mat[jk,:])

    @partial(jit, static_argnums=(0,))
    def get_Pge_tot_ks(self, jb):
        """
        Get the galaxy-electron cross power spectrum for a given lens bin jb
        """
        fx_intz = jsi.trapezoid(self.Pge_zarray * self.Wg_mat[jb][None,:], x=self.z_array_for_Cls)
        return jnp.exp(jnp.interp(jnp.log(self.k_array_survey), jnp.log(self.kPk_array), jnp.log(fx_intz + 1e-40)))

    def _build_cls_1h2h_dict(self):
        """
        Build self.Cl_1h2h_dict with 1h, 2h, and total Dl = ell(ell+1)Cl/2pi
        for all active probes.

        Structure::

            Cl_1h2h_dict = {
                'ell': array,            # (nell,)
                'yy': [{'label': 'yy', 'tot': ..., '1h': ..., '2h': ...}],
                'ky': [{'label': 'b0', ...}, ...],   # one entry per source bin
                'kk': [{'label': '(0,1)', ...}, ...],
                'gy': [...], 'gk': [...], 'gg': [...],
            }

        All Dl arrays have shape (nell,) and units matching the Cl attributes
        (multiply by 1e12 for typical plot scaling).
        """
        from math import pi as _pi

        # ---- numpy copies of integration arrays ----
        ell     = _np.array(self.ell_array)
        ell_fac = ell * (ell + 1) / (2.0 * _pi)
        z_cls   = _np.array(self.z_array_for_Cls)
        chi_cls = _np.array(self.chi_array_for_Cls)
        dchi    = _np.array(self.dchi_dz_array_for_Cls)
        z_Pk    = _np.array(self.z_array)
        kPk     = _np.array(self.kPk_array)
        nell    = len(ell)
        nz      = len(z_cls)

        if self.model_tSZ:
            Bl = _np.exp(-0.5 * ell * (ell + 1) * float(self.sig_beam) ** 2)
        else:
            Bl = _np.ones(nell)

        mult   = _np.array(self.mult_shear_bias_array)
        Wk     = _np.array(self.Wk_mat)
        Wk_eff = _np.array([
            (1.0 + mult[jb]) * Wk[jb] / _np.maximum(chi_cls ** 2, 1e-10)
            for jb in range(self.nbins)
        ])
        Wy_eff = _np.array(self.Wy_array) / _np.maximum(chi_cls ** 2, 1e-10)

        if self.model_galaxies:
            Wg = _np.array(self.Wg_mat)
            Wg_eff = _np.array([
                Wg[jb] / _np.maximum(dchi * chi_cls ** 2, 1e-10)
                for jb in range(self.nbins_lens)
            ])

        def _limber_cl(Pkz_kz, W1, W2, beam_pow=0):
            Pkz  = _np.maximum(_np.asarray(Pkz_kz), 1e-100)
            lkPk = _np.log(kPk)
            lz_Pk  = _np.log(_np.maximum(z_Pk, 1e-10))
            lz_cls = _np.log(_np.maximum(z_cls, 1e-10))
            Pkz_zcls = _np.vstack([
                _np.exp(_np.interp(lz_cls, lz_Pk, _np.log(_np.maximum(Pkz[ik, :], 1e-100))))
                for ik in range(len(kPk))
            ])
            log_Pkz_zcls = _np.log(_np.maximum(Pkz_zcls, 1e-100))
            k_ell_mat = _np.outer(ell + 0.5, 1.0 / _np.maximum(chi_cls, 1.0))
            log_k_ell = _np.log(_np.maximum(k_ell_mat, 1e-100))
            Pk_lz = _np.zeros((nell, nz))
            for iz in range(nz):
                Pk_lz[:, iz] = _np.exp(
                    _np.interp(log_k_ell[:, iz], lkPk, log_Pkz_zcls[:, iz])
                )
            if beam_pow == 1:
                Pk_lz *= Bl[:, None]
            elif beam_pow == 2:
                Pk_lz *= (Bl ** 2)[:, None]
            integrand = (W1 * W2 * chi_cls ** 2 * dchi)[None, :] * Pk_lz
            Cl = _np.trapz(integrand, z_cls, axis=1)
            return ell_fac * Cl

        panels = {}

        if self.model_tSZ:
            panels['yy'] = [{'label': 'yy',
                'tot': ell_fac * _np.array(self.Cl_y_y_signal_mat),
                '1h' : _limber_cl(self.Pyy_1h_kz_mat, Wy_eff, Wy_eff, beam_pow=2),
                '2h' : _limber_cl(self.Pyy_2h_kz_mat, Wy_eff, Wy_eff, beam_pow=2),
            }]
            ky_curves = []
            for jb in range(self.nbins):
                ky_curves.append({'label': f'b{jb}',
                    'tot': ell_fac * _np.array(self.Cl_kappa_y_tot_mat[:, jb]),
                    '1h' : _limber_cl(self.Pym_1h_kz_mat, Wk_eff[jb], Wy_eff, beam_pow=1),
                    '2h' : _limber_cl(self.Pym_2h_kz_mat, Wk_eff[jb], Wy_eff, beam_pow=1),
                })
            panels['ky'] = ky_curves

        kk_curves = []
        for jb1 in range(self.nbins):
            for jb2 in range(jb1, self.nbins):
                kk_curves.append({'label': f'({jb1},{jb2})',
                    'tot': ell_fac * _np.array(self.Cl_kappa_kappa_tot_mat[:, jb1, jb2]),
                    '1h' : _limber_cl(self.Pmm_dmb_1h_kz_mat, Wk_eff[jb1], Wk_eff[jb2]),
                    '2h' : _limber_cl(self.Pmm_dmb_2h_kz_mat, Wk_eff[jb1], Wk_eff[jb2]),
                })
        panels['kk'] = kk_curves

        if self.model_tSZ and self.model_galaxies:
            gy_curves = []
            for jb in range(self.nbins_lens):
                gy_curves.append({'label': f'b{jb}',
                    'tot': ell_fac * _np.array(self.Cl_gal_y_tot_mat[:, jb]),
                    '1h' : _limber_cl(self.Pgy_1h_kz_mat, Wg_eff[jb], Wy_eff, beam_pow=1),
                    '2h' : _limber_cl(self.Pgy_2h_kz_mat, Wg_eff[jb], Wy_eff, beam_pow=1),
                })
            panels['gy'] = gy_curves

        if self.model_galaxies:
            gk_curves = []
            for jb1 in range(self.nbins_lens):
                for jb2 in range(self.nbins):
                    gk_curves.append({'label': f'({jb1},{jb2})',
                        'tot': ell_fac * _np.array(self.Cl_gal_kappa_tot_mat[:, jb1, jb2]),
                        '1h' : _limber_cl(self.Pgm_1h_kz_mat, Wg_eff[jb1], Wk_eff[jb2]),
                        '2h' : _limber_cl(self.Pgm_2h_kz_mat, Wg_eff[jb1], Wk_eff[jb2]),
                    })
            panels['gk'] = gk_curves

            gg_curves = []
            for jb1 in range(self.nbins_lens):
                for jb2 in range(jb1, self.nbins_lens):
                    gg_curves.append({'label': f'({jb1},{jb2})',
                        'tot': ell_fac * _np.array(self.Cl_gal_gal_tot_mat[:, jb1, jb2]),
                        '1h' : _limber_cl(self.Pgg_1h_kz_mat, Wg_eff[jb1], Wg_eff[jb2]),
                        '2h' : _limber_cl(self.Pgg_2h_kz_mat, Wg_eff[jb1], Wg_eff[jb2]),
                    })
            panels['gg'] = gg_curves

        panels['ell'] = ell
        self.Cl_1h2h_dict = panels