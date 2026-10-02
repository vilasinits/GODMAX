"""cloelib cosmology in GODMAX units.

cloelib backgrounds and perturbations work in Mpc (k in 1/Mpc, P in Mpc^3, chi in Mpc,
H in km/s/Mpc). GODMAX works in h-units (k in h/Mpc, P in (Mpc/h)^3, chi in Mpc/h,
H / h). ``CloelibCosmology`` wraps any objects implementing cloelib's ``Background`` and
``Perturbations`` protocols and returns the quantities ``base_class`` needs in GODMAX
units, so GODMAX and cloelib probes can share one cosmology.
"""

import jax.numpy as jnp

C_KM_S = 299792.458


class CloelibCosmology:
    """GODMAX (h-unit) view of a cloelib background, linear and non-linear perturbations."""

    def __init__(self, background, linear, nonlinear):
        self.background = background
        self.linear = linear
        self.nonlinear = nonlinear
        self.h = background.h

    @classmethod
    def from_godmax_params(cls, cosmo_params):
        """Build cloelib's JAX backend from a GODMAX ``sim_params['cosmo']`` dict.

        GODMAX has no neutrinos, curvature or wa, so the JAX backend is set up flat, massless
        and w0CDM. It takes As, not sigma8; its P(k) normalisation reads sigma8 from
        ``interface_args``, so GODMAX's sigma8 is written there (As is only a placeholder).
        """
        from cloelib.cosmology.jax_cosmology import (
            JAXBackground,
            JAXLinearPerturbations,
            JAXNonLinearPerturbations,
        )

        background = JAXBackground(
            H0=cosmo_params['H0'],
            Omega_b0=cosmo_params['Ob0'],
            Omega_cdm0=cosmo_params['Om0'] - cosmo_params['Ob0'],
            Omega_k0=0.0,
            As=2.1e-9,
            ns=cosmo_params['ns'],
            mnu=0.0,
            w0=cosmo_params['w0'],
            wa=0.0,
            gamma_MG=0.55,
            N_mnu=0,
        )
        background.interface_args['JAXparams']['sigma_8'] = cosmo_params['sigma8']
        return cls(background, JAXLinearPerturbations(background), JAXNonLinearPerturbations(background))

    def godmax_params(self):
        """GODMAX ``sim_params['cosmo']`` dict of the wrapped cloelib cosmology.

        sigma8 is the value cloelib's linear P(k) is normalised to, so GODMAX and cloelib
        share one amplitude whatever the cloelib background was built from (As).
        """
        background = self.background
        return {
            'flat': True,
            'H0': background.H0,
            'Om0': background.Omega_b0 + background.Omega_cdm0,
            'Ob0': background.Omega_b0,
            'sigma8': self.linear.sigma8_0(),
            'ns': background.ns,
            'w0': background.w0,
        }

    def comoving_distance(self, z):
        """Comoving distance [Mpc/h]."""
        return self.background.comoving_distance(jnp.atleast_1d(z)) * self.h

    def angular_diameter_distance(self, z):
        """Angular diameter distance [Mpc/h]."""
        return self.background.angular_diameter_distance(jnp.atleast_1d(z)) * self.h

    def dchi_dz(self, z):
        """c / H(z) [Mpc/h]."""
        return C_KM_S / self.background.hubble_parameter(jnp.atleast_1d(z)) * self.h

    def Esqr(self, a):
        """E(a)^2 = (H(a) / H0)^2."""
        return (self.background.hubble_parameter(1.0 / a - 1.0) / self.background.H0) ** 2

    def growth_factor(self, z):
        """Linear growth factor normalised to D(z=0) = 1."""
        z = jnp.atleast_1d(z)
        return self.linear.growth_factor(z) / self.linear.growth_factor(jnp.zeros(1))

    def linear_power(self, k, z):
        """Linear P(k, z) [(Mpc/h)^3] at k [h/Mpc], shape (nk, nz)."""
        return self._power(self.linear, k, z)

    def nonlinear_power(self, k, z):
        """Non-linear P(k, z) [(Mpc/h)^3] at k [h/Mpc], shape (nk, nz)."""
        return self._power(self.nonlinear, k, z)

    def _power(self, perturbations, k, z):
        pk = perturbations.matter_power_spectrum(jnp.atleast_1d(z), k * self.h)
        return jnp.reshape(pk, (len(jnp.atleast_1d(z)), len(k))).T * self.h**3
