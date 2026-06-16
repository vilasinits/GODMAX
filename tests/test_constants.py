"""
Sanity checks on physical constants. No pipeline needed — instant.
"""

import pytest
from godmax.helpers.constants import (
    RHO_CRIT_0_KPC3, RHO_CRIT_0_MPC3, DELTA_COLLAPSE,
    C, M_PROTON, KB, MPC, G_CGS,
)


class TestConstants:
    def test_rho_crit_unit_consistency(self):
        """RHO_CRIT_0_MPC3 / RHO_CRIT_0_KPC3 == (Mpc/kpc)^3 == 1e9."""
        ratio = RHO_CRIT_0_MPC3 / RHO_CRIT_0_KPC3
        assert abs(ratio - 1e9) / 1e9 < 1e-6

    def test_rho_crit_value(self):
        """Critical density at z=0 near 2.775e11 M_sun h^2 Mpc^-3."""
        assert 2.7e11 < RHO_CRIT_0_MPC3 < 2.8e11

    def test_delta_collapse(self):
        """Linear collapse threshold ~ 1.686."""
        assert abs(DELTA_COLLAPSE - 1.686) < 0.002

    def test_speed_of_light_cgs(self):
        """Speed of light ~ 3e10 cm/s."""
        assert abs(C - 2.998e10) / 2.998e10 < 1e-3

    def test_mpc_in_cm(self):
        """1 Mpc ~ 3.086e24 cm."""
        assert abs(MPC - 3.086e24) / 3.086e24 < 1e-3

    def test_proton_mass_grams(self):
        """Proton mass ~ 1.673e-24 g."""
        assert abs(M_PROTON - 1.6726e-24) / 1.6726e-24 < 1e-3
