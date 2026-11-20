"""Zheng_HOD: the parameters it fixes and the threshold it uses."""
import jax.numpy as jnp
import pytest

from godmax.base_class import base_class

HALO = dict(z_array=[0.1, 0.5, 1.0], nk=16, nM=6, nr=16, ell_array=jnp.array([10.0, 100.0]))
# init_power=False: only the input reading is tested here.
SIM = {'init_power': False}


def test_zheng_reads_log10Mmin_Ncen():
    model = base_class(dict(SIM, log10Mmin_Ncen=12.5), HALO, {'Zheng_HOD': True}, {})
    assert model.log10Mmin_Ncen == 12.5
    assert model.gamma_fshmr == 0.0


@pytest.mark.parametrize("key", ["gamma_fshmr", "Bcut_Nsat_array"])
def test_zheng_rejects_fixed_parameters(key):
    with pytest.raises(ValueError, match="fixed to 0 by the Zheng HOD"):
        base_class(dict(SIM, **{key: 1.0}), HALO, {'Zheng_HOD': True}, {})


@pytest.mark.parametrize("key", ["log10Mmin_Ncen", "log10Mmin_Ncen_array"])
def test_log10Mmin_Ncen_needs_zheng(key):
    with pytest.raises(ValueError, match="only used with Zheng_HOD=True"):
        base_class(dict(SIM, **{key: 12.5}), HALO, {}, {})
