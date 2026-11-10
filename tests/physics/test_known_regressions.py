"""Known bugs, recorded as strict expected failures.

Each test asserts the CORRECT behaviour. While the bug is present the test fails and pytest
reports it as XFAIL. When the bug is fixed the test passes, `strict=True` turns that into a
FAILURE, and the xfail marker must be removed so the test guards the fix from then on.
"""
import pytest

from godmax.get_Pkzs import get_Pkz

@pytest.mark.xfail(strict=True, reason="14ae4e4 removed the HMF name aliases; the code default 'tinker08' "
                                       "is no longer recognised")
def test_hmf_model_code_default(model):
    model("on", cls=get_Pkz, halo_over={"hmf_model": "tinker08"})


@pytest.mark.xfail(strict=True, reason="14ae4e4 changed jnp.ones((self.nk, ...)) to len(self.nk); nk is an int")
def test_do_corr_2h_mm_false(model):
    model("on", cls=get_Pkz, halo_over={"do_corr_2h_mm": False})
