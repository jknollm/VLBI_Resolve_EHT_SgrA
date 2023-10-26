# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Jakob Knollmüller

from functools import reduce
from operator import add

import nifty8 as ift
import numpy as np
import resolve as rve

from .closure_magic import Visibilities2ClosureMat

def build_interferometry_response(domain, scattering, obs, **kwargs):
    res = []
    do_wgridding = False
    for ii, oo in enumerate(obs):
        R = rve.InterferometryResponse(
            oo,
            domain["sky"],
            do_wgridding=do_wgridding,
            epsilon=1e-6,
            nthreads=ift.nthreads(),
        )
        if scattering:
            def get_scattering_kernel(obs):
                import ehtim

                kernel = np.zeros(len(obs.uvw))
                sm = ehtim.scattering.ScatteringModel(
                    model="dipole",
                    scatt_alpha=1.38,
                    observer_screen_distance=2.82 * 3.086e21,
                    source_screen_distance=5.53 * 3.086e21,
                    theta_maj_mas_ref=1.380,
                    theta_min_mas_ref=0.703,
                    POS_ANG=81.9,
                    wavelength_reference_cm=1.0,
                    r_in=800e5,
                    r_out=1e20,
                )
                for i in range(len(kernel)):
                    kernel[i] = sm.Ensemble_Average_Kernel_Visibility(
                        obs.uvw[i][0],
                        obs.uvw[i][1],
                        wavelength_cm=rve.SPEEDOFLIGHT / obs.freq * 100.0,
                        use_approximate_form=True,
                    )
                return kernel

            assert not do_wgridding
            kernel = get_scattering_kernel(oo)
            K = ift.makeOp(
                ift.makeField(R.target, kernel.reshape(R.target.shape))
            ) 
            R = K @ R
        res.append(R.ducktape("sky").ducktape_left(f"vis #{ii}"))
    return reduce(add, res)
    
def Visibilities2NormClosurePhases(observation, systematics=0.):
    ind = np.lexsort((observation.ant2, observation.ant1, observation.time))
    assert np.all(np.diff(ind) > 0)

    ops = []
    for pp in range(observation.npol):
        for ff in range(observation.nfreq):
            key = _vis_multi_key(pp, ff)
            d = _observation_to_d(observation, pp, ff)
            smo, clos2eig, _, _ = Visibilities2ClosureMat(d, False, systematics)
            inp = ift.ScalingOperator(smo.domain, 1.)
            op = smo @ inp.log().imag
            ima = ift.Imaginizer(op.target).adjoint
            vis2clos = ima(op).exp()
            ops.append((clos2eig @ vis2clos).ducktape_left(key).ducktape(key))
    closure_operators = reduce(add, ops)
    split = Vis2MultiField(observation.vis.domain)
    return ift.Multifield2Vector(closure_operators.target) @ closure_operators @ split


def Visibilities2NormClosureAmplitudes(observation,systematics=0.):
    ind = np.lexsort((observation.ant2, observation.ant1, observation.time))
    assert np.all(np.diff(ind) > 0)

    ops = []
    for pp in range(observation.npol):
        for ff in range(observation.nfreq):
            key = _vis_multi_key(pp, ff)
            d = _observation_to_d(observation, pp, ff)
            vis2closeig, _, _ = Visibilities2ClosureMat(d, True, systematics)
            inp = ift.ScalingOperator(vis2closeig.domain, 1.)
            ops.append((vis2closeig @ inp.log().real).ducktape_left(key).ducktape(key))
    closure_operators = reduce(add, ops)
    split = Vis2MultiField(observation.vis.domain)
    return ift.Multifield2Vector(closure_operators.target) @ closure_operators @ split


class Vis2MultiField(ift.LinearOperator):
    def __init__(self, domain):
        self._domain = ift.DomainTuple.make(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        dom0 = self._domain[1]
        tgt = {_vis_multi_key(ii, jj): dom0
               for ii in range(self.domain.shape[0])     # Polarization
               for jj in range(self.domain.shape[2])}    # Frequencies
        self._target = ift.makeDomain(tgt)

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = {}
            for ii in range(self.domain.shape[0]):
                for jj in range(self.domain.shape[2]):
                    res[_vis_multi_key(ii, jj)] = x[ii, :, jj]
        else:
            dtypes = [xx.dtype for xx in x.values()]
            if not all(dd == dtypes[0] for dd in dtypes):
                raise TypeError
            res = np.empty(self.domain.shape, dtypes[0])
            for ii in range(self.domain.shape[0]):
                for jj in range(self.domain.shape[2]):
                    res[ii, :, jj] = x[_vis_multi_key(ii, jj)]
        return ift.makeField(self._tgt(mode), res)


def _vis_multi_key(ipol, ifreq):
    return f"pol{ipol}freq{ifreq}"


def _observation_to_d(observation, polarization_index, freq_index):
    ind = np.lexsort((observation.ant2, observation.ant1, observation.time))
    assert np.all(np.diff(ind) > 0)
    return {"sigma": 1/observation.weight.sqrt().val[polarization_index, :, freq_index],
            "vis": observation.vis.val[polarization_index, :, freq_index],
            "time": observation.time,
            "ant1": observation.ant1,
            "ant2": observation.ant2,
            "uv": observation.uvw[..., 0:2]}
