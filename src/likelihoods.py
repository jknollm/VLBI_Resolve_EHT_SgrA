# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Jakob Knollmüller

from functools import reduce
from operator import add
from functools import lru_cache, reduce
from scipy import interpolate

import nifty8 as ift
import numpy as np
import resolve as rve

from .utilities import gaussian_profile, disk_profile
from .response import (
    Visibilities2NormClosureAmplitudes,
    Visibilities2NormClosurePhases,
    build_interferometry_response
)


@lru_cache(maxsize=None)
def likelihood_closure_phases(closure_phase_systematics, obs,**kwargs):
    lh_closure_phase = []
    for ii, oo in enumerate(obs):
        vis2ph = (
            Visibilities2NormClosurePhases(oo, closure_phase_systematics)
            .ducktape(f"vis #{ii}")
            .ducktape_left(f"cl. ph. #{ii}")
        )
        vis = oo.vis.ducktape_left(f"vis #{ii}")
        dph = vis2ph(vis)
        if oo.freq < 100e9:
            channel_weight = 0.05
        elif oo.freq < 300e9:
            channel_weight = 1
        else:
            channel_weight = 1
        foo = ift.GaussianEnergy(data=dph) @ vis2ph
        foo.name = f"o{ii} cl. ph."
        lh_closure_phase.append(channel_weight * foo)
    return reduce(add, lh_closure_phase)

@lru_cache(maxsize=None)
def likelihood_closure_amplitudes(closure_amplitude_systematics, obs,**kwargs):
    lh_closure_amplitude = []
    for ii, oo in enumerate(obs):
        vis2ampl = (
            Visibilities2NormClosureAmplitudes(
                oo, closure_amplitude_systematics
            )
            .ducktape(f"vis #{ii}")
            .ducktape_left(f"cl. ampl. #{ii}")
        )
        vis = oo.vis.ducktape_left(f"vis #{ii}")
        dampl = vis2ampl(vis)
        if oo.freq < 100e9:
            channel_weight = 0.05
        elif oo.freq < 300e9:
            channel_weight = 1
        else:
            channel_weight = 1
        # Likelihoods with amplitudes
        assert dampl.domain.size > 0
        foo = ift.GaussianEnergy(data=dampl) @ vis2ampl
        foo.name = f"o{ii} cl amplitude"
        lh_closure_amplitude.append(channel_weight * foo)
    return reduce(add, lh_closure_amplitude)

@lru_cache(maxsize=None)
def likelihood_calib_logamplitudes(log_amplitude_systematics, obs,**kwargs):
    lh_calibrated_amplitude = []
    for ii, oo in enumerate(obs):
        vis = oo.vis.ducktape_left(f"vis #{ii}")
        weight =1/(1/oo.weight + (vis.abs()*log_amplitude_systematics)**2).ducktape_left(f"vis #{ii}")
        snr = ift.makeOp(vis.abs() * weight.sqrt())
        inp_vis = ift.ScalingOperator(vis.domain, 1.0)
        inp_cal = (
            ift.ScalingOperator(oo.vis.domain, 1.0)
            .ducktape(f"cal #{ii}")
            .ducktape_left(f"vis #{ii}")
        )
        vis2calibnormlogampl = snr @ (inp_vis.log().real + inp_cal)
        vis2normlogampl = snr @ inp_vis.log().real
        if oo.freq < 100e9:
            channel_weight = 0.05
        elif oo.freq < 300e9:
            channel_weight = 1
        else:
            channel_weight = 1
        foo = ift.GaussianEnergy(data=vis2normlogampl(vis)) @ vis2calibnormlogampl
        foo.name = f"o{ii} calib. log amplitudes"
        lh_calibrated_amplitude.append(channel_weight * foo)
    return reduce(add, lh_calibrated_amplitude)

def lhmix(
    sky_model,
    calibration_model,
    mix_scale,
    scattering,
    alma_lc,
    lh_closure_phases,
    lh_closure_amplitudes,
    lh_calibration_logamplitudes, **kwargs
):
    if 0 < mix_scale < 1:
        lhleft = (
            mix_scale * lh_closure_amplitudes
            + (1 - mix_scale) * lh_calibration_logamplitudes
            + lh_closure_phases
        )
    elif mix_scale == 0:
        lhleft = lh_calibration_logamplitudes + lh_closure_phases
    elif mix_scale == 1:
        lhleft = lh_closure_amplitudes + lh_closure_phases
    else:
        raise ValueError(f"Value for mix not allowed: {mix}")
    R = build_interferometry_response(sky_model.target, scattering, **kwargs)
    signal_response = R 
    if calibration_model != "None":
        signal_response += calibration_model
    lhleft = lhleft @ signal_response
    if alma_lc != "None":
        if 0 < mix_scale < 1:
            lhleft += 0.5*(1 - mix_scale) * alma_lc
        else:
            lhleft += alma_lc

    return lhleft.partial_insert(sky_model)
    


def likelihood_alma_lc(alma_lcs, sky_model, lc_systematics, **kwargs):
    FA = ift.FieldAdapter(sky_model['sky'].target,'sky')
    flux = FA.integrate(spaces=(3))
    tdom = flux.target[1]
    max_time = tdom.coordinates[-1]
    min_time = tdom.coordinates[0]
    rg_tdom = ift.RGSpace(tdom.shape)
    flux_duck = flux.ducktape_left(rg_tdom).ducktape_left("flux")
    llhds = []
    for i,lc in enumerate(alma_lcs):
        start_of_day = lc['t'].dt.floor('D')
        times = (lc['t'] - start_of_day).dt.total_seconds().to_numpy()
        time_locs = ((times - min_time)/(max_time-min_time)).reshape((1,)+times.shape)
        LIP = ift.LinearInterpolator(rg_tdom,time_locs).ducktape("flux")
        alma_flux = ift.makeField(LIP.target, lc["y"].to_numpy())
        alma_uncertainty = ift.makeField(LIP.target, lc["y_std"].to_numpy())**2 + (alma_flux*lc_systematics)**2
        icov = ift.makeOp(alma_uncertainty,sampling_dtype=np.float64).inverse
        llh = ift.GaussianEnergy(alma_flux, icov) @ LIP
        llh.name = f"alma_lc_{i}"
        llhds.append(llh)
    return reduce(add, llhds) @ flux_duck



def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False, fill_value="extrapolate")
    B = np.where(np.isfinite(A),A,f(inds))
    return B

def get_initial_likelihood(mode, sky_model, flux, initial_max_std, freq_npix, **kwargs):
    sky_target = sky_model.target["sky"]
    contr = ift.ContractionOperator(sky_target, (0, 1, 2))

    if mode == 'gaussian':
        angle = kwargs["gaussian_angle"]/360*np.pi*2
        locx = rve.str2rad(kwargs.get("gaussian_location_x", "0as"))
        locy = rve.str2rad(kwargs.get("gaussian_location_y", "0as"))
        std1 = rve.str2rad(kwargs["gaussian_std_1"])
        std2 = rve.str2rad(kwargs["gaussian_std_2"])
        prof = gaussian_profile(sky_target[3], (locx, locy), (std1, std2), angle)
    elif mode == 'disk':
        locx = rve.str2rad(kwargs.get("disk_location_x", "0as"))
        locy = rve.str2rad(kwargs.get("disk_location_y", "0as"))
        radius = rve.str2rad(kwargs["disk_radius"])
        smoothing = rve.str2rad(kwargs["disk_smoothing"])
        prof = disk_profile(sky_target[3], (locx, locy), radius, smoothing)
    else:
        raise NotImplementedError(f'strategy {mode} not implemented')

    # Normalize
    prof = prof + 0.01*prof.val.max()
    prof = prof / prof.s_integrate()
    prof = contr.adjoint(prof)

    # Spectral index
    if sky_target.shape[2] == 1 or kwargs["spectral_index"] is None:
        if isinstance(flux, list):
            lc = flux[0]
            FA = ift.FieldAdapter(sky_model['sky'].target,'sky')
            flux = FA.integrate(spaces=(3))
            tdom = flux.target[1]
            max_time = tdom.coordinates[-1]
            min_time = tdom.coordinates[0]
            rg_tdom = ift.RGSpace(tdom.shape)
            start_of_day = lc['t'].dt.floor('D')
            times = (lc['t'] - start_of_day).dt.total_seconds().to_numpy()
            time_locs = ((times - min_time)/(max_time-min_time)).reshape((1,)+times.shape)
            LIP = ift.LinearInterpolator(rg_tdom,time_locs).ducktape("flux")
            alma_flux = ift.makeField(LIP.target, lc["y"].to_numpy())
            ones = ift.full(LIP.target,1.)
            multi = LIP.adjoint_times(ones)
            flu = LIP.adjoint_times(alma_flux) / multi
            flu = fill_nan(flu.val['flux'])
            cont = ift.ContractionOperator(sky_target, (0,2,3))
            flu = ift.makeField(cont.target, flu)
            fac = cont.adjoint(flu)

        else:
            fac = flux
            fac = ift.makeField(sky_target[2], fac)
            fac = ift.ContractionOperator(sky_target, (0, 1, 3)).adjoint(fac)
    else:
        nu = np.array(sky_target[2].coordinates)
        fac = flux * (nu/kwargs["reference_frequency"])**kwargs["spectral_index"]
        fac = ift.makeField(sky_target[2], fac)
        fac = ift.ContractionOperator(sky_target, (0, 1, 3)).adjoint(fac)
    prof = fac * prof
    # Apply mask
    if prof.domain[0].shape[0]>1:
        profvals = prof.val.copy()
        profvals[1:] = 0.
        prof = ift.Field(prof.domain, profvals)
    prof_original = prof
    mask = np.ones(sky_target.shape, dtype=bool)
    mask[:, :, :freq_npix] = False
    if prof.domain[0].shape[0]>1 and kwargs["mask_pol"]:
        mask[1:] = False
    mask = ift.MaskOperator(ift.makeField(sky_target, mask))
    prof = mask(prof)
    assert prof_original.domain == sky_target

    cmax = prof.val.max()
    N = ift.ScalingOperator(prof.domain, (initial_max_std*cmax) ** 2, sampling_dtype=np.float64)
    my_d = prof + N.draw_sample()
    return ift.GaussianEnergy(my_d, N.inverse) @ mask.ducktape("sky") @ sky_model


def sum_ignore_zero(lst):
    from functools import reduce
    from operator import add
    lst = list(filter(lambda x: x != 0, lst))
    if len(lst) == 0:
        return 0
    return reduce(add, lst)

def sum_ignore_zero_apply_right(lst, op):
    lst = sum_ignore_zero(lst)
    if lst == 0:
        return 0
    return lst.partial_insert(op)

def sum_ignore_zero_apply_right_append(lh_lst, op_lst, op):
    op = sum_ignore_zero_apply_right(op_lst, op)
    lh_lst.append(op)
