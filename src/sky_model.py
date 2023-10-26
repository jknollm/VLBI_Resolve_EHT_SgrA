# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Jakob Knollmüller

import nifty8 as ift
import numpy as np
import resolve as rve
from math import ceil
from functools import reduce
from operator import add


def sky(**kwargs):
    res, _ = make_skymodel( **kwargs)
    return res.ducktape_left("sky")

def make_skymodel(
    *,
    obs,
    space_fov_x,
    space_fov_y,
    space_npix_x,
    space_npix_y,
    model,
    prefactor_model,
    delta_t,
    freq_npix,
    **kwargs,
):
    tmin = min([np.min(oo.time) for oo in obs])
    tmax = max([np.max(oo.time) for oo in obs])
    fmin = min([np.min(oo.freq) for oo in obs])
    fmax = max([np.max(oo.freq) for oo in obs])
    
    if delta_t == -1:
        time_npix = 1
    else:
        time_npix = ceil((tmax - tmin) / delta_t)
    time_bins = np.linspace(tmin, tmax, num=time_npix, endpoint=True)
    freq_bins = np.linspace(fmin, fmax, num=freq_npix, endpoint=True)

    space_fov_x = rve.str2rad(space_fov_x)
    space_fov_y = rve.str2rad(space_fov_y)
    dx = space_fov_x / space_npix_x
    dy = space_fov_y / space_npix_y

    tdom = rve.IRGSpace(time_bins)
    fdom = rve.IRGSpace(freq_bins)
    sdom = ift.RGSpace([space_npix_x, space_npix_y], [dx, dy])
    pdom = rve.PolarizationSpace(kwargs.pop("polarization").split(","))
    dom = pdom, tdom, fdom, sdom

# FIXME Move all sky models to resolve
    if model == "cfm":
        sky_model, additional_operators = _cfm(total_domain=dom, **kwargs)
    elif model == "polynom":
        sky_model, additional_operators = _multi_freq_movie_polynom(dom, **kwargs)
    else:
        raise ValueError("Model '{model}' not known")
# FIXME /Move all sky models to resolve

    assert len(sky_model.target) == 4

    # Normalize
    if prefactor_model == "average all lognormal":
        prefactor = ift.LognormalTransform(
            *_parse_mean_std(kwargs, "prefactor"),
            "sky prefactor",
            0,
        )
        bc = ift.ContractionOperator(sky_model.target, None).adjoint
        norm = normalize(sky_model.target, True, True, True)
        sky_model = (bc @ prefactor) * (norm @ sky_model)

    elif prefactor_model == "average space/time lognormal":
        prefactor = _fdom_prefactor(fdom, **kwargs)
        bc = ift.ContractionOperator(sky_model.target, (0, 1, 3)).adjoint
        norm = normalize(sky_model.target, True, False, True)
        sky_model = (bc @ prefactor) * (norm @ sky_model)
    elif prefactor_model == 'None': #FIXME should this be converted to None?
        pass
    elif prefactor_model == 'normalized': #FIXME should this be converted to None?
        norm = normalize(sky_model.target, True, True, True)
        sky_model = norm @ sky_model
    else:
        raise NotImplementedError(f"Do not know prefactor model '{prefactor_model}'")
    # /Normalize

    return sky_model, additional_operators


def normalize(domain, normalize_time, normalize_freq, normalize_space):
    domain = ift.makeDomain(domain)
    # assert domain[0].shape == (1,)
    newdom = list(domain[1:])

    trivial_time = newdom[0].shape == (1,)
    trivial_freq = newdom[1].shape == (1,)

    if trivial_time and trivial_freq:
        newdom = newdom[2]
    elif trivial_time:
        newdom = newdom[1:]
    elif trivial_freq:
        newdom = (newdom[0], newdom[2])

    inp = ift.ScalingOperator(domain, 1)

    pdom, tdom, fdom, sdom = domain
    ntime = tdom.size
    nfreq = fdom.size

    # Divide by sum of all Stokes I, divide by spatial volume
    if pdom.size != 1: # FIXME something fishy here
        broadcast = ift.ContractionOperator(domain, spaces=None).adjoint 
        I_extract = np.ones(domain.shape)
        I_extract[0] = 0
        I_extract = ift.Field(domain,I_extract)
        I_extract = ift.MaskOperator(I_extract)
        return inp*(broadcast @ (I_extract @ inp).sum().scale(sdom.scalar_dvol/ntime/nfreq).ptw('reciprocal'))

    # 1 Jy = 1e-26 W / m² / Hz = 1e-26 J / s / m² / Hz
    # m² refers to the antenna collection area
    # Jy is flux integrated over sky

    # Standard resolve unit: Jy/sr -> differential in spatial dimensions
    # Jy is already differential in freq and time
    # Unit of multifrequency movies: Jy/sr
    # Because then: integral over time and freq and space -> J / m²

    # But also: we want that single-time, single-freq reconstructions have the
    # same numbers as multifreq movies

    if normalize_time and normalize_freq and normalize_space:
        broadcast = ift.ContractionOperator(domain, spaces=None).adjoint
        return inp*(broadcast @ inp.sum().scale(sdom.scalar_dvol/ntime/nfreq).ptw('reciprocal'))
        # return inp*(broadcast @ inp.sum().scale(sdom.scalar_dvol).ptw('reciprocal'))
    if normalize_space:
        broadcast = ift.ContractionOperator(domain, spaces=3).adjoint
        return inp*(broadcast @ inp.integrate(3).ptw('reciprocal'))
    raise NotImplementedError


def _cfm(*, total_domain, **kwargs):
    from ducc0.fft import good_size

    # Prepare domains
    pdom, tdom, fdom, sdom = total_domain
    total_domain_padded = list(total_domain)
    with_time = tdom.shape[0] > 1
    if with_time:
        dt = np.diff(tdom.coordinates)
        try:
            np.testing.assert_allclose(
                dt, dt[0]
            )  # This model work only for equidistant time
        except AssertionError:
            s = "The Correlated Field model works only if the time domain is equispaced."
            raise AssertionError(s)
        dt = dt[0]
        nt = good_size(int(np.round(tdom.shape[0] * kwargs.pop("time_zero_padding_factor"))))
        rg_time = ift.RGSpace(nt, dt)
        padded_tdom = rve.IRGSpace(np.arange(nt) * dt + tdom.coordinates[0])
        total_domain_padded[1] = padded_tdom

        foo = rg_time.distances[0] / 3600
        if foo > 1.0:
            foo = f"{foo:.1f} h"
        else:
            foo = f"{foo*60:.1f} min"
    else:
        rg_time = tdom

    with_freq = fdom.shape[0] > 1
    if with_freq:
        df = np.diff(fdom.coordinates)
        try:
            np.testing.assert_allclose(
                df, df[0]
            )  # This model work only for equidistant time
        except AssertionError:
            s = "The Correlated Field model works only if the frequency domain is equispaced."
            raise AssertionError(s)
        df = df[0]
        nf = good_size(int(np.round(fdom.shape[0] * kwargs.pop("freq_zero_padding_factor"))))
        rg_freq = ift.RGSpace(nf, df)
        padded_fdom = rve.IRGSpace(np.arange(nf) * df + fdom.coordinates[0])
        total_domain_padded[2] = padded_fdom
    else:
        rg_freq = fdom
    # /Prepare domains

    # Assemble operator parts
    additional = {}
    logsky = {}
    for lbl in pdom.labels:
        lbl = lbl.upper()

        # CFM
        cfg_zm = {
            "offset_mean": kwargs[f"stokes{lbl}_zero_mode_offset"],
            "offset_std": _parse_mean_std(kwargs, f"stokes{lbl}_zero_mode"),
        }

        cfg_time = {}
        cfg_space = {}
        cfg_freq = {}
        for kk in ["fluctuations", "flexibility", "asperity", "loglogavgslope"]:
            cfg_space[kk] = _parse_mean_std(kwargs, f"stokes{lbl}_space_{kk}")
            if with_time:
                cfg_time[kk] = _parse_mean_std(kwargs, f"stokes{lbl}_time_{kk}")
            if with_freq:
                cfg_freq[kk] = _parse_mean_std(kwargs, f"stokes{lbl}_freq_{kk}")

        cfm = ift.CorrelatedFieldMaker(f"stokes{lbl} ")
        if with_time:
            cfm.add_fluctuations(rg_time, **cfg_time, prefix="time ")
        if with_freq:
            cfm.add_fluctuations(rg_freq, **cfg_freq, prefix="freq ")
        cfm.add_fluctuations(sdom, **cfg_space, prefix="space ")
        cfm.set_amplitude_total_offset(**cfg_zm)
        op = cfm.finalize(0)
        additional['raw sky'] = op
        # /CFM

        # Zero-padding
        op = op.ducktape_left((rg_time, rg_freq, sdom))
        if op.target[0].size > 1:
            tmpdom = ift.RGSpace(tdom.shape)
            tmpfdom = op.target[1]
            zeropadder = ift.FieldZeroPadder(
                (tmpdom, tmpfdom, sdom), rg_time.shape, space=0
            ).adjoint
            op = zeropadder.ducktape(op.target) @ op
        if op.target[1].size > 1:
            tmpdom = ift.RGSpace(fdom.shape)
            zeropadder = ift.FieldZeroPadder(
                (tdom, tmpdom, sdom), rg_freq.shape, space=1
            ).adjoint
            op = zeropadder.ducktape(op.target) @ op
        # /Zero-padding

        logsky[lbl] = op.ducktape_left((tdom, fdom, sdom))

        normampl = list(cfm.get_normalized_amplitudes())
        if with_freq:
            additional[f"stokes{lbl} freq normalized power spectrum"] = normampl.pop(0) ** 2
        if with_time:
            additional[f"stokes{lbl} time normalized power spectrum"] = normampl.pop(0) ** 2
        additional[f"stokes{lbl} space normalized power spectrum"] = normampl.pop(0) ** 2
    # /Assemble operator parts

    logsky = reduce(add, (oo.ducktape_left(lbl) for lbl, oo in logsky.items()))
    mexp = rve.polarization_matrix_exponential_mf2f(logsky.target, nthreads=ift.nthreads())
    
    sky = mexp @ logsky

    sky = sky.ducktape_left(total_domain)

    return sky, additional


def _parse_mean_std(dct, key):
    res = dct[f"{key}_mean"], dct[f"{key}_std"]
    if res == ("None", "None") or res == (None, None):
        return None
    return res


def _multi_freq_movie_polynom(dom, cfg_section):
    from ducc0.fft import good_size

    pdom, tdom, fdom, sdom = dom
    with_time = tdom.shape[0] > 1
    total_domain_padded = list(dom)

    if with_time:
        dt = np.diff(tdom.coordinates)
        try:
            np.testing.assert_allclose(dt, dt[0])  # This model work only for equidistant time
        except AssertionError:
            s = "The Correlated Field model works only if the time domain is equispaced."
            raise AssertionError(s)
        dt = dt[0]
        nt = good_size(int(np.round(tdom.shape[0]*cfg_section.getfloat("time zero-padding factor"))))
        rg_time = ift.RGSpace(nt, dt)
        padded_tdom = rve.IRGSpace(np.arange(nt)*dt + tdom.coordinates[0])
        total_domain_padded[1] = padded_tdom
        print(f"Time domain (dt={rg_time.distances[0]/3600/24}days):")
        print(rg_time)
    else:
        rg_time = tdom

    ops = {}
    for key in ["i0", "alpha", "beta"]:
        cfg_zm = {
            "offset_mean": cfg_section.getfloat(f"{key} zero mode offset"),
            "offset_std": _parse_mean_std(cfg_section, f"{key} zero mode"),
        }
        cfg_space, cfg_time = {}, {}
        for kk in ["fluctuations", "flexibility", "asperity", "loglogavgslope"]:
            cfg_space[kk] = _parse_mean_std(cfg_section, f"{key} space {kk}")
            if with_time:
                cfg_time[kk] = _parse_mean_std(cfg_section, f"{key} time {kk}")
        cfm = ift.CorrelatedFieldMaker(key)
        if with_time:
            cfm.add_fluctuations(rg_time, **cfg_time, prefix="time ")
        cfm.add_fluctuations(sdom, **cfg_space, prefix="space ")
        cfm.set_amplitude_total_offset(**cfg_zm)
        op = cfm.finalize(0)
        assert len(op.target.shape) == 3 if with_time else 2
        # Remove padded area
        if with_time and op.target[1].size > 1:
            tmpdom = ift.RGSpace(tdom.shape)
            zeropadder = ift.FieldZeroPadder((tmpdom, sdom), rg_time.shape, space=0).adjoint
            op = zeropadder.ducktape(op.target) @ op
            op = op.ducktape_left((tdom, sdom))
        if not with_time:
            op = op.ducktape_left((tdom,) + tuple(op.target))
        # /Remove padded area
        ops[key] = op
    additional_operators = ops

    freq = np.array(fdom.coordinates)
    freq0 = freq.mean()
    freq = ift.makeField(fdom, freq)

    cfm_expander = ift.ContractionOperator((tdom, fdom, sdom), 1).adjoint
    freq_expander = ift.ContractionOperator((tdom, fdom, sdom), (0, 2)).adjoint
    i0 = cfm_expander @ ops["i0"]
    alpha = cfm_expander @ ops["alpha"]
    beta = cfm_expander @ ops["beta"]

    normalized_freq = freq_expander(ift.log(freq/freq0))

    logsky = i0 + ift.makeOp(normalized_freq) @ alpha + ift.makeOp(normalized_freq**2) @ beta


    sky = logsky.exp()

    sky = sky.ducktape_left(dom)

    return sky, additional_operators


def _fdom_prefactor(fdom, cfg_section, order=2):
    freq = np.array(fdom.coordinates)
    freq0 = freq.mean()
    freq = ift.makeField(fdom, freq)
    normalized_freq = ift.log(freq/freq0)
    freq_expander = ift.ContractionOperator(fdom, None).adjoint

    c = [freq_expander @ ift.NormalTransform(cfg_section.getfloat(f"prefactor c{ii} mean"),
                                             cfg_section.getfloat(f"prefactor c{ii} stddev"),
                                             f"prefactor c{ii}") for ii in range(order+1)]

    x = normalized_freq
    logop = reduce(add, (ift.makeOp(x**iorder) @ cc for iorder, cc in enumerate(c)))

    op = logop.exp()
    assert op.target == ift.makeDomain(fdom)
    return op
