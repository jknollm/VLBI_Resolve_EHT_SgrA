# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Jakob Knollmüller


import resolve as rve
import nifty8 as ift
from functools import reduce
from operator import add
import numpy as np

def calibration_op(obs, **kwargs):
    res = []

    for ii, oo in enumerate(obs):
        utimes = rve.unique_times(oo)
        uants = oo.antenna_positions.unique_antennas()
        dom = [oo.polarization.space] + [
            ift.UnstructuredDomain(nn) for nn in [len(uants), len(utimes), oo.nfreq]
        ]
        uants = rve.unique_antennas(oo)
        time_dct = {aa: ii for ii, aa in enumerate(utimes)}
        antenna_dct = {aa: ii for ii, aa in enumerate(uants)}

        antenna_stds = []
        stations = oo.auxiliary_table("ANTENNA")["STATION"]
        for uant in list(uants):
            antenna_stds.append(kwargs[f"{stations[uant]}_std"])

        AntennaExpander = ift.ContractionOperator(dom, spaces=(0, 2, 3)).adjoint
        STDs = ift.makeOp(
            AntennaExpander(
                ift.makeField(AntennaExpander.domain, np.array(antenna_stds))
            )
        )
        inp = ift.ScalingOperator(dom, 1.0).ducktape(f"amplitude calib xi #{ii}")

        cop1 = rve.CalibrationDistributor(
            dom, oo.vis.domain, oo.ant1, oo.time, antenna_dct, time_dct
        )
        cop2 = rve.CalibrationDistributor(
            dom, oo.vis.domain, oo.ant2, oo.time, antenna_dct, time_dct
        )
        cop = (cop1 + cop2) @ STDs @ inp
        res.append(cop.ducktape_left(f"cal #{ii}"))
    return reduce(add, res)


def simple_calibration_op(obs,std, **kwargs):
    res = []
    for ii, oo in enumerate(obs):
        utimes = rve.unique_times(oo)
        uants = oo.antenna_positions.unique_antennas()
        dom = [oo.polarization.space] + [
            ift.UnstructuredDomain(nn) for nn in [len(uants), len(utimes), oo.nfreq]
        ]
        uants = rve.unique_antennas(oo)
        time_dct = {aa: ii for ii, aa in enumerate(utimes)}
        antenna_dct = {aa: ii for ii, aa in enumerate(uants)}
        inp = ift.ScalingOperator(dom, 1.0).ducktape(f"amplitude calib xi #{ii}")

        cop1 = rve.CalibrationDistributor(
            dom, oo.vis.domain, oo.ant1, oo.time, antenna_dct, time_dct
        )
        cop2 = rve.CalibrationDistributor(
            dom, oo.vis.domain, oo.ant2, oo.time, antenna_dct, time_dct
        )
        cop = (cop1 + cop2) @ (std * inp)
        res.append(cop.ducktape_left(f"cal #{ii}"))
    return reduce(add, res)