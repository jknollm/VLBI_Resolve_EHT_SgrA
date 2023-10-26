# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Jakob Knollmüller


from .data_loading import load_observations, load_alma_lightcurves
from .likelihoods import lhmix, likelihood_calib_logamplitudes, likelihood_closure_amplitudes, likelihood_closure_phases, likelihood_alma_lc
from .transitions import transition_initial, transition_zoom
from .sky_model import sky
from .calibration import calibration_op, simple_calibration_op


def get_builders_dct():
    dct =  {
        "data_stokesI" : load_observations,
        "alma_lightcurves" : load_alma_lightcurves,
        "lhmix_stokesI": lhmix,
        "lhcalib_stokesI": lhmix,
        "initial": transition_initial,
        "zoom0": transition_zoom,
        "zoom00": transition_zoom,
        "zoom01": transition_zoom,
        "zoom1": transition_zoom,
        "zoom12": transition_zoom,
        "zoom2": transition_zoom,
        "zoom02": transition_zoom,
        "sky_stokesI": sky,
        "sky_res0": sky,
        "sky_res1": sky,
        "sky_res2": sky,
        "sky_pol": sky,
        "lhmix_res0": lhmix,
        "lhmix_res1": lhmix,
        "lhmix_res2": lhmix,
        "lhcalib_res0": lhmix,
        "lhcalib_res1": lhmix,
        "lhcalib_res2": lhmix,
        "lh_alma_lightcurves0": likelihood_alma_lc,
        "lh_alma_lightcurves1": likelihood_alma_lc,
        "lh_alma_lightcurves2": likelihood_alma_lc,
        "calibration.independentAmplitudes_stokesI": calibration_op,
        "calibration.simpleAmplitudes_stokesI": simple_calibration_op,
        "lh_closure_phases_stokesI": likelihood_closure_phases,
        "lh_closure_amplitudes_stokesI": likelihood_closure_amplitudes,
        "lh_calib_logamplitudes_stokesI": likelihood_calib_logamplitudes,

        "lh_closure_phases_stokesI_res0": likelihood_closure_phases,
        "lh_closure_amplitudes_stokesI_res0": likelihood_closure_amplitudes,
        "lh_calib_logamplitudes_stokesI_res0": likelihood_calib_logamplitudes,

        "lh_closure_phases_stokesI_res1": likelihood_closure_phases,
        "lh_closure_amplitudes_stokesI_res1": likelihood_closure_amplitudes,
        "lh_calib_logamplitudes_stokesI_res1": likelihood_calib_logamplitudes,

        "lh_closure_phases_stokesI_res2": likelihood_closure_phases,
        "lh_closure_amplitudes_stokesI_res2": likelihood_closure_amplitudes,
        "lh_calib_logamplitudes_stokesI_res2": likelihood_calib_logamplitudes,
    }
    return dct