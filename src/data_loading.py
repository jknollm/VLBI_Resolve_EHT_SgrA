# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Jakob Knollmüller


import os
from functools import lru_cache

import numpy as np
import resolve as rve
import pandas as pd

import fnmatch

from .utilities import get_files_in_folder
from .load_via_ehtim import load_uvfits


def filter_files(directory, file_filter):
    files = get_files_in_folder(directory)
    if file_filter is not None:
        direc = os.path.split(files[0])[0]
        files = [os.path.split(ff)[1] for ff in files]
        files = fnmatch.filter(files, file_filter)
        files = [os.path.join(direc, ff) for ff in files]
    return files

@lru_cache(maxsize=None) 
def load_observations(*,file_filter, minimum_antenna_distance,
                        minimum_number_of_data_points_in_time_averaging_bin, load_polarization, directory,
                        gap_time_for_averaging, **kwargs):
    double_precision = True
    files = filter_files(directory, file_filter)

    obs0 = []
    for ff in files:
        if os.path.splitext(ff)[1] == ".uvfits" or os.path.splitext(ff)[1] == ".uvf":
            obs0.append(load_uvfits(ff, load_polarization))
        elif os.path.splitext(ff)[1] == ".npz":
            obs0.append(rve.Observation.load(ff))
        else:
            raise RuntimeError(f"File format not known: {ff}")

    obs = []

    for oo in obs0:
        # Flag short baselines
        min_dist = minimum_antenna_distance
        if min_dist is not None:
            coords = oo.auxiliary_table("ANTENNA")["POSITION"]
            for ant1 in range(coords.shape[0]):
                for ant2 in range(ant1, coords.shape[0]):
                    dist = np.linalg.norm(coords[ant1] - coords[ant2])
                    if dist < min_dist:
                        oo = oo.flag_baseline(ant1, ant2)
        # /Flag short baselines

        # Prune empty rows
        oo = oo[~np.any(oo.weight.val == 0., (0,2))]

        assert np.all(oo.weight.val > 0.)
        # /Prune empty rows

        # Average scans
        ts_per_bin = minimum_number_of_data_points_in_time_averaging_bin
        gap_time = gap_time_for_averaging
        list_of_timebins = rve.fair_share_averaging(ts_per_bin, oo.time, gap_time)
        oo = oo.time_average(list_of_timebins)
        # /Average scans

        ind = np.lexsort((oo.ant2, oo.ant1, oo.time))
        oo = oo[ind]
        if double_precision:
            oo = oo.to_double_precision()
        obs.append(oo)
    return tuple(obs)

def load_alma_lightcurves(*, directory, folder, day, **kwargs):
    full_dir = os.path.join(directory, folder)
    files = filter_files(full_dir, f"ALMA_lc_hi_{day}.h5")
    dfs = []
    for f in files:
        df = pd.read_hdf(f, key='data')
        dfs.append(df)
    return dfs
    



