# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Jakob Knollmüller

import numpy as np
import nifty8 as ift
from scipy.stats import multivariate_normal
from configparser import ConfigParser
import os
def profile_operator(op, ntries, position=None):
    import cProfile
    import io
    import pstats
    from pstats import SortKey

    if position is None:
        position = ift.full(op.domain, 1.2)
    with cProfile.Profile() as pr:
        for ii in range(ntries):
            op(position)

    s = io.StringIO()
    sortby = SortKey.TIME
    pstats.Stats(pr, stream=s).sort_stats(sortby).print_stats(10)
    return s.getvalue()

# Compute initial position
def gaussian_profile(dom, loc=(0,0),stds=(1,1), angle=0):
    dom = ift.makeDomain(dom)
    assert len(dom) == 1
    assert len(loc) == 2
    dom = dom[0]
    xf, xp = dom.distances[0]*dom.shape[0]/2, dom.shape[0]
    yf, yp = dom.distances[1]*dom.shape[1]/2, dom.shape[1]
    xx, yy = np.meshgrid(np.linspace(-xf, xf, xp), np.linspace(-yf, yf, yp), indexing='ij')
    pos = np.dstack((xx, yy))
    mean = np.array(loc)
    eigval = np.diag(stds)**2
    eigvec = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    cov = eigvec.T @ eigval @ eigvec
    gauss = multivariate_normal(mean, cov)
    profile = gauss.pdf(pos)
    return ift.makeField(dom, profile)

def disk_profile(dom, loc=(0,0),rad=30, smoothing=1):
    dom = ift.makeDomain(dom)
    assert len(dom) == 1
    assert len(loc) == 2
    dom = dom[0]

    dom = ift.makeDomain(dom)[0]
    xf, xp = dom.distances[0]*dom.shape[0]/2, dom.shape[0]
    yf, yp = dom.distances[1]*dom.shape[1]/2, dom.shape[1]
    xx, yy = np.meshgrid(np.linspace(-xf, xf, xp),
                         np.linspace(-yf, yf, yp),
                         indexing='ij')
    r = np.sqrt((xx-loc[0])**2 + (yy-loc[1])**2)
    beam = sigmoid((np.abs(rad) - np.abs(r))/dom.distances[0]/smoothing)

    return ift.makeField(dom, beam)


def sigmoid(x):
    return .5*(1 + np.tanh(x))


def get_files_in_folder(folder):
    import os

    folder = os.path.expanduser(folder)
    dirpath, _, files = next(os.walk(folder))
    return [os.path.join(dirpath, ff) for ff in files]


def comma_separated_str_to_list(cfg, length, allow_none=False, output_type=None):
    lst = cfg.split(",")
    lst = list(map(_nonestr_to_none, lst))

    if len(lst) == 1:
        lst = length * lst
    # Parse *
    if lst.count("*") > 1:
        raise ValueError("Only one * allowed")
    if len(lst) != length:
        ind = lst.index("*")
        if ind == 0:
            raise ValueError("* at beginning not allowed")
        lst.pop(ind)
        for _ in range(length - len(lst)):
            lst.insert(ind - 1, lst[ind - 1])
    # /Parse *

    if None in lst and not allow_none:
        raise ValueError("None is not allowed")

    if output_type is not None:
        lst = list(map(lambda x: _to_type(x, output_type), lst))

    return lst


def _nonestr_to_none(s):
    if s.lower() in ["none", ""]:
        return None
    return s


def _to_type(obj, output_type):
    if obj is None:
        return None
    return output_type(obj)


def invert_dct(dct):
    assert len(set(dct.values())) == len(dct.values())
    return {vv: kk for kk, vv in dct.items()}

def parse_config_file(config_file):
    if not os.path.isfile(config_file):
        raise RuntimeError(f"Config file {config_file} not found")
    cfg = ConfigParser()
    cfg.optionxform = str  # make keys case-sensitive
    cfg.read(config_file)
    # /Read config file

    # Normalize paths in cfg file
    split_path = os.path.split(config_file)
    cfg_file_folder = split_path[0] if len(split_path)>2 else ''
    cfg["base.data"]["directory"] = rel_to_abs_path(
        cfg_file_folder, cfg["base.data"]["directory"]
    )
    output_directory = rel_to_abs_path(
        cfg_file_folder, cfg["optimization"]["output directory"]
    )
    cfg["optimization"]["output directory"] = output_directory
    return cfg
def rel_to_abs_path(rel_path, where):
    # if rel_path is absolute, return
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.abspath(os.path.join(where, rel_path))


