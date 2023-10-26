# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Jakob Knollmüller

import os
import faulthandler; faulthandler.enable()
from argparse import ArgumentParser
from functools import partial
from operator import add


import matplotlib.pyplot as plt
import nifty8 as ift
import numpy as np
import resolve as rve
from resolve.mpi import master

import src


import ehtplot.theme
cmap = 'afmhot_u'



def main():
    parser = ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument(
        "-j", type=int, default=1, help="Number of threads for thread parallelization"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--terminate", type=int)
    args = parser.parse_args()

    nthreads = args.j
    ift.set_nthreads(nthreads)

    cfg = src.parse_config_file(args.config_file)


    if args.terminate is None:
        terminate_callback = lambda iglobal: False
    else:
        terminate_callback = lambda iglobal: iglobal == args.terminate

    builders_dct = src.get_builders_dct()
    optimize_kl_config = ift.OptimizeKLConfig(cfg, builders_dct)

    # Seeds
    seed = cfg["optimization"].getint("random seed")
    if seed is not None:
        ift.random.push_sseq_from_seed(seed)
    # /Seeds
    output_directory = cfg["optimization"]["output directory"]
    sky_models = {}

    for key in ["sky_pol", "sky_stokesI", 'sky_res0', 'sky_res1', 'sky_res2']:
        try:    
            sky_models[key]=optimize_kl_config.instantiate_section(key)

        except:
            continue

    optimize_kl_config.optimize_kl(
        export_operator_outputs=sky_models,
        comm=rve.mpi.comm,
        resume=args.resume,
        terminate_callback=terminate_callback,
        inspect_callback=partial(inspect_callback, sky_models, output_directory),
        sanity_checks=False,
    )


def inspect_callback(sky_models, output_directory, sl, iglobal):
    my_sky = None
    for sky_i in sky_models.values():
        try:
            sky_i = ift.FieldAdapter(sky_i.target["sky"], "sky") @ sky_i
            my_sky = sl.average(sky_i.force)
            samps = sl.iterator(sky_i.force)
            sky = sky_i
            break
        except:
            continue
    assert my_sky is not None

    pol = my_sky.domain[0].size > 1
    movie = my_sky.domain[1].size > 1

    if movie:
        if iglobal > 2 and iglobal % 10 == 9:
            foo = os.path.join(output_directory, "movie")
            os.makedirs(foo, exist_ok=True)
            src.plot_movie(os.path.join(foo, "latest_sky"), sky, sl, 12, mean=True)

    if my_sky.domain.shape[:2] == (1, 1):
        skymean = ift.makeField(my_sky.domain[2:],my_sky.val[0,0])

    else:
        skymean = ift.Field(
            ift.DomainTuple.make(my_sky.domain[2:]), my_sky.val[0].mean(axis=(0,))
        )
    maxvals = [skymean.val[kk].max() for kk in range(skymean.shape[0])]
    minmax = min(maxvals)
    from matplotlib.colors import LogNorm

    if master:
        dd = os.path.join(output_directory, "sky_lognorm")
        os.makedirs(dd, exist_ok=True)
        p = ift.Plot()
        for kk in range(skymean.shape[0]):
            sfreq = f"{skymean.domain[0].coordinates[kk]*1e-9:.1f} GHz"
            p.add(
                ift.makeField(my_sky.domain[3], skymean.val[kk]),
                title=sfreq,
                norm=LogNorm(vmax=minmax),
                cmap=cmap
            )
            p.add(
                ift.makeField(my_sky.domain[3], skymean.val[kk]),
                title=sfreq,
                vmax=minmax,
                cmap=cmap
            )
            plt.figure()
            plt.imshow(
                np.rot90(skymean.val[kk]), norm=LogNorm(vmax=minmax), cmap=cmap
            )
            plt.title(sfreq)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(
                dd + f"/sky_log_{skymean.domain[0].coordinates[kk]*1e-9:.0f}.png",
                bbox_inches="tight",
            )
            plt.figure()
            plt.imshow(np.rot90(skymean.val[kk]), vmax=0.9 * minmax, cmap=cmap)
            plt.title(sfreq)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(
                dd + f"/sky_{skymean.domain[0].coordinates[kk]*1e-9:.0f}.png",
                bbox_inches="tight",
            )
        p.output(name=os.path.join(dd, f"sky{iglobal}.png"))
        plt.close("all")


if __name__ == "__main__":
    main()
