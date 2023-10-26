# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Jakob Knollmüller

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import nifty8 as ift
from os import system
import numpy as np
from resolve.mpi import master


def vis_plot(file_name, vis, weight, u, v):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = list(axs.ravel())
    axx = axs.pop(0)
    axx.hist(vis.real, 100)
    axx.set_title("real")
    axx = axs.pop(0)
    axx.hist(vis.imag, 100)
    axx.set_title("imag")
    axx = axs.pop(0)
    axx.hist(np.abs(vis), 100)
    axx.set_title("abs")
    axx = axs.pop(0)
    axx.set_title("weight")
    axx.hist(weight, 100)

    axx = axs.pop(0)
    axx.set_title("abs")
    axx.scatter(u, v, c=np.abs(vis), vmin=0, vmax=0.25, s=1, cmap="viridis")

    axx = axs.pop(0)
    axx.set_title("real")
    axx.scatter(u, v, c=vis.real, vmin=0, vmax=0.25, s=1, cmap="viridis")
    axx = axs.pop(0)
    axx.set_title("imag")
    axx.scatter(u, v, c=vis.imag, vmin=0, vmax=0.25, s=1, cmap="viridis")

    axx = axs.pop(0)
    axx.set_title("angle")
    axx.scatter(u, v, c=np.angle(vis, deg=True), vmin=0, vmax=360, s=1, cmap="hsv")

    plt.tight_layout()
    if master:
        plt.savefig(file_name)
    plt.close()


def plot_movie(file_name_base, sky, samples, length=2, mean=False):
    movie_samples = list(samples.iterator(sky))
    if mean:
        dom = movie_samples[0].domain
        movie_samples = [ss.val for ss in movie_samples]
        movie_samples = [ift.makeField(dom, np.mean(np.array(movie_samples), axis=0))]

    assert sky.target[0].shape == (1,)
    if master:
        sdom = sky.target[-1]

        nt = sky.target[1].size
        nf = sky.target[2].size

        for t in range(nt):
            p = ift.Plot()
            for ss in movie_samples:
                for f in range(nf):
                    mi, ma = np.min(ss.val), np.max(ss.val)
                    # p.add(ift.makeField(sdom, ss.val[0, t, f]), vmin=mi, vmax=ma)
                    if mean:
                        p.add(ift.makeField(sdom, ss.val[0, t, f]), vmin=mi,vmax=ma,cmap='afmhot')
                        # p.add(ift.makeField(sdom, ss.val[0, t, f]), norm=LogNorm(vmin=mi, vmax=ma))

            p.output(name=f"{file_name_base}_{t:04d}.png", xsize=10, ysize=10)

        framerate = max([int(nt / length), 1])
        system(f"ffmpeg -framerate {framerate} -i {file_name_base}_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 23 -y {file_name_base}.mp4")
        system(f"rm -rf {file_name_base}_*.png")
