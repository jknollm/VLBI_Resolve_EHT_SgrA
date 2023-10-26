# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Jakob Knollmüller

import nifty8 as ift
import resolve as rve
from .likelihoods import get_initial_likelihood
import numpy as np
from matplotlib import pyplot as plt


def transition_initial(
    target_likelihood,
    total_iterations,
    n_samples,
    kl_minimizer,
    sampling_iteration_controller,
    nonlinear_sampling_minimizer,
    **kwargs,
):
    likelihood_energy = get_initial_likelihood(**kwargs)
    _domain = ift.makeDomain({})
    _target = target_likelihood.domain

    def initial_trans(x):
        _, mean = ift.optimize_kl(
            likelihood_energy,
            total_iterations,
            n_samples,
            kl_minimizer,
            sampling_iteration_controller,
            nonlinear_sampling_minimizer,
            constants=["stokesI space loglogavgslope","stokesI time fluctuations","stokesI time loglogavgslope"],
            comm=rve.mpi.comm,
            return_final_position=True,
        )

        return ift.MultiField.union(
                [0.01 * ift.from_random(_target), mean])
    return initial_trans  

def transition_zoom(
    sky_old,
    sky_new,
    target_likelihood,
    domain_likelihood,
    total_iterations,
    n_samples,
    kl_minimizer,
    sampling_iteration_controller,
    nonlinear_sampling_minimizer,
    initial_max_std,
    output_directory,
    **kwargs,
):
    from scipy.ndimage import zoom
    _domain = domain_likelihood.domain
    _target = target_likelihood.domain

    def zoom_trans(x):
        template = x.average(sky_old.force).val["sky"]
        One = ift.ScalingOperator(x.domain,1)
        old_pos =  x.average(One.force)
        zoom_facs = [
            a / b for a, b in zip(sky_new.target["sky"].shape, template.shape)
        ]
        rescaled_template = zoom(template, zoom_facs)
        # rescaled_template[rescaled_template < 0.1*rescaled_template.max()] = 0.001*rescaled_template.max()
        # smoothed_template = gaussian_filter(rescaled_template, 1)


        prof = ift.makeField(sky_new.target, {"sky": rescaled_template})
        cmax = prof.val["sky"].max()
        N = ift.ScalingOperator(
            prof.domain, (initial_max_std * cmax) ** 2, sampling_dtype=np.float64
        )
        def inspect_callback(samps, iglobal):
            import os
            sky = samps.average(sky_new)['sky'].val.mean(axis=(0,1,2))
            plt.figure()
            plt.imshow(np.rot90(sky),cmap=cmap)
            path = os.path.join(output_directory, "transitions")
            os.makedirs(path, exist_ok = True)
            plt.savefig(os.path.join(path,f'trans_{iglobal}.png'))
            plt.close()
        my_d = prof + N.draw_sample()
        keep_vals = ["stokesI space loglogavgslope","stokesI time fluctuations","stokesI time loglogavgslope"]
        likelihood_energy = ift.GaussianEnergy(my_d, N.inverse) @ sky_new
        initial_position = 0.1*ift.from_random(likelihood_energy.domain)
        initial_dict = initial_position.to_dict()
        for key in keep_vals:
            initial_dict[key] = old_pos[key]
        initial_position = ift.MultiField.from_dict(initial_dict)
        _, mean = ift.optimize_kl(
            likelihood_energy,
            total_iterations,
            n_samples,
            kl_minimizer,
            sampling_iteration_controller,
            nonlinear_sampling_minimizer,
            initial_position = initial_position,
            inspect_callback=inspect_callback,
            constants=keep_vals,
            comm=rve.mpi.comm,
            return_final_position=True,
        )
        new_pos = old_pos.to_dict()
        for key in mean.keys():
            new_pos[key] = mean[key]
        new_pos = ift.MultiField.from_dict(new_pos)
        assert new_pos.domain == _target
        return new_pos

    return zoom_trans
