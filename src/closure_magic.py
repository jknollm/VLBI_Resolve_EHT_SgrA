# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Philipp Frank, Philipp Haim, Reimar Leike,
# Jakob Knollmueller

from itertools import combinations

import nifty8 as ift
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import aslinearoperator


MAXSHORTBASELINE = 2000000


def Visibilities2ClosureMat(d, amplitudes, systematics):
    rows, cols, values = [], [], []
    offset_closure, offset_vis = 0, 0
    if not amplitudes:
        rowsp, colsp, valuesp = [], [], []
        offset_closurep, offset_visp = 0, 0
    evals = []

    # Footnote 8 in https://iopscience.iop.org/article/10.3847/1538-4357/ab8469

    relsigma = (abs(d['vis'])/(d['sigma'] + systematics * abs(d['vis'])))**2
    relweights = 1/relsigma

    timestamps = []
    for tt in np.unique(d['time']):
        ind = tt == d['time']
        aa1, aa2 = d['ant1'][ind], d['ant2'][ind]
        nstations = len(set(aa1) | set(aa2))
        tm, missing_inds = insert_missing_baselines_into_weights(aa1, aa2, np.ones(sum(ind)))
        tm = np.diag(tm)
        if nstations < (4 if amplitudes else 3):
            offset_vis += sum(ind)
            continue
        if amplitudes:
            psi = closure_amplitude_design_matrix(nstations)
            nontrivial = remove_short_diagonals(psi, aa1, aa2, d['uv'][ind])   
            psi = psi @ tm
            goodclosures = np.logical_and(np.sum(psi != 0, axis=1) == 4, nontrivial)
            psi = psi[goodclosures]
            if psi.shape[0] == 0:
                offset_vis += sum(ind)
                continue

            psi = np.delete(psi, missing_inds, axis=1)
            psi = to_nonredundant(psi)

            mdecomp = psi
        else:
            psi = closure_phase_design_matrix(nstations) @ tm
            psi = psi[np.sum(psi != 0, axis=1) == 3]
            if psi.shape[0] == 0:
                offset_vis += sum(ind)
                continue
            psi = np.delete(psi, missing_inds, axis=1)
            psi = to_nonredundant(psi)
            mdecomp = np.diag(1.j*np.exp(1.j*psi @ np.log(d['vis'][ind]).imag)) @ psi

        U, ivsq = get_decomp(mdecomp, relweights[ind])
        evals += list(1./ivsq.diagonal())
        if amplitudes:
            psi = ivsq @ U @ psi
        else:
            proj = ivsq @ U

        for i in range(psi.shape[0]):
            row = psi[i]
            for c in range(psi.shape[1]):
                rows += [i + offset_closure]
                cols += [c + offset_vis]
                values += [row[c]]
        offset_closure += psi.shape[0]
        offset_vis += psi.shape[1]
        if not amplitudes:
            for i in range(proj.shape[0]):
                row = proj[i]
                for c in range(proj.shape[1]):
                    rowsp += [i + offset_closurep]
                    colsp += [c + offset_visp]
                    valuesp += [row[c]]
            offset_closurep += proj.shape[0]
            offset_visp += proj.shape[1]
            timestamps += [tt,]*proj.shape[0]
        else:
            timestamps += [tt,]*psi.shape[0]

    smo = SparseMatrixOperator((values, (rows, cols)), (offset_closure, offset_vis))
    evals = ift.makeField(smo.target, np.array(evals))
    if amplitudes:
        timestamps = ift.makeField(smo.target, np.array(timestamps))
        return smo, evals, timestamps
    clos2eig = SparseMatrixOperator((valuesp, (rowsp, colsp)), (offset_closurep, offset_visp))
    timestamps = ift.makeField(clos2eig.target, np.array(timestamps))
    return smo, clos2eig, evals, timestamps


class SparseMatrixOperator(ift.LinearOperator):
    def __init__(self, arg1, shape):
        assert len(shape) == 2
        self._smat = aslinearoperator(coo_matrix(arg1, shape=shape))
        self._domain = ift.makeDomain(ift.UnstructuredDomain(shape[1]))
        self._target = ift.makeDomain(ift.UnstructuredDomain(shape[0]))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        f = self._smat.matvec if mode == self.TIMES else self._smat.rmatvec
        return ift.makeField(self._tgt(mode), f(x.val))


def to_nonredundant(mat):
    rnk = np.linalg.matrix_rank(mat)
    result = mat[0:1]
    current_rank = 1
    for i in range(1, mat.shape[0]):
        if current_rank >= rnk:
            break
        tmp = np.append(result, mat[i:i+1], axis=0)
        new_rank = np.linalg.matrix_rank(tmp)
        if new_rank > current_rank:
            result = tmp
            current_rank = new_rank
    return result


def insert_missing_baselines_into_weights(ants1, ants2, weights):
    weights = np.copy(weights)
    missing_inds = []
    aa = set(ants1) | set(ants2)
    if binom2(len(aa)) != len(weights):
        baselines = list(zip(ants1, ants2))
        ants = np.sort(list(aa))
        counter, missing_inds = 0, []
        for ii, xx in enumerate(ants):
            for yy in ants[ii + 1:]:
                if (xx, yy) not in baselines:
                    missing_inds.append(counter)
                counter += 1
        for ii in missing_inds:
            weights = np.insert(weights, ii, 0)
    return weights, missing_inds


def get_decomp(psi, diag, triv_cutoff=1e-9):
    m = psi@np.diag(diag)@psi.conj().T 
    ev, eh = np.linalg.eigh(m)
    inds = ev > triv_cutoff
    U = eh.conj().T[inds]
    ivsq = np.diag(1./np.sqrt(ev[inds]))
    assert np.allclose(np.eye(U.shape[0]), ivsq @ U @ m @ U.conj().T @ ivsq) 
    assert ivsq.dtype == float
    return U, ivsq


def remove_short_diagonals(psi, aa1, aa2, uv):
    assert aa1.shape == aa2.shape
    assert (aa1.size, 2) == uv.shape
    assert psi.shape[1] >= aa1.size

    # gets an array with all possible baselines where there is a 0 if the
    # baselines is missing and ii+1 if it is the ii-th in the data
    missing_baseline_index, _ = insert_missing_baselines_into_weights(aa1, aa2, np.arange(len(aa1))+1)
    nontrivial = np.ones(psi.shape[0], dtype=bool)
    for ii in range(len(nontrivial)):
        bls = np.where(psi[ii] != 0)[0]
        bls = missing_baseline_index[bls]
        if np.any(0 == bls):
            continue
        bls = (bls-1).astype(np.int64)
        b0 = bls[0]
        for b in bls[1:]:
            if aa1[b] == aa1[b0] or aa2[b] == aa2[b0]:
                if np.linalg.norm(uv[b0]-uv[b]) < MAXSHORTBASELINE:
                    nontrivial[ii] = False
            if aa2[b] == aa1[b0] or aa1[b] == aa2[b0]:
                if np.linalg.norm(uv[b0]+uv[b]) < MAXSHORTBASELINE:
                    nontrivial[ii] = False
    return nontrivial


def visibility_design_matrix(n):
    if n < 2:
        raise ValueError
    lst = range(binom2(n))
    x = np.zeros((binom2(n), n), dtype=int)
    x[(lst, [ii for ii in range(n) for _ in range(ii + 1, n)])] = 1
    x[(lst, [jj for ii in range(n) for jj in range(ii + 1, n)])] = -1
    return x


def closure_phase_design_matrix(n):
    if n < 3:
        raise ValueError
    x = np.zeros((binom3(n), binom2(n)), dtype=int)
    n = n - 1
    vdm = visibility_design_matrix(n)
    nb = vdm.shape[0]
    x[:nb, :n] = vdm
    x[:nb, n:] = np.diag(np.ones(nb))
    if nb > 1:
        x[nb:, n:] = closure_phase_design_matrix(n)
    return x


def closure_amplitude_design_matrix(n):
    if n < 4:
        raise ValueError
    m = 3
    block = np.zeros((m, 6), dtype=int)
    block[([0, 0, 1, 1, 2, 2], [0, 5, 0, 5, 2, 3])] = 1
    block[([0, 0, 1, 1, 2, 2], [1, 4, 2, 3, 1, 4])] = -1
    bl = baselines(range(n))
    x = np.zeros((m*binom4(n), binom2(n)))
    for ii, ants in enumerate(combinations(range(n), 4)):
        inds = [bl.index(bb) for bb in baselines(ants)]
        x[ii*m:(ii + 1)*m, inds] = block
    return x


def baselines(alst):
    return [(aa, bb) for ii, aa in enumerate(alst) for bb in alst[ii + 1:]]


def binom2(n):
    return (n*(n-1))//2


def binom3(n):
    return (n*(n-1)*(n-2))//6


def binom4(n):
    return (n*(n-1)*(n-2)*(n-3))//24
