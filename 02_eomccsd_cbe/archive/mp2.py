#!/usr/bin/env python
from pyscf.pbc import mp
from fileutils import load
from support import load_mf
import sys
import h5py
import numpy as np
from functools import reduce

material = load("data/{}".format(sys.argv[1]))
mymf = load_mf(sys.argv[1])
# Check if material["frozen"] is an empty list
mypt = mp.KMP2(mymf, frozen=material["frozen"])
ekmp2, t2 = mypt.kernel()

mo_energy = mymf.mo_energy
mo_coeff = mymf.mo_coeff
nocc = mypt.nocc
nvir_act = material["nvir_act"]
dm = mypt.make_rdm1()
nkpts = mypt.nkpts
no_coeff = []

for k in range(nkpts):
    n, v = np.linalg.eigh(dm[k][nocc:, nocc:])
    idx = np.argsort(n)[::-1]
    n, v = n[idx], v[:, idx]
    fvv = np.diag(mo_energy[k][nocc:])
    fvv_no = reduce(np.dot, (v.T.conj(), fvv, v))
    _, v_canon = np.linalg.eigh(fvv_no[:nvir_act,:nvir_act])
    no_coeff_1 = reduce(np.dot, (mo_coeff[k][:, nocc:], v[:, :nvir_act], v_canon))
    no_coeff_2 = np.dot(mo_coeff[k][:, nocc:], v[:, nvir_act:])
    no_coeff_k = np.concatenate((mo_coeff[k][:, :nocc], no_coeff_1, no_coeff_2), axis=1)
    no_coeff.append(no_coeff_k)

with h5py.File("data/{}".format(material["imds"]), 'w') as fout: # output
    fout.create_dataset('no_coeff', data=no_coeff)
