#!/usr/bin/env python
from pyscf import lib
from pyscf.pbc.scf import chkfile
from pyscf.pbc import scf, mp
from fileutils import load, dump
import os
import h5py
from functools import reduce
import numpy as np
import sys

#!/usr/bin/env python
from pyscf.pbc import cc
from fileutils import load, dump
from support import load_mf
import sys
import h5py

material = load("data/{}".format(sys.argv[1]))
mymf = load_mf(sys.argv[1])
mycc = cc.KRCCSD(mymf, frozen=material["frozen"])
mycc.keep_exxdiv = True
ekrmp2, t1, t2 = mycc.kernel(mbpt2=True)
converged = mycc.converged = True
# Check t1, t2 type
print("t1 type: {}".format(type(t1)))
print("t2 type: {}".format(type(t2)))
mycc_data = dict([("ekrmp2", float(ekrmp2)),
                  ("converged", converged)])

dump(mycc_data, "data/{}".format(material["ekrmp2"]))
# Save t1, t2
with h5py.File("data/{}".format(material["imds"]), "w") as fout:
    grp = fout.create_group("t_amps")
    grp.create_dataset("t1", data=t1)
    grp.create_dataset("t2", data=t2)

material = load("data/{}".format(sys.argv[1]))
cell, scfdata = chkfile.load_scf("data/{}".format(material["chk_ip"])) # input
mymf = scf.KRHF(cell, kpts=scfdata['kpts'], exxdiv="ewald")
mymf.__dict__.update(scfdata)
mymf = mymf.density_fit()
mymf.with_df._cderi = "data/{}".format(material["cderi_ip"]) # input
mymf.chkfile = "data/{}".format(material["chk_ip"]) # input
mymf.converged = True

##############################
# K-point MP2
##############################
frozen = np.arange(material["nocc"] + material["mp2_act"], material["nmo"])
if frozen.size == 0:
    frozen = None
mypt = mp.KMP2(mymf, frozen=frozen)
ekmp2, t2 = mypt.kernel()

mo_energy = mymf.mo_energy
mo_coeff = mymf.mo_coeff
nocc = mypt.nocc
mp2_act = material["mp2_act"]
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

with h5py.File("data/{}".format(material["nos"]), 'w') as fout: # output
    fout.create_dataset('no_coeff', data=no_coeff)
