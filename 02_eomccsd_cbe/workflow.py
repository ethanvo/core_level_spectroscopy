#!/usr/bin/env python3
from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.tools import lattice, pyscf_ase
from fileutils import load, dump
import numpy as np
from functools import reduce
import sys
import os
import h5py
from pyscf.pbc.cc.eom_kccsd_rhf import _IMDS, EOMIP, CVSEOMIP
from pyscf.pbc.cc.kintermediates_rhf import Wovoo, Woooo, Wooov

class _ERIS:
    def __init__(self, cc):
        pass

def load_h5(h5file, key):
    data = {}
    for k, v in h5file[key].items():
        data[k] = v
    return data

au2ev = 27.211386245988

material = load("data/{}".format(sys.argv[1]))
h5file = h5py.File("data/{}".format(material["imds"]), "a")
################################################################################
# Cell
################################################################################
cell = gto.Cell()
ase_atom = lattice.get_ase_atom(material["formula"])
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
cell.a = ase_atom.cell[:]
cell.unit = "B"
cell.basis = material["basis"]
cell.exp_to_discard = material["exp_to_discard"]
cell.verbose = 7
cell.build()

################################################################################
# Mean Field
################################################################################
kmesh = material["kmesh"]
kpts = cell.make_kpts(kmesh, scaled_center=material["vb_scaled_center"])
mymf = scf.KRHF(cell, kpts=kpts, exxdiv="ewald")
mymf.chkfile = "data/{}".format(material["chk"])
mymf = mymf.density_fit()
mymf.with_df._cderi_to_save = "data/{}".format(material["cderi"])
ekrhf = mymf.kernel()
convergence = mymf.converged
material["ekrhf"] = float(ekrhf)
material["mf_convergence"] = bool(convergence)
dump(material, f'data/{material["source"]}')

################################################################################
# Natural Orbital Coefficients
################################################################################
nvir_act = material["nvir_act"]
mo_coeff = mymf.mo_coeff
mo_energy = mymf.mo_energy
mypt = mp.KMP2(mymf, frozen=material["frozen"])
mypt.kernel()
nocc = mypt.nocc
nkpts = mypt.nkpts
dm = mypt.make_rdm1()
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
no_coeff = np.asarray(no_coeff)
h5file.create_dataset('no_coeff', data=no_coeff)
material["Made NOs"] = True

################################################################################
# Coupled Cluster
################################################################################
mycc = cc.KRCCSD(mymf, frozen=material["frozen"], mo_coeff=no_coeff)
t1 = t2 = eris = None
if os.path.isfile(f"data/{material['diis']}"):
    mycc.restore_from_diis(f"data/{material['diis']}")
    t1, t2 = mycc.t1, mycc.t2
mycc.keep_exxdiv = True
# check if eris is already saved
if h5py.get("eris", None) is not None:
    eris = _ERIS(mycc)
    eris_dict = load_h5(h5file, "eris")
    eris.__dict__.update(eris_dict)
else:
    eris = mycc.ao2mo(mo_coeff=no_coeff)
    grp_eris = h5file.create_group("eris")
    for key in ["mo_coeff", "fock"]:
        grp_eris.create_dataset(key, data=getattr(eris, key))
    eris_keys = ["oooo", "ooov", "oovv", "ovov", "voov", "vovv"]
    if getattr(eris, "feri1", None):
        for key in eris_keys:
            eris.feri1.copy(key, grp_eris)
    else:
        for key in eris_keys:
            grp_eris.create_dataset(key, data=getattr(eris, key))
ekrccsd, t1, t2 = mycc.kernel(t1=t1, t2=t2, eris=eris)

################################################################################
# IMDS 1e
################################################################################
imds = _IMDS(mycc, eris)
imds._make_shared_1e()
grp_imds = h5file.create_group("imds")
for k, v in imds.__dict__.items():
    if (k[0] == "F" or k[0] == "L") and v is not None:
        print("IMDS key to save:", k, ", type:", type(v))
        grp_imds.create_dataset(k, data=v)
################################################################################
# IMDS 2e
################################################################################
nkpts, nocc, nvir = t1.shape
ovoo_dest = grp_imds.create_dataset("Wovoo", (nkpts, nkpts, nkpts, nocc, nvir, nocc, nocc), dtype=t1.dtype.char)
ovoo = Wovoo(t1, t2, eris, mycc.khelper.kconserv, ovoo_dest)

oooo_dest = grp_imds.create_dataset("Woooo", (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), dtype=t1.dtype.char)
oooo = Woooo(t1, t2, eris, mycc.khelper.kconserv, oooo_dest)

ooov_dest = grp_imds.create_dataset("Wooov", (nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir), dtype=t1.dtype.char)
ooov = Wooov(t1, t2, eris, mycc.khelper.kconserv, ooov_dest)

imds.Wovoo = ovoo
material["made_wovoo"] = True
imds.Woooo = oooo
material["made_woooo"] = True
imds.Wooov = ooov
material["made_wooov"] = True
################################################################################
# EOM-CCSD
################################################################################
myeom = EOMIP(mycc)
eip, vip = myeom.ipccsd(nroots=material["vb_nroots"], imds=imds, kptlist=[0])
convergence = np.real(myeom.converged) != 0
myeom_data = dict([("eip", eip.tolist()), 
                   ("convergence", convergence.tolist())])
################################################################################
# CVS-EOM-CCSD
################################################################################
myeom = CVSEOMIP(mycc)
myeom.mandatory = material["core"]
eip, vip = myeom.ipccsd(nroots=len(material["core"]), imds=imds, kptlist=[0])
convergence = np.real(myeom.converged) != 0
myeom_data = dict([("eip", eip.tolist()), 
                   ("convergence", convergence.tolist())])