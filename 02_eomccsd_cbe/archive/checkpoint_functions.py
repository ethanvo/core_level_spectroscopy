#!/usr/bin/env python3
from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.tools import lattice, pyscf_ase
from fileutils import dump
from pyscf.pbc.scf import chkfile
from pyscf.pbc.cc.eom_kccsd_rhf import _IMDS
import numpy as np
from functools import reduce
import os


class _ERIS:
    def __init__(self, cc):
        pass

def load_h5(h5file, key):
    data = {}
    for k, v in h5file[key].items():
        data[k] = v
    return data

def make_cell(material):
    cell = gto.Cell()
    ase_atom = lattice.get_ase_atom(material["formula"])
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell[:]
    cell.unit = "B"
    cell.basis = material["basis"]
    cell.exp_to_discard = material["exp_to_discard"]
    cell.verbose = 7
    cell.build()
    return cell

def make_meanfield(material, cell):
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
    return mymf

def load_mf(material):
    cell, scfdata = chkfile.load_scf("data/{}".format(material["chk"]))  # input
    mymf = scf.KRHF(cell, kpts=scfdata["kpts"], exxdiv="ewald")
    mymf.__dict__.update(scfdata)
    mymf = mymf.density_fit()
    mymf.with_df._cderi = "data/{}".format(material["cderi"])  # input
    mymf.chkfile = "data/{}".format(material["chk"])  # input
    mymf.converged = True
    return mymf

def make_no_coeff(material, mymf, h5file):
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
    return no_coeff

def make_cc(mymf, material, no_coeff):
    mycc = cc.KRCCSD(mymf, frozen=material["frozen"], mo_coeff=no_coeff)
    t1 = t2 = eris = None
    if os.path.isfile(f"data/{material['diis']}"):
        mycc.restore_from_diis(f"data/{material['diis']}")
    mycc.keep_exxdiv = True
    ekrccsd, t1, t2 = mycc.kernel(t1=t1, t2=t2, eris=eris)
    return mycc

def load_cc(mymf, material, h5file, no_coeff):
    mycc = cc.KRCCSD(mymf, frozen=material["frozen"], mo_coeff=no_coeff)
    mycc.keep_exxdiv = True
    t_amps_dict = load_h5(h5file, "t_amps")
    mycc.__dict__.update(t_amps_dict)
    mycc.converged = True
    return mycc

def load_eris():
    no_coeff = load_h5(h5file, "no_coeff")
    eris = _ERIS(mycc, no_coeff)
    eris_dict = load_h5(h5file, "eris")
    eris.__dict__.update(eris_dict)
    return eris

def load_imds():
    eris = load_eris(mycc, h5file)
    imds = _IMDS(mycc, eris)
    imds_dict = load_h5(h5file, "imds")
    imds.__dict__.update(imds_dict)
    imds.Wooov = imds.eris.ooov
    return imds

imds = _IMDS(mycc)
eris = imds.eris
imds._make_shared_1e()

ovoo_dest = grp_imds.create_dataset("Wovoo", (nkpts, nkpts, nkpts, nocc, nvir, nocc, nocc), dtype=t1.dtype.char)
ovoo = Wovoo(t1, t2, eris, kconserv, ovoo_dest)
oooo_dest = grp_imds.create_dataset("Woooo", (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), dtype=t1.dtype.char)
oooo = Woooo(t1, t2, eris, kconserv, oooo_dest)
ooov_dest = grp_imds.create_dataset("Wooov", (nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir), dtype=t1.dtype.char)
ooov = Wooov(t1, t2, eris, kconserv, ovoo_dest)
'''
CC initialization
mycc = cc.KRCCSD(mymf, frozen=material["frozen"], mo_coeff=no_coeff)
IMDS initialization
imds = _IMDS(mycc, eris=eris)

'''