#!/usr/bin/env python3
from fileutils import load, dump
import sys
import h5py
from pyscf.pbc.scf import chkfile
from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.tools import lattice, pyscf_ase
import numpy as np
from functools import reduce

class _ERIS:
    def __init__(self, cc):
        pass


def load_h5(h5file, key):
    data = {}
    for k, v in h5file[key].items():
        data[k] = v
    return data


def save_data(material, h5file, key, data):
    if key in h5file and not material[f"made_{key}"]:
        del h5file[key]
    h5file.create_dataset(key, data=data)
    material[f"made_{key}"] = True
    material[f"{key}_attempt"] += 1
    dump(material, f"data/{sys.argv[1]}")


def check_data(material, h5file, key):
    # Throw error if data has been made more than once
    if not material[f"made_{key}"] and material[f"{key}_attempt"] > 0:
        raise ValueError(f"{key} has been made more than once.")
    # Check if data is in h5file
    

def load_mf(material):
    cell, scfdata = chkfile.load_scf(f"data/{material['chk']}")
    mymf = scf.KRHF(cell, kpts=scfdata["kpts"], exxdiv="ewald")
    mymf.__dict__.update(scfdata)
    mymf = mymf.density_fit()
    mymf.with_df._cderi = f"data/{material['cderi']}"
    mymf.chkfile = f"data/{material['chk']}"
    mymf.converged = True
    return mymf


def build_cell(material):
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


def run_mf(material, mymf):
    ekrhf = mymf.kernel()
    material["ekrhf"] = float(ekrhf)
    material["mf_convergence"] = mymf.converged
    dump(material, f"data/{sys.argv[1]}")


def run_mf(material, cell):
    kmesh = material["kmesh"]
    kpts = cell.make_kpts(kmesh, scaled_center=material["vb_scaled_center"])
    mymf = scf.KRHF(cell, kpts=kpts, exxdiv="ewald")
    mymf.chkfile = f"data/{material['chk']}"
    mymf = mymf.density_fit()
    mymf.with_df._cderi_to_save = f"data/{material['cderi']}"
    return mymf


def make_no_coeff(material, mymf, imdsfile):
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
        fvv_no = reduce(np.dot, (
            v.T.conj(),
            fvv,
            v
        ))
        _, v_canon = np.linalg.eigh(fvv_no[:nvir_act, :nvir_act])
        no_coeff_1 = reduce(np.dot, (
            mo_coeff[k][:, nocc:],
            v[:, :nvir_act],
            v_canon
        ))
        no_coeff_2 = np.dot(mo_coeff[k][:, nocc:], v[:, nvir_act:])
        no_coeff_k = np.concatenate((
            mo_coeff[k][:, :nocc],
            no_coeff_1,
            no_coeff_2
        ), axis=1)
        no_coeff.append(no_coeff_k)
    no_coeff = np.asarray(no_coeff)
    save_data(material, imdsfile, "no_coeff", no_coeff)


def run_cc(material, mymf, no_coeff, imdsfile):
    mycc = cc.KRCCSD(mymf, frozen=material["frozen"], mo_coeff=no_coeff)
    if material["made_eris"]:
        eris = _ERIS(mycc)
        eris.__dict__.update(load_h5(imdsfile, "eris"))
    else:
        eris = mycc.ao2mo(mo_coeff=no_coeff)
        grp_eris = imdsfile.create_group("eris")
        for key in ["mo_coeff", "fock"]:
            grp_eris.create_dataset(key, data=getattr(eris, key))
        eris_keys = ["oooo", "ooov", "oovv", "ovov", "voov", "vovv"]
        if getattr(eris, "feri1", None):
            for key in eris_keys:
                eris.feri1.copy(key, grp_eris)
        else:
            for key in eris_keys:
                grp_eris.create_dataset(key, data=getattr(eris, key))
    if os.path.isfile(f"data/{material['diis']}"):
        mycc.restore_from_diis(f"data/{material['diis']}")
    mycc.keep_exxdiv = True
    ekrccsd, t1, t2 = mycc.kernel(t1=t1, t2=t2, eris=eris)
    save_data(material, imdsfile, "t_amps", {"t1": t1, "t2": t2})

def run_eomcc():
myeom = EOMIP(mycc)
eip, vip = myeom.ipccsd(nroots=material["vb_nroots"], imds=imds, kptlist=[0])
convergence = np.real(myeom.converged) != 0
myeom_data = dict([
    ("eip", eip.tolist()),
    ("convergence", convergence.tolist())
])

def run_cvs_eomcc():
myeom = CVSEOMIP(mycc)
myeom.mandatory = material["core"]
eip, vip = myeom.ipccsd(nroots=len(material["core"]), imds=imds, kptlist=[0])
convergence = np.real(myeom.converged) != 0
myeom_data = dict([
    ("eip", eip.tolist()),
    ("convergence", convergence.tolist())
])

def run_cbe():


au2ev = 27.211386245988

material = load(f"data/{sys.argv[1]}")

imdsfile = h5py.File(f"data/{material['imds']}", "a")

# Check fo chkfile and mf_convergence
# Cell and mean field

if material["mf_convergence"]:

    

# natural orbitals
  
if material['made_no_coeff']:

else:



imdsfile.close()
