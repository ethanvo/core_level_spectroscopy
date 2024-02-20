#!/usr/bin/env python3
from fileutils import load, dump, remove
import sys
import os
import numpy as np
import h5py
from functools import reduce
from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.tools import lattice, pyscf_ase
from pyscf.pbc.scf import chkfile
from pyscf.pbc.cc.eom_kccsd_rhf import _IMDS, EOMIP, CVSEOMIP
from pyscf.pbc.cc.kintermediates_rhf import Wovov, Wovvo, Wovoo, Woooo, Wooov


class _ERIS:
    def __init__(self, cc):
        pass


def load_h5(h5file, key):
    data = {}
    for k, v in h5file[key].items():
        data[k] = v
    return data


au2ev = 27.211386245988

material = load(f"data/{sys.argv[1]}")
if not os.path.isfile(f"data/{material['cbe']}"):
    imdsfile = h5py.File(f"data/{material['imds']}", "a")

    if not os.path.isfile(f"data/{material['chk']}"):
        cell = gto.Cell()
        ase_atom = lattice.get_ase_atom(material["formula"])
        cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
        cell.a = ase_atom.cell[:]
        cell.unit = "B"
        cell.basis = material["basis"]
        cell.exp_to_discard = material["exp_to_discard"]
        cell.verbose = 7
        cell.build()

        kpts = cell.make_kpts(material["kmesh"],
                              scaled_center=material["vb_scaled_center"])
        mymf = scf.KRHF(cell, kpts=kpts, exxdiv="ewald")
        mymf.chkfile = f"data/{material['chk']}"
        mymf = mymf.density_fit()
        mymf.with_df._cderi_to_save = f"data/{material['cderi']}"

    else:
        cell, scfdata = chkfile.load_scf(f"data/{material['chk']}")
        mymf = scf.KRHF(cell, kpts=scfdata["kpts"], exxdiv="ewald")
        mymf.__dict__.update(scfdata)
        mymf = mymf.density_fit()
        mymf.with_df._cderi = f"data/{material['cderi']}"
        mymf.chkfile = f"data/{material['chk']}"

    if not os.path.isfile(f"data/{material['ekrhf']}"):
        ekrhf = mymf.kernel()
        convergence = mymf.converged
        mymf_data = {"ekrhf": float(ekrhf), "convergence": bool(convergence)}
        dump(mymf_data, f"data/{material['ekrhf']}")
    else:
        mymf.converged = True

    if not material["frozen"]:
        no_coeff = mymf.mo_coeff
    else:
        if not material['made_no_coeff']:
            nvir_act = material["nvir_act"]
            mo_coeff = mymf.mo_coeff
            mo_energy = mymf.mo_energy
            mypt = mp.KMP2(mymf)
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
            if "no_coeff" in imdsfile:
                del imdsfile["no_coeff"]
            imdsfile.create_dataset("no_coeff", data=no_coeff)
            material["made_no_coeff"] = True
            dump(material, f"data/{sys.argv[1]}")
        else:
            no_coeff = imdsfile["no_coeff"][:]

    mycc = cc.KRCCSD(mymf, frozen=material['frozen'], mo_coeff=no_coeff)
    mycc.keep_exxdiv = True

    if not material['made_eris']:
        eris = mycc.ao2mo(mo_coeff=no_coeff)
        if "eris" in imdsfile:
            del imdsfile["eris"]
        grp_eris = imdsfile.create_group("eris")
        for key in ["mo_coeff", "mo_energy", "fock"]:
            grp_eris.create_dataset(key, data=getattr(eris, key))
        eris_keys = ["oooo", "ooov", "oovv", "ovov", "voov", "vovv"]
        if getattr(eris, "feri1", None):
            for key in eris_keys:
                eris.feri1.copy(key, grp_eris)
        else:
            for key in eris_keys:
                grp_eris.create_dataset(key, data=getattr(eris, key))
        material["made_eris"] = True
        dump(material, f"data/{sys.argv[1]}")
    else:
        eris = _ERIS(mycc)
        eris.__dict__.update(load_h5(imdsfile, "eris"))

    if not os.path.isfile(f"data/{material['ekrccsd']}"):
        t1 = t2 = None
        if os.path.isfile(f"data/{material['diis']}"):
            mycc.restore_from_diis_(f"data/{material['diis']}")
            t1, t2 = mycc.t1, mycc.t2
        mycc.diis_file = f"data/{material['diis']}"
        ekrccsd, t1, t2 = mycc.kernel(t1=t1, t2=t2, eris=eris)
        mycc_data = {"ekrccsd": float(ekrccsd),
                     "converged": bool(mycc.converged)}
        dump(mycc_data, f"data/{material['ekrccsd']}")
        if "t_amps" in imdsfile:
            del imdsfile["t_amps"]
        grp = imdsfile.create_group("t_amps")
        grp.create_dataset("t1", data=t1)
        grp.create_dataset("t2", data=t2)
    else:
        t_amps_dict = load_h5(imdsfile, "t_amps")
        mycc.__dict__.update(t_amps_dict)
        mycc.converged = True

    imds = _IMDS(mycc, eris=eris)
    t1, t2 = mycc.t1, mycc.t2
    kconserv = mycc.khelper.kconserv
    nkpts, nocc, nvir = t1.shape

    if "imds" in imdsfile:
        grp_imds = imdsfile["imds"]
    else:
        grp_imds = imdsfile.create_group("imds")

    if not material["made_imds"]:
        if not material['made_imds_1e']:
            imds._make_shared_1e()
            for k, v in imds.__dict__.items():
                if (k[0] == "F" or k[0] == "L") and v is not None:
                    print("IMDS key to save:", k, ", type:", type(v))
                    if k in grp_imds:
                        del grp_imds[k]
                    grp_imds.create_dataset(k, data=v)
            material["made_imds_1e"] = True
            dump(material, f"data/{sys.argv[1]}")

        if not material['made_Wovov']:
            if "Wovov" in grp_imds:
                del grp_imds["Wovov"]
            ovov_dest = grp_imds.create_dataset(
                "Wovov", (nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir),
                dtype=t1.dtype.char
            )
            print("Making Wovov")
            Wovov(t1, t2, eris, kconserv, ovov_dest)
            material["made_Wovov"] = True
            dump(material, f"data/{sys.argv[1]}")
            print("Made Wovov")

        if not material['made_Wovvo']:
            if "Wovvo" in grp_imds:
                del grp_imds["Wovvo"]
            ovvo_dest = grp_imds.create_dataset(
                "Wovvo", (nkpts, nkpts, nkpts, nocc, nvir, nvir, nocc),
                dtype=t1.dtype.char
            )
            print("Making Wovvo")
            Wovvo(t1, t2, eris, kconserv, ovvo_dest)
            material["made_Wovvo"] = True
            dump(material, f"data/{sys.argv[1]}")
            print("Made Wovvo")

        if not material['made_Wovoo']:
            if "Wovoo" in grp_imds:
                del grp_imds["Wovoo"]
            ovoo_dest = grp_imds.create_dataset(
                "Wovoo", (nkpts, nkpts, nkpts, nocc, nvir, nocc, nocc),
                dtype=t1.dtype.char
            )
            print("Making Wovoo")
            Wovoo(t1, t2, eris, kconserv, ovoo_dest)
            material["made_Wovoo"] = True
            dump(material, f"data/{sys.argv[1]}")
            print("Made Wovoo")

        if not material['made_Woooo']:
            if "Woooo" in grp_imds:
                del grp_imds["Woooo"]
            oooo_dest = grp_imds.create_dataset(
                "Woooo", (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc),
                dtype=t1.dtype.char
            )
            print("Making Woooo")
            Woooo(t1, t2, eris, kconserv, oooo_dest)
            material["made_Woooo"] = True
            dump(material, f"data/{sys.argv[1]}")
            print("Made Woooo")

        if not material['made_Wooov']:
            if "Wooov" in grp_imds:
                del grp_imds["Wooov"]
            ooov_dest = grp_imds.create_dataset(
                "Wooov", (nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir),
                dtype=t1.dtype.char
            )
            print("Making Wooov")
            Wooov(t1, t2, eris, kconserv, ooov_dest)
            material["made_Wooov"] = True
            dump(material, f"data/{sys.argv[1]}")
            print("Made Wooov")

        material["made_imds"] = True
        dump(material, f"data/{sys.argv[1]}")

    imds.__dict__.update(load_h5(imdsfile, "imds"))

    imds.Woovv = eris.oovv

    if not os.path.isfile(f"data/{material['eomip_valence']}"):
        myeom = EOMIP(mycc)
        eip, vip = myeom.ipccsd(nroots=material["vb_nroots"],
                                imds=imds,
                                kptlist=[0])
        convergence = np.real(myeom.converged) != 0
        myeom_data = {"eip": eip.tolist(),
                      "convergence": convergence.tolist()}
        dump(myeom_data, f"data/{material['eomip_valence']}")
        with h5py.File(f"data/{material['vip']}", "w") as f:
            f.create_dataset("vip", data=vip)
        ip = eip
    else:
        ip = np.array(load(f"data/{material['eomip_valence']}")["eip"])

    if not os.path.isfile(f"data/{material['eomip_core']}"):
        mycvseom = CVSEOMIP(mycc)
        mycvseom.mandatory = material["core"]
        cvseip, cvsvip = mycvseom.ipccsd(nroots=len(material["core"]),
                                         imds=imds,
                                         kptlist=[0])
        cvsconvergence = np.real(mycvseom.converged) != 0
        mycvseom_data = {"eip": cvseip.tolist(),
                         "convergence": cvsconvergence.tolist()}
        dump(mycvseom_data, f"data/{material['eomip_core']}")
        with h5py.File(f"data/{material['vip']}", "a") as f:
            f.create_dataset("vip_core", data=vip)
        core = cvseip
    else:
        core = np.array(load(f"data/{material['eomip_core']}")["eip"])

    cbe = (np.amin(core) - np.amax(ip)) * au2ev
    core_data = {"cbe": float(cbe)}
    dump(core_data, f"data/{material['cbe']}")

    imdsfile.close()
    remove(f"data/{material['chk']}")
    remove(f"data/{material['cderi']}")
    remove(f"data/{material['diis']}")
    remove(f"data/{material['imds']}")
else:
    remove(f"data/{material['chk']}")
    remove(f"data/{material['cderi']}")
    remove(f"data/{material['diis']}")
    remove(f"data/{material['imds']}")
