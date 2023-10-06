#!/usr/bin/env python3
from fileutils import load, dump
from pyscf.pbc.scf import chkfile
from pyscf.pbc import scf, cc
import h5py
from pyscf.pbc.cc.eom_kccsd_rhf import _IMDS

class _ERIS:
    def __init__(self, cc):
        pass

def load_h5(h5file, key):
    data = {}
    for k, v in h5file[key].items():
        data[k] = v
    return data

def load_mf(inputfile):
    material = load("data/{}".format(inputfile))
    cell, scfdata = chkfile.load_scf("data/{}".format(material["chk"]))  # input
    mymf = scf.KRHF(cell, kpts=scfdata["kpts"], exxdiv="ewald")
    mymf.__dict__.update(scfdata)
    mymf = mymf.density_fit()
    mymf.with_df._cderi = "data/{}".format(material["cderi"])  # input
    mymf.chkfile = "data/{}".format(material["chk"])  # input
    mymf.converged = True
    return mymf

def load_mp(inputfile):
    material = load("data/{}".format(inputfile))
    h5file = h5py.File("data/{}".format(material["imds"]), "a")
    mymf = load_mf(inputfile)
    mycc = cc.KRCCSD(mymf, frozen=material["frozen"])
    mycc.keep_exxdiv = True
    t_amps_dict = load_h5(h5file, "t_amps")
    mycc.__dict__.update(t_amps_dict)
    mycc.converged = True
    return mycc, h5file

def load_eris(mycc, h5file):
    eris = _ERIS(mycc)
    eris_dict = load_h5(h5file, "eris")
    eris.__dict__.update(eris_dict)
    return eris

def load_imds(mycc, h5file):
    eris = load_eris(mycc, h5file)
    imds = _IMDS(mycc, eris)
    imds_dict = load_h5(h5file, "imds")
    imds.__dict__.update(imds_dict)
    imds.Wooov = imds.eris.ooov
    return imds
