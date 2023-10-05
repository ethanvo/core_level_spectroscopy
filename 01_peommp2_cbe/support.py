#!/usr/bin/env python3
from fileutils import load, dump
from pyscf.pbc.scf import chkfile
from pyscf.pbc import scf, cc
import h5py
from pyscf.pbc.cc.eom_kccsd_rhf import _IMDS

class _ERIS:
    def __init__(self, cc):
        pass

def load_h5(inputfile, key):
    f = h5py.File(inputfile, "r")
    data = {}
    for k, v in f[key].items():
        data[k] = v[:]
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
    mymf = load_mf(inputfile)
    mycc = cc.KRCCSD(mymf, frozen=material["frozen"])
    mycc.keep_exxdiv = True
    t_amps_dict = load_h5("data/{}".format(material["imds"]), "t_amps")
    mycc.__dict__.update(t_amps_dict)
    mycc.converged = True
    return mycc

def load_eris(inputfile):
    material = load("data/{}".format(inputfile))
    mycc = load_mp(inputfile)
    eris = _ERIS(mycc)
    eris_dict = load_h5("data/{}".format(material["imds"]), "eris")
    eris.__dict__.update(eris_dict)
    return eris

def load_imds(inputfile):
    material = load("data/{}".format(inputfile))
    mycc = load_mp(inputfile)
    eris = _ERIS(mycc)
    eris_dict = load_h5("data/{}".format(material["imds"]), "eris")
    eris.__dict__.update(eris_dict)
    imds = _IMDS(mycc, eris)
    imds_dict = load_h5("data/{}".format(material["imds"]), "imds")
    imds.__dict__.update(imds_dict)
    imds.Wooov = imds.eris.ooov
    return imds
