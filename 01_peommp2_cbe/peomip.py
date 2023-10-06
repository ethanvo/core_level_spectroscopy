#!/usr/bin/env python3
from fileutils import load, dump
from pyscf.pbc.cc.eom_kccsd_rhf import EOMIP
from support import load_mp, load_imds
import sys
import h5py
import numpy as np

material = load("data/{}".format(sys.argv[1]))
mycc, h5file = load_mp(sys.argv[1])
imds = load_imds(mycc, h5file)
myeom = EOMIP(mycc)
eip, vip = myeom.ipccsd(nroots=material["vb_nroots"], imds=imds, partition="mp", kptlist=[0])
convergence = np.real(myeom.converged) != 0
myeom_data = dict([("eip", eip.tolist()), 
                   ("convergence", convergence.tolist())])
dump(myeom_data, "data/{}".format(material["eomip_valence"]))
h5file.close()
with h5py.File("data/{}".format(material["vip"]), "w") as f:
    f.create_dataset("vip", data=vip)
