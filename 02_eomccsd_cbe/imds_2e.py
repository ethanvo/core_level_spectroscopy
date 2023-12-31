#!/usr/bin/env python
from fileutils import load, dump
import sys
from pyscf.pbc.scf import chkfile
from pyscf.pbc.cc.kintermediates_rhf import Wovoo
from support import load_mp, load_eris
import h5py

material = load("data/{}".format(sys.argv[1]))
mycc, h5file = load_mp(sys.argv[1])
t1, t2 = mycc.t1, mycc.t2
eris = load_eris(mycc, h5file)
kconserv = mycc.khelper.kconserv
nkpts, nocc, nvir = t1.shape
grp_imds = h5file["imds"]
ovoo_dest = grp_imds.create_dataset("Wovoo", (nkpts, nkpts, nkpts, nocc, nvir, nocc, nocc), dtype=t1.dtype.char)
ovoo = Wovoo(t1, t2, eris, kconserv, ovoo_dest)
h5file.close()
print("Made 2e intermediates")
