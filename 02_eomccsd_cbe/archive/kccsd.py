#!/usr/bin/env python
from pyscf.pbc import cc
from fileutils import load, dump
from support import load_mf
import sys
import h5py

material = load("data/{}".format(sys.argv[1]))
mymf = load_mf(sys.argv[1])
with h5py.File("data/{}".format(material["imds"]), "r") as fin:
    no_coeff = fin["no_coeff"][:]
mycc = cc.KRCCSD(mymf, frozen=material["frozen"], mo_coeff=no_coeff)
mycc.keep_exxdiv = True
ekrccsd, t1, t2 = mycc.kernel()
# Check t1, t2 type
print("t1 type: {}".format(type(t1)))
print("t2 type: {}".format(type(t2)))
mycc_data = dict([("ekrmp2", float(ekrccsd)),
                  ("converged", bool(mycc.converged))])

dump(mycc_data, "data/{}".format(material["ekrmp2"]))
# Save t1, t2
with h5py.File("data/{}".format(material["imds"]), "w") as fout:
    grp = fout.create_group("t_amps")
    grp.create_dataset("t1", data=t1)
    grp.create_dataset("t2", data=t2)
