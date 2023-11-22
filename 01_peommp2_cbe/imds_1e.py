#!/usr/bin/env python
from fileutils import load, dump
from support import load_mp
from pyscf.pbc.cc.eom_kccsd_rhf import _IMDS
import h5py
import sys

material = load("data/{}".format(sys.argv[1]))
mycc, h5file = load_mp(sys.argv[1])
imds = _IMDS(mycc)
eris = imds.eris
imds._make_shared_1e()
grp_eris = h5file.create_group("eris")
# Save mo_coeff, fock, oooo, ooov, oovv, ovov, voov, vovv
for key in ["mo_coeff", "fock"]:
    grp_eris.create_dataset(key, data=getattr(eris, key))
eris_keys = ["oooo", "ooov", "oovv", "ovov", "voov", "vovv"]
if getattr(eris, "feri1", None):
    for key in eris_keys:
        eris.feri1.copy(key, grp_eris)
else:
    for key in eris_keys:
        grp_eris.create_dataset(key, data=getattr(eris, key))
grp_imds = h5file.create_group("imds")
for k, v in imds.__dict__.items():
    if (k[0] == "F" or k[0] == "L") and v is not None:
        print("IMDS key to save:", k, ", type:", type(v))
        grp_imds.create_dataset(k, data=v)
h5file.close()
print("Made 1e intermediates")
