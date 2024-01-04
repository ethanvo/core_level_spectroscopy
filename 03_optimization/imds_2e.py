#!/usr/bin/env python
from fileutils import load
import sys
from pyscf.pbc.cc.kintermediates_rhf import Wovoo
from support import load_mp, load_eris
from pyscf.lib import logger

material = load("data/{}".format(sys.argv[1]))
mycc, h5file = load_mp(sys.argv[1])
t1, t2 = mycc.t1, mycc.t2
eris = load_eris(mycc, h5file)
kconserv = mycc.khelper.kconserv
nkpts, nocc, nvir = t1.shape
grp_imds = h5file["imds"]
if "Wovoo" in grp_imds:
    del grp_imds["Wovoo"]
ovoo_dest = grp_imds.create_dataset("Wovoo", (nkpts, nkpts, nkpts, nocc, nvir, nocc, nocc), dtype=t1.dtype.char)
cpu0 = (logger.process_clock(), logger.perf_counter())
ovoo = Wovoo(t1, t2, eris, kconserv, ovoo_dest)
logger.timer(mycc, 'Wovoo', *cpu0)
h5file.close()
print("Made 2e intermediates")
