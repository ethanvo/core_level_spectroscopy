#!/usr/bin/env python
import json
from pyscf.pbc import gto, scf
from pyscf.pbc.tools import lattice, pyscf_ase
from pyscf.pbc.scf import chkfile
from fileutils import load, dump
import sys

material = load("data/{}".format(sys.argv[1]))
# Create Cell
cell = gto.Cell()
ase_atom = lattice.get_ase_atom(material["formula"])
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
cell.a = ase_atom.cell[:]
cell.unit = "B"
cell.basis = material["basis"]
cell.exp_to_discard = material["exp_to_discard"]
cell.verbose = 7
cell.build()

# Mean Field
kdensity = material["kdensity"]
kmesh = [kdensity, kdensity, kdensity]
kpts = cell.make_kpts(kmesh, scaled_center=material["vb_scaled_center"])
mymf = scf.KRHF(cell, kpts=kpts, exxdiv="ewald")
mymf.chkfile = "data/{}".format(material["chk"])
mymf = mymf.density_fit()
mymf.with_df._cderi_to_save = "data/{}".format(material["cderi"])
ekrhf = mymf.kernel()
convergence = mymf.converged
mymf_data = dict([("ekrhf", float(ekrhf)),
                  ("convergence", bool(convergence))])
dump(mymf_data, "data/{}".format(material["ekrhf"]))
