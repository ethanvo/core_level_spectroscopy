#!/usr/bin/env python3
from bandstructure import make_cell, get_all_electron_bandstructure
from pyscf.pbc.tools import lattice
from ase.dft.kpoints import get_bandpath

cell = make_cell('c', 'ccecpccpvdz', pseudo='ccecp')
ase_atom = lattice.get_ase_atom('c')
bandpath = get_bandpath('GW', ase_atom.cell, npoints=100)
band_kpts_scaled = bandpath.kpts
print(band_kpts_scaled[0])
print(type(band_kpts_scaled[0]))