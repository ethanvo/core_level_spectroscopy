#!/usr/bin/env python
from pyscf.pbc import gto, scf
from pyscf.pbc.tools import lattice, pyscf_ase
from fileutils import load, dump
import numpy as np
import pandas as pd

vb_scaled_centers = {
    "c": [0.0, 0.0, 0.0],
    "si": [0.0, 0.0, 0.0],
    "sic": [0.0, 0.0, 0.0],
    "bn": [0.0, 0.0, 0.0],
    "aln": [0.0, 0.0, 0.0],
    "alp": [0.0, 0.0, 0.0],
    "mgo": [0.0, 0.0, 0.0],
    "mgs": [0.0, 0.0, 0.0],
    "zno": [0.0, 0.0, 0.0],
    "gan": [0.010101010101010102, 0.0, 0.010101010101010102],
    "gaas": [0.0, 0.0, 0.0],
}

basis = "ccpcvtz"

filenames = []

materials = ["c", "si", "sic", "bn", "aln", "alp", "mgo", "mgs", "zno", "gan", "gaas"]

with open("submit_jobs.sh", "w") as f:
    f.write("#!/bin/bash\n")
    for formula in materials:
        # Create cell
        cell = gto.Cell()
        ase_atom = lattice.get_ase_atom(formula)
        cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
        cell.a = ase_atom.cell[:]
        cell.unit = "B"
        cell.basis = basis
        cell.verbose = 0
        cell.build()

        # Print cell exponents
        nbas = cell.nbas
        print("Cell exponents")
        exps = []
        exps = np.array(exps)
        for i in range(nbas):
            exps = np.concatenate((exps, cell.bas_exp(i)))
        exps = exps.flatten()
        exps = np.unique(exps)
        exps = np.sort(exps)
        print(exps)

        kdensity = 4
        for i in range(0, 6):
            if i == 0:
                exp = None
            else:
                exp = exps[i]
            key = "{}_{}_exp{}".format(formula, basis, i)
            inputfile = "{}.json".format(key)
            datafile = "{}_result.json".format(key)
            material = dict(
                [
                    ("formula", formula),
                    ("basis", basis),
                    ("exp_to_discard", exp),
                    ("kdensity", kdensity),
                    ("vb_scaled_center", vb_scaled_centers[formula]),
                    ("datafile", datafile),
                ]
            )
            dump(material, "data/{}".format(inputfile))
            f.write("sbatch -J {} expdiscard.sh {}\n".format(key, inputfile))
