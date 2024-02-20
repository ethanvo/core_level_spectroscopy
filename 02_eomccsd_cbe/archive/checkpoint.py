#!/usr/bin/env python3
from fileutils import load
import sys
import os
from checkpoint_functions import *
import h5py

material = load(f"data/{sys.argv[1]}")
h5file = h5py.File(f"data/{material['imds']}", "a")

# Meanfield
# Check if meanfield is converged
if not material["mf_converged"] and not os.path.exists(f"data/{material['chk']}"):
    cell = make_cell(material)
    mymf = make_meanfield(material, cell)
else:
    mymf = load_mf(material)

# MP2 Natural Orbitals
if not material["frozen"]:
    no_coeff = mymf.mo_coeff
else:
    if not material["Made NOs"]:
        no_coeff = make_no_coeff(material, mymf, h5file)
    else:
        no_coeff = h5file["no_coeff"][:] 

# Ground State CCSD
# Check if CCSD is converged
# Save diis, t1, t2, and eris

# Make 1e Intermediates
# Check if 1e intermediates are made `

# Make 2e Intermediates

## Wovoo
# Check if Wovoo is made

## Woooo
# Check if Woooo is made

## Wooov
# Check if Wooov is made

# EOM-CCSD
# Check if EOM-CCSD is converged
        
# CVS-EOM-CCSD
# Check if CVS-EOM-CCSD is converged

# Core Binding Energy


h5file.close()