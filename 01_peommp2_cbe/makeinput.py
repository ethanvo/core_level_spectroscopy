#!/usr/bin/env python3
from pyscf.pbc import gto
from pyscf.pbc.tools import lattice, pyscf_ase
from fileutils import dump
from scipy.optimize import root_scalar

# Components of input file
# "formula"
# "basis"
# "kdensity"
# "vb_scaled_center"
# "vb_nroots"
# "exp_to_discard"
# "core"
# "nmo
# "nocc"
# "nvir"
# "frozen"
# "chk"
# "cderi"
# "ekrhf" : store ekrhf, convergence
# "ekrmp2" : store ekrmp2, convergence
# "imds"
# "eomip_core" : store eomip_core, convergence
# "eomip_valence" : store eomip_valence, convergence
# "cbe"
# "vip"

# Check within storage limits
# if not freeze

au2ev = 27.211386245988

materials = ["bn", "aln", "alp", "mgo"]
#materials = ["bn", "aln", "alp", "mgo", "zno", "gan", "gaas", "lif", "licl", "mno"]

basis_sets = ["ccpcvtz"]

vb_scaled_centers = {
        "c": [0.0, 0.0, 0.0],
        'si': [0.0, 0.0, 0.0],
        "sic": [0.0, 0.0, 0.0],
        "bn": [0.0, 0.0, 0.0],
        "aln": [0.0, 0.0, 0.0],
        "alp": [0.0, 0.0, 0.0],
        "mgo": [0.0, 0.0, 0.0],
        "zno": [0.0, 0.0, 0.0],
        "gan": [0.010101010101010102, 0.0, 0.010101010101010102], # Check
        "gaas": [0.0, 0.0, 0.0],
        "lif": [0.0, 0.0, 0.0],
        "licl": [0.0530303, 0.0530303, 0.10606061],
        "mno" : [0.0, 0.0, 0.0],
        }

vb_nroots = {
        "c": 3,
        "si": 3,
        "sic": 3,
        "bn" : 3,
        "aln" : 1,
        "alp" : 3,
        "mgo" : 3,
        "zno" : 1, # Check
        "gan" : 3, # Check
        "gaas" : 3, # Check
        "lif" : 3,
        "licl" : 1,
        "mno" : 3, # Check
        }

core_orbitals = {
        "c": {"C_1s": [0, 1]},
        "si": {"Si_2p": [0, 1, 2, 3, 4, 5]},
        "sic": {"C_1s": [0]},
        "bn" : {"N_1s": [0], "B_1s": [1]},
        "aln" : {"N_1s": [0, 1], "Al_2p": [4, 5, 6, 7, 8, 9]},
        "alp" : {"Al_2p": [0, 1, 2]},
        "mgo" : {"O_1s": [0], "Mg_2p": [2, 3, 4]},
        "zno" : {},
        "gan" : {},
        "gaas" : {},
        "lif" : {},
        "licl" : {},
        "mno" : {},
        }


exp_to_discard = {
        "c": None,
        "si": None,
        "sic": None,
        "bn" : None,
        "aln" : None,
        "alp" : None,
        "mgo" : 0.033,
        "zno" : None,
        "gan" : None,
        "gaas" : None,
        "lif" : None,
        "licl" : None,
        "mno" : None,
        }

frozen_core = {
        "c": [],
        "si": [0, 1, 2, 3],
        "sic": [0],
        "bn" : [],
        "aln" : [0, 1],
        "alp" : [0, 1, 2, 3, 4, 5, 6],
        "mgo" : [0],
        "zno" : [],
        "gan" : [],
        "gaas" : [],
        "mno" : [],
}

def get_nmo(formula, basis):
    cell = gto.Cell()
    ase_atom = lattice.get_ase_atom(formula)
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell[:]
    cell.unit = "B"
    cell.basis = basis
    cell.exp_to_discard = exp_to_discard[formula]
    cell.verbose = 0
    cell.build()
    nmo = cell.nao_nr()
    nocc = cell.nelectron // 2
    nvir = nmo - nocc
    return int(nocc), int(nvir), int(nmo)

def peommp2_ip_storage(nkpts, nocc, nvir):
    total = 0
    nmo = nocc + nvir
    # SCF
    total += nkpts * nmo * nmo # mo_coeff
    total += nkpts * nmo # mo_energy
    total += nkpts * nkpts * nkpts * nmo * nmo * nmo # CDERI
    # CCSD
    total += nkpts * nocc * nvir # t1
    total += nkpts * nkpts * nkpts * nocc * nocc * nvir * nvir # t2
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nocc # eris.oooo
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nvir # eris.ooov
    total += nkpts * nkpts * nkpts * nocc * nocc * nvir * nvir # eris.oovv
    total += nkpts * nkpts * nkpts * nocc * nvir * nocc * nvir # eris.ovov
    total += nkpts * nkpts * nkpts * nvir * nocc * nocc * nvir # eris.voov
    total += nkpts * nkpts * nkpts * nvir * nocc * nvir * nvir # eris.vovv
    # EOM
    total += nkpts * nocc * nocc # Loo
    total += nkpts * nvir * nvir # Lvv
    total += nkpts * nocc * nvir # Fov
    total += nkpts * nkpts * nkpts * nocc * nvir * nocc * nocc # Wovoo
    total += nkpts * nmo * nmo # fock
    total += nkpts * nkpts # Lpv
    # return total in bytes in terabytes
    return total * 16 / 1000**4

def get_keydict(formula, basis, kdensity, orbital, core, nmo, nocc, nvir, frozen, res, fout):
    key = "{}_{}_k{}_{}".format(formula, basis, kdensity, orbital)
    if frozen is not None:
        key += "_nvir_act{}".format(int(res.root))
    material = dict([
        ("formula", formula),
        ("basis", basis),
        ("kdensity", kdensity),
        ("vb_scaled_center", vb_scaled_centers[formula]),
        ("vb_nroots", vb_nroots[formula]),
        ("exp_to_discard", exp_to_discard[formula]),
        ("core", core),
        ("nmo", nmo),
        ("nocc", nocc),
        ("nvir", nvir),
        ("frozen_core", frozen_core[formula]),
        ("frozen", frozen),
        ("chk", "{}.chk".format(key)),
        ("cderi", "{}_cderi.h5".format(key)),
        ("ekrhf", "{}_ekrhf.json".format(key)),
        ("ekrmp2", "{}_ekrmp2.json".format(key)),
        ("imds", "{}_imds.h5".format(key)),
        ("eomip_valence", "{}_eomip_valence.json".format(key)),
        ("eomip_core", "{}_eomip_core.json".format(key)),
        ("cbe", "{}_cbe.json".format(key)),
        ("vip", "{}_vip.h5".format(key)),
        ])
    filename = "data/{}.json".format(key)
    fout.write("./submit.sh {} 716800 5-00:00:00\n".format(key))
    dump(material, filename)
    return key, material, filename


with open("joblist.txt", "w") as fout:
    for formula in materials:
        for basis in basis_sets:
            for orbital, core in core_orbitals[formula].items():
                for kmax in range(4, 1, -1):
                    nocc, nvir, nmo = get_nmo(formula, basis)
                    nkpts = kmax**3
                    # Check if storage is within limits 16 TB
                    frozen = None
                    if peommp2_ip_storage(nkpts, nocc - len(frozen_core[formula]), nvir) > 1.5:
                        def f(x):
                            return peommp2_ip_storage(nkpts, nocc, x) - 1.5
                        res = root_scalar(f, x0=float(nvir), x1=float(nocc))
                        frozen = frozen_core[formula] + list(range(nocc + int(res.root), nmo))
                    if frozen is not None:
                        for kdensity in range(kmax, 1, -1):
                            key, material, filename = get_keydict(formula, basis, kdensity, orbital, core, nmo, nocc, nvir, frozen, res, fout)
                    else:
                        kdensity = kmax
                        key, material, filename = get_keydict(formula, basis, kdensity, orbital, core, nmo, nocc, nvir, frozen, res, fout)
