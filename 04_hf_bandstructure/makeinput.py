#!/usr/bin/env python3
from fileutils import dump

materials = ["zno", "gan", "gaas", "mno"]
basis = "ccpcvtz"
paths = {"zno" : "GMKGALHA",
         "gan" : "LGXWKG",
         "gaas" : "LGXWKG",
         "mno" : "LGXWKG"}

fout = open("sendall.sh", "w")
fout.write("#!/bin/bash\n")

for formula in materials:
    key = "{}_{}_222_wide".format(formula, basis)
    data = {}
    data["formula"] = formula
    data["basis"] = basis
    data["path"] = paths[formula]
    data["npoints"] = 100
    data["kmesh"] = [2, 2, 2]
    data["e_kn_file"] = "data/{}_e_kn.h5".format(key)
    data["output_file"] = "data/{}_data.json".format(key)
    dump(data, "data/{}.json".format(key))
    fout.write("sbatch -J {} /burg/berkelbach/users/eav2136/builds/work_tools/slurm/pyscf/pyscf-job.sh hf_bandstructure.py {}.json\n".format(key, key))

fout.close()
