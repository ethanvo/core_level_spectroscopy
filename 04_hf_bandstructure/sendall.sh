#!/bin/bash
sbatch -J zno_ccpcvtz_222_wide /burg/berkelbach/users/eav2136/builds/work_tools/slurm/pyscf/pyscf-job.sh hf_bandstructure.py zno_ccpcvtz_222_wide.json
sbatch -J gan_ccpcvtz_222_wide /burg/berkelbach/users/eav2136/builds/work_tools/slurm/pyscf/pyscf-job.sh hf_bandstructure.py gan_ccpcvtz_222_wide.json
sbatch -J gaas_ccpcvtz_222_wide /burg/berkelbach/users/eav2136/builds/work_tools/slurm/pyscf/pyscf-job.sh hf_bandstructure.py gaas_ccpcvtz_222_wide.json
sbatch -J mno_ccpcvtz_222_wide /burg/berkelbach/users/eav2136/builds/work_tools/slurm/pyscf/pyscf-job.sh hf_bandstructure.py mno_ccpcvtz_222_wide.json
