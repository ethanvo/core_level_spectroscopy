#!/bin/bash
sbatch -J zno_ccpcvtz_wide /burg/berkelbach/users/eav2136/builds/work_tools/slurm/pyscf-job.sh hf_bandstructure.py zno_ccpcvtz_wide.json
sbatch -J gan_ccpcvtz_wide /burg/berkelbach/users/eav2136/builds/work_tools/slurm/pyscf-job.sh hf_bandstructure.py gan_ccpcvtz_wide.json
sbatch -J gaas_ccpcvtz_wide /burg/berkelbach/users/eav2136/builds/work_tools/slurm/pyscf-job.sh hf_bandstructure.py gaas_ccpcvtz_wide.json
sbatch -J mno_ccpcvtz_wide /burg/berkelbach/users/eav2136/builds/work_tools/slurm/pyscf-job.sh hf_bandstructure.py mno_ccpcvtz_wide.json
