#!/bin/bash

#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --error="%x.err"
#SBATCH --output="%x.output"
#SBATCH --account=ccce
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eav2136@columbia.edu

export PYTHONPATH=/burg/berkelbach/users/eav2136/builds/work_tools/utilities:$PYTHONPATH
export MODULEPATH=/burg/berkelbach/users/eav2136/modulefiles:$MODULEPATH
cd ${SLURM_SUBMIT_DIR}
module load pyscf/projected-cvs

python core.py $1
