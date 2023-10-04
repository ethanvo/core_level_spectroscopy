#!/bin/bash

#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --error="%x.err"
#SBATCH --output="%x.out"
#SBATCH --account=berkelbach
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eav2136@columbia.edu

export PYTHONPATH=/burg/berkelbach/users/eav2136/builds/work_tools/utilities:$PYTHONPATH
export MODULEPATH=/burg/berkelbach/users/eav2136/modulefiles:$MODULEPATH
cd ${SLURM_SUBMIT_DIR}
module load pyscf/projected-cvs
export PYSCF_MAX_MEMORY=$(bc <<< "$SLURM_MEM_PER_NODE / 1.024 / 1")
export PYSCF_TMPDIR=$(readlink -f tmp)
export OMP_NUM_THREADS=32

python cvspeomip.py $1
