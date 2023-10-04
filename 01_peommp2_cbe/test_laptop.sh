#!/bin/bash
export PYTHONPATH=/Users/ethanvo/builds/pyscf/projected_cvs/pyscf:$PYTHONPATH
export OMP_NUM_THREADS=10
export PYSCF_MAX_MEMORY=64000
cd /Users/ethanvo/projects/core_level_spectrscopy/01_peommp2_cbe
export PYSCF_TMPDIR=$(readlink -f tmp)
python3 meanfield.py c_ccpcvtz_k2_C_1s.json
python3 krmp2.py c_ccpcvtz_k2_C_1s.json
python3 imds_1e.py c_ccpcvtz_k2_C_1s.json
python3 imds_2e.py c_ccpcvtz_k2_C_1s.json
python3 peomip.py c_ccpcvtz_k2_C_1s.json
python3 cvspeomip.py c_ccpcvtz_k2_C_1s.json
python3 core.py c_ccpcvtz_k2_C_1s.json
