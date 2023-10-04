#!/bin/bash
# meanfield krmp2 imds_1e imds_2e peomip cvspeomip core
J1=$(sbatch --parsable -J ${1}_meanfield --mem=${2}MB --time=${3} meanfield.sh ${1}.json)
J2=$(sbatch --dependency=afterok:${J1} --parsable -J ${1}_krmp2 --mem=${2}MB --time=${3} krmp2.sh ${1}.json)
J3=$(sbatch --dependency=afterok:${J2} --parsable -J ${1}_imds_1e --mem=${2}MB --time=${3} imds_1e.sh ${1}.json)
J4=$(sbatch --dependency=afterok:${J3} --parsable -J ${1}_imds_2e --mem=${2}MB --time=${3} imds_2e.sh ${1}.json)
J5=$(sbatch --dependency=afterok:${J4} --parsable -J ${1}_peomip --mem=${2}MB --time=${3} peomip.sh ${1}.json)
J6=$(sbatch --dependency=afterok:${J5} --parsable -J ${1}_cvspeomip --mem=${2}MB --time=${3} cvspeomip.sh ${1}.json)
J7=$(sbatch --dependency=afterok:${J6} -J ${1}_core core.sh ${1}.json)
