#!/bin/bash -l

# BUILD step, Reactome. Runs run_interpret_pathway.sh -> pathway_interpret_reactome/ scores +
# figures AND the shared, collection-independent gene_attribution_bundle.npz. Run this BEFORE
# the matching test job (qsub_discovery_reactome.sh). Build is quick, so the debug queue is fine.

#PBS -A GeomicVar
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -N pw_build_reactome
#PBS -q debug
#PBS -k doe
#PBS -o /grand/GeomicVar/jozhw/multimodal_learning_T1/tmp
#PBS -e /grand/GeomicVar/jozhw/multimodal_learning_T1/tmp
#PBS -j n
#PBS -m be
#PBS -M johnzhouyangwu@hsph.harvard.edu

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

module use /soft/modulefiles
module load conda
conda activate /grand/GeomicVar/embeddings_for_john/multimodal_env

echo Jobid: $PBS_JOBID
echo Running on host `hostname`

export COLLECTIONS="c2.cp.reactome"
export OUT_NAME="pathway_interpret_reactome"
export MIN_MEMBERS="10"
./joint_fusion/run/run_interpret_pathway.sh
