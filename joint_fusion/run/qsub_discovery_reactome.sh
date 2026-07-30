#!/bin/bash -l

# TEST step, Reactome. Runs run_discovery_pathway_tests.sh -> permutation / GSEA / ORA into
# pathway_interpret_reactome/. REQUIRES the gene bundle from qsub_interpret_reactome.sh (run
# that first). Uses the capacity queue with a long walltime so a full N_PERM=10000 run is not
# killed mid-permutation. For a fast first pass submit with:  N_PERM=1000 qsub <this>.

#PBS -A GeomicVar
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:eagle
#PBS -N pw_test_reactome
#PBS -q capacity
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

# The tests need scipy/statsmodels (permutation FDR) and gseapy (GSEA). Warn if missing.
python - <<'PY'
import importlib, sys
missing = [m for m in ("numpy", "pandas", "scipy", "statsmodels", "gseapy")
           if importlib.util.find_spec(m) is None]
if missing:
    print(f"WARNING: missing modules in this env: {missing}. "
          "Install them, or set SKIP_GSEA=1 if only gseapy is missing.", file=sys.stderr)
PY

./joint_fusion/run/run_discovery_reactome.sh
