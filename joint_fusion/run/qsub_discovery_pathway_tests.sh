#!/bin/bash -l

# Discovery pathway tests (permutation null + GSEA + ORA) on the saved analysis bundle.
# CPU-only: this reads pathway_analysis_bundle.npz and needs no GPU. The debug queue
# (<=1h) is plenty; switch -q to capacity and bump walltime only for very large N_PERM.

#PBS -A GeomicVar
#PBS -l select=1
#PBS -l walltime=01:00:00
#file systems used by the job
#PBS -l filesystems=home:eagle

#PBS -N pathway_discovery

#PBS -q debug

# Write stdout/stderr straight to the destination (globally mounted FS).
#PBS -k doe
#PBS -o /grand/GeomicVar/jozhw/multimodal_learning_T1/tmp
#PBS -e /grand/GeomicVar/jozhw/multimodal_learning_T1/tmp
#PBS -j n

#PBS -m be
#PBS -M johnzhouyangwu@hsph.harvard.edu

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# load conda environment
module use /soft/modulefiles
module load conda
conda activate /grand/GeomicVar/embeddings_for_john/multimodal_env

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

# Sanity-check the analysis deps are present in this env before the long run. GSEApy is
# the one most likely to be missing from the training env; if so, either install it or
# export SKIP_GSEA=1 before qsub (it is read by run_discovery_pathway_tests.sh).
python - <<'PY'
import importlib, sys
missing = [m for m in ("numpy", "pandas", "scipy", "statsmodels", "gseapy")
           if importlib.util.find_spec(m) is None]
if missing:
    print(f"WARNING: missing modules in this env: {missing}. "
          "Install them, or set SKIP_GSEA=1 if only gseapy is missing.", file=sys.stderr)
PY

# This is the Reactome discovery run (COLLECTIONS=c2.cp.reactome, OUT_NAME=
# pathway_interpret_reactome). Do NOT pool collections into one run -- each collection is
# tested as its own FDR family. For the driver-pathway analysis use the dedicated, SEPARATE
# presets:
#   qsub joint_fusion/run/qsub_discovery_hallmark.sh    # Hallmark named driver buckets
#   qsub joint_fusion/run/qsub_discovery_c6.sh          # C6 oncogenic signatures
# BUILD_BUNDLE=1 builds the shared, collection-independent gene bundle first -- REQUIRED the
# first time (the tests read gene_attribution_bundle.npz and rebuild membership per
# collection). Once the gene bundle exists, any collection's tests reuse it, so omit
# BUILD_BUNDLE. JOBS / GSEA_THREADS / NCORES / SKIP_GSEA are overridable by exporting them.
export COLLECTIONS="${COLLECTIONS:-c2.cp.reactome}"
export OUT_NAME="${OUT_NAME:-pathway_interpret_reactome}"
if [[ "${BUILD_BUNDLE:-0}" == "1" ]]; then
  ./joint_fusion/run/run_interpret_pathway.sh
fi
./joint_fusion/run/run_discovery_pathway_tests.sh
