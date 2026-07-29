#!/bin/bash -l

# Driver-pathway discovery over MSigDB HALLMARK (h.all) ONLY -- a separate FDR family from
# the Reactome and C6 runs (never pooled: pooling collections double-counts drivers and
# conflates corrections). Hallmark is 50 minimally-redundant, named sets, so this run gives
# clean driver buckets (PI3K_AKT_MTOR, MTORC1, KRAS_SIGNALING, P53_PATHWAY, ...) with low
# within-family redundancy. CPU-only; the debug queue (<=1h) is plenty.

#PBS -A GeomicVar
#PBS -l select=1
#PBS -l walltime=01:00:00
#file systems used by the job
#PBS -l filesystems=home:eagle

#PBS -N pathway_hallmark

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
echo Running on nodes `cat $PBS_NODEFILE`

python - <<'PY'
import importlib, sys
missing = [m for m in ("numpy", "pandas", "scipy", "statsmodels", "gseapy")
           if importlib.util.find_spec(m) is None]
if missing:
    print(f"WARNING: missing modules in this env: {missing}. "
          "Install them, or set SKIP_GSEA=1 if only gseapy is missing.", file=sys.stderr)
PY

# Hallmark universe -> its own output dir / FDR family.
export COLLECTIONS="h.all"
export OUT_NAME="pathway_interpret_hallmark"
export MIN_MEMBERS="10"

# Build the Hallmark bundle, then run permutation + GSEA + ORA on it.
./joint_fusion/run/run_interpret_pathway.sh
./joint_fusion/run/run_discovery_pathway_tests.sh
