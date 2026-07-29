#!/bin/bash -l

# Driver-pathway discovery over MSigDB C6 oncogenic signatures (c6.all) ONLY -- a separate
# FDR family from the Reactome and Hallmark runs (never pooled). C6 sets are empirical
# expression signatures of oncogene/tumor-suppressor perturbations (KRAS, EGFR, PIK3CA,
# AKT, MTOR, PTEN, RB, P53, incl. KRAS.LUNG), so they are directional (UP/DN) and read on
# the same footing as the model's expression attributions. Read gsea_prerank.csv (and
# ora_up/ora_down) as the primary result -- direction is the point; ora_magnitude ignores
# it. CPU-only; the debug queue (<=1h) is plenty.

#PBS -A GeomicVar
#PBS -l select=1
#PBS -l walltime=01:00:00
#file systems used by the job
#PBS -l filesystems=home:eagle

#PBS -N pathway_c6

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

# C6 universe -> its own output dir / FDR family. Lower MIN_MEMBERS so smaller oncogenic
# signatures are not dropped.
export COLLECTIONS="c6.all"
export OUT_NAME="pathway_interpret_c6"
export MIN_MEMBERS="10"

# Build the C6 bundle, then run permutation + GSEA + ORA on it.
./joint_fusion/run/run_interpret_pathway.sh
./joint_fusion/run/run_discovery_pathway_tests.sh
