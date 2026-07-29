#!/bin/bash

set -euo pipefail

# Run from the repository root, regardless of where the script is launched.
cd "$(dirname "$0")/../.."

# Adjust these values for a different fold or output folder.
CONFIG="${CONFIG:-joint_fusion/config/config_checkpoint_2026-04-07-04-58-17_fold1.yaml}"
MSIGDB_DIR="${MSIGDB_DIR:-assets/msigdb}"

# Which discovery universe to test. Membership is rebuilt from COLLECTIONS on the shared,
# collection-independent gene bundle at test time -- run ONE collection per invocation so
# each keeps its own FDR family (do not pool). OUT_NAME is this collection's output folder
# and should match the one used by run_interpret_pathway.sh:
#   Reactome (default):  COLLECTIONS=c2.cp.reactome  OUT_NAME=pathway_interpret_reactome
#   Hallmark:            COLLECTIONS=h.all           OUT_NAME=pathway_interpret_hallmark
#   C6:                  COLLECTIONS=c6.all          OUT_NAME=pathway_interpret_c6
CKPT_BASE="${CKPT_BASE:-checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1}"
COLLECTIONS="${COLLECTIONS:-c2.cp.reactome}"
OUT_NAME="${OUT_NAME:-pathway_interpret_reactome}"
PATHWAY_DIR="${PATHWAY_DIR:-$CKPT_BASE/$OUT_NAME}"
# The gene bundle is shared across collections (written to CKPT_BASE by pathway_interpret).
GENE_BUNDLE="${GENE_BUNDLE:-$CKPT_BASE/gene_attribution_bundle.npz}"

# PHYSICAL cores to give BLAS/GSEA. Physical, NOT logical: each permutation is a small
# matmul, and setting the thread count to the logical CPU count (SMT/hyperthreads, e.g. 64
# or 128 on a many-core node) OVERSUBSCRIBES it -- benchmarked ~10x slower, effectively a
# standstill. Physical-core count (e.g. 32 on a Polaris node) is 1:1 with the hardware and
# safe. Override by exporting NCORES.
if [[ -z "${NCORES:-}" ]]; then
  if command -v lscpu >/dev/null 2>&1; then
    _sock=$(lscpu | awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2); print $2}')
    _cps=$(lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); print $2}')
    NCORES=$(( ${_sock:-0} * ${_cps:-0} ))
    if [[ "$NCORES" -lt 1 ]]; then NCORES=$(nproc 2>/dev/null || echo 8); fi
  elif [[ "$(uname)" == "Darwin" ]]; then
    NCORES=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 8)
  elif command -v nproc >/dev/null 2>&1; then
    NCORES=$(nproc)
  else
    NCORES=8
  fi
fi
if [[ -z "$NCORES" || "$NCORES" -lt 1 ]]; then NCORES=8; fi

# Give BLAS the physical-core count. This both uses the hardware and guards against an HPC
# module env that pins the thread vars to 1. No --jobs / process pool is needed.
export OMP_NUM_THREADS="$NCORES" OPENBLAS_NUM_THREADS="$NCORES" MKL_NUM_THREADS="$NCORES"
export VECLIB_MAXIMUM_THREADS="$NCORES"
# NumExpr (pulled in by pandas) hard-errors at import if its thread count exceeds
# NUMEXPR_MAX_THREADS (default 64) -- which crashes on nodes with >64 visible CPUs. It is
# not a bottleneck here (the permutation is a BLAS matmul), so cap it at 64.
export NUMEXPR_NUM_THREADS="$(( NCORES < 64 ? NCORES : 64 ))"

# Discovery settings. The discovery universe is whatever collection the bundle was
# built with (Reactome C2:CP by default in pathway_interpret.py).
N_PERM=10000
SEED=40
TAIL="gpd"           # permutation tail below the empirical floor: gpd (default) or empirical
ORA_TOP_N=100                       # top genes per ORA list (magnitude / up / down)
# Size floor for the WHOLE tested family (perm / GSEA / ORA), since membership is now
# rebuilt at test time. 10 matches the documented Reactome discovery (1,118 sets; see
# pathway_tests_statistical_methods.tex) and the gene bundle. Keep it in sync with
# run_interpret_pathway.sh's MIN_MEMBERS for the same collection.
MIN_MEMBERS="${MIN_MEMBERS:-10}"

# GSEA is computed with GSEApy; it must be installed in your active Python env
# (set SKIP_GSEA=1 to run only the permutation stats + ORA). Unlike the permutation,
# GSEApy prerank is genuinely single-threaded unless told otherwise and is the longest
# step at N_PERM=10000, so give it all the cores too. Override with GSEA_THREADS.
GSEA_THREADS="${GSEA_THREADS:-$NCORES}"

# --jobs (process parallelism for the permutation) is NOT used by default: the BLAS
# threading set above already runs the permutation on every core and is the fast path.
# It is only worth setting on a single-threaded BLAS build -- then submit with e.g.
#   JOBS=-1 qsub joint_fusion/run/qsub_discovery_pathway_tests.sh   (or JOBS=$NCORES).

# Set to 1 to skip GSEA (permutation stats + ORA only).
SKIP_GSEA=0

ARGS=(
  --config "$CONFIG"
  --out-dir "$PATHWAY_DIR"
  --gene-bundle "$GENE_BUNDLE"
  --collections "$COLLECTIONS"
  --msigdb-dir "$MSIGDB_DIR"
  --n-perm "$N_PERM"
  --seed "$SEED"
  --tail "$TAIL"
  --ora-top-n "$ORA_TOP_N"
  --min-members "$MIN_MEMBERS"
  --gsea-threads "$GSEA_THREADS"
)

if [[ "$SKIP_GSEA" == "1" ]]; then
  ARGS+=(--skip-gsea)
fi

# Opt-in process parallelism (single-threaded-BLAS fallback only): JOBS=-1 qsub ...
if [[ -n "${JOBS:-}" ]]; then
  ARGS+=(--jobs "$JOBS")
fi

python -m joint_fusion.testing.pathway_tests "${ARGS[@]}"
