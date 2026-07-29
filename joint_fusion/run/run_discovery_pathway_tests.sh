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

# BLAS threading is LEFT TO NUMPY (do not set OMP/OPENBLAS/MKL here). The permutation is a
# BLAS matmul that numpy already multithreads at its own default (usually the physical-core
# count), exactly as the build step (run_interpret_pathway.sh) does -- and that works. My
# earlier attempts to force the count caused oversubscription / slowdowns, so we no longer
# touch it. If your site's module env pins OMP_NUM_THREADS=1 (single-threaded, slow) or you
# want a specific count, export it yourself before running, e.g. OMP_NUM_THREADS=32.
#
# One guard we DO keep: NumExpr (pulled in by pandas) hard-errors at import if
# NUMEXPR_NUM_THREADS exceeds NUMEXPR_MAX_THREADS (default 64). Only cap it if it is already
# set too high in the environment; otherwise leave NumExpr's own default alone.
if [[ -n "${NUMEXPR_NUM_THREADS:-}" && "${NUMEXPR_NUM_THREADS}" -gt 64 ]]; then
  export NUMEXPR_NUM_THREADS=64
fi

# Threads for GSEApy prerank only (it is single-threaded unless told; not BLAS). Defaults to
# numpy's view of the core count; override with GSEA_THREADS.
NCORES="${NCORES:-$(python -c 'import os; print(os.cpu_count() or 8)' 2>/dev/null || echo 8)}"

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
