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

# Discovery settings. The discovery universe is whatever collection the bundle was
# built with (Reactome C2:CP by default in pathway_interpret.py).
# N_PERM is overridable: e.g. N_PERM=1000 for a fast first pass (10x cheaper); 10000 is the
# documented run. The GPD tail still works at 1000 (empirical floor 1/1001, then extrapolated).
N_PERM="${N_PERM:-10000}"
SEED="${SEED:-40}"
TAIL="${TAIL:-exponential}"  # tail below the empirical floor: exponential (default, xi=0) or empirical
ORA_TOP_N=100                       # top genes per ORA list (magnitude / up / down)
# Size floor for the WHOLE tested family (perm / GSEA / ORA), since membership is now
# rebuilt at test time. 10 matches the documented Reactome discovery (1,118 sets; see
# pathway_tests_statistical_methods.tex) and the gene bundle. Keep it in sync with
# run_interpret_pathway.sh's MIN_MEMBERS for the same collection.
MIN_MEMBERS="${MIN_MEMBERS:-10}"

# GSEA (GSEApy) and ORA are NOT parallelised here -- GSEApy runs at its own default and both
# are fast enough. Forcing a high thread count made GSEApy's Rust backend crash on the node's
# thread limit, so we do not pass a thread count at all. Set SKIP_GSEA=1 to run permutation +
# ORA only.
SKIP_GSEA=0

# GSEA enrichment plots are ON by default: GSEApy writes one enrichment-score plot per gene
# set (plus report files) into $PATHWAY_DIR/gsea_plots/. That is one image per tested set,
# so a large collection (e.g. Reactome) produces many files. Set SKIP_GSEA_PLOTS=1 to run
# GSEA without drawing plots.
SKIP_GSEA_PLOTS="${SKIP_GSEA_PLOTS:-0}"

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
)

if [[ "$SKIP_GSEA" == "1" ]]; then
  ARGS+=(--skip-gsea)
fi

if [[ "$SKIP_GSEA_PLOTS" == "1" ]]; then
  ARGS+=(--skip-gsea-plots)
fi

# Opt-in process parallelism (single-threaded-BLAS fallback only): JOBS=-1 qsub ...
if [[ -n "${JOBS:-}" ]]; then
  ARGS+=(--jobs "$JOBS")
fi

python -m joint_fusion.testing.pathway_tests "${ARGS[@]}"
