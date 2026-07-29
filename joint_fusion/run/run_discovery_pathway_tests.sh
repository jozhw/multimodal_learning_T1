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

NCORES="${NCORES:-$(python -c 'import os; print(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4))' 2>/dev/null || echo 4)}"

export OMP_NUM_THREADS="$NCORES" OPENBLAS_NUM_THREADS="$NCORES" MKL_NUM_THREADS="$NCORES"
export VECLIB_MAXIMUM_THREADS="$NCORES" NUMEXPR_NUM_THREADS="$NCORES"

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

# Set to 1 to skip GSEA (permutation stats + ORA only).
SKIP_GSEA=0

# The known-LUAD/NSCLC panel has its own script (run_known_luad_pathway_tests.sh),
# so it is skipped here by default to avoid overwriting that output. Set to 0 to
# also run the panel as part of this discovery pass (uses the settings below).
SKIP_KNOWN_LUAD=1
N_BOOT=5000
CI_LEVEL=0.95
PANEL_MIN_MEMBERS=3

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
  --n-boot "$N_BOOT"
  --ci-level "$CI_LEVEL"
  --panel-min-members "$PANEL_MIN_MEMBERS"
)

if [[ "$SKIP_GSEA" == "1" ]]; then
  ARGS+=(--skip-gsea)
fi

if [[ "$SKIP_KNOWN_LUAD" == "1" ]]; then
  ARGS+=(--skip-known-luad)
fi

if [[ -n "${JOBS:-}" ]]; then
  ARGS+=(--jobs "$JOBS")
fi

python -m joint_fusion.testing.pathway_tests "${ARGS[@]}"
