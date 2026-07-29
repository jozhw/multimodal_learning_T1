#!/bin/bash

set -euo pipefail

# Run from the repository root, regardless of where the script is launched.
cd "$(dirname "$0")/../.."

# Builds the per-collection scores/figures AND the shared, collection-INDEPENDENT gene
# bundle (gene_attribution_bundle.npz, written to CKPT_BASE). run_discovery_pathway_tests.sh
# reads that one gene bundle and rebuilds membership for whatever COLLECTIONS it tests, so
# the gene bundle only needs building once; re-run this per collection only if you also want
# that collection's scores.csv / figures.

CONFIG="${CONFIG:-joint_fusion/config/config_checkpoint_2026-04-07-04-58-17_fold1.yaml}"
MSIGDB_DIR="${MSIGDB_DIR:-assets/msigdb}"
CKPT_BASE="${CKPT_BASE:-checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1}"

# --- Discovery universe -------------------------------------------------------------------
# Choose ONE (or override COLLECTIONS/OUT_NAME from the environment). Run ONE collection per
# invocation so each keeps its own FDR family downstream (do not pool):
#   Reactome (default): COLLECTIONS=c2.cp.reactome  OUT_NAME=pathway_interpret_reactome
#   Hallmark:           COLLECTIONS=h.all           OUT_NAME=pathway_interpret_hallmark
#   C6:                 COLLECTIONS=c6.all          OUT_NAME=pathway_interpret_c6
# Each universe writes its scores/figures to its own OUT_NAME dir so they coexist.
COLLECTIONS="${COLLECTIONS:-c2.cp.reactome}"
OUT_NAME="${OUT_NAME:-pathway_interpret_reactome}"

# Min measured members per set. Reactome/Hallmark sets are large; C6 signatures vary and
# some are small, so lower the floor when testing C6 or you will drop them. Keep this value
# in sync with run_discovery_pathway_tests.sh's MIN_MEMBERS for the same collection.
MIN_MEMBERS="${MIN_MEMBERS:-10}"
TOP_N="${TOP_N:-20}"

OUT_DIR="$CKPT_BASE/$OUT_NAME"
GENE_BUNDLE="${GENE_BUNDLE:-$CKPT_BASE/gene_attribution_bundle.npz}"

echo "Building: collections='$COLLECTIONS' -> scores/figures in $OUT_DIR, gene bundle $GENE_BUNDLE (min_members=$MIN_MEMBERS)"

python -m joint_fusion.testing.pathway_interpret \
  --config "$CONFIG" \
  --msigdb-dir "$MSIGDB_DIR" \
  --collections "$COLLECTIONS" \
  --output-dir "$OUT_DIR" \
  --gene-bundle "$GENE_BUNDLE" \
  --min-members "$MIN_MEMBERS" \
  --top-n "$TOP_N" \
  --no-supplemental-all
