#!/bin/bash

set -euo pipefail

# Run from the repository root, regardless of where the script is launched.
cd "$(dirname "$0")/../.."

# Adjust these values for a different fold or output folder.
CONFIG="joint_fusion/config/config_checkpoint_2026-04-07-04-58-17_fold1.yaml"
PATHWAY_DIR="checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1/pathway_interpret"
MSIGDB_DIR="assets/msigdb"

# Discovery settings. The discovery universe is whatever collection the bundle was
# built with (Reactome C2:CP by default in pathway_interpret.py).
N_PERM=1000
SEED=40
TAIL="gpd"           # permutation tail below the empirical floor: gpd (default) or empirical
ORA_TOP_N=100        # top genes per ORA list (magnitude / up / down)
MIN_MEMBERS=10       # drop discovery sets with fewer than this many measured genes
GSEA_THREADS=1
# GSEA is computed with GSEApy; it must be installed in your active Python env
# (set SKIP_GSEA=1 to run only the permutation stats + ORA).

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
  --dir "$PATHWAY_DIR"
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

python -m joint_fusion.testing.pathway_tests "${ARGS[@]}"
