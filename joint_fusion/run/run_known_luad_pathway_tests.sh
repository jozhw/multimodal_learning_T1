#!/bin/bash

set -euo pipefail

# Run from the repository root, regardless of where the script is launched.
cd "$(dirname "$0")/../.."

# Adjust these values for a different fold or output folder.
CONFIG="joint_fusion/config/config_checkpoint_2026-04-07-04-58-17_fold1.yaml"
PATHWAY_DIR="checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1/pathway_interpret"
MSIGDB_DIR="assets/msigdb"

# Known LUAD/NSCLC panel settings.
N_PERM=10000
N_BOOT=5000
SEED=40
PANEL_MIN_MEMBERS=3
CI_LEVEL=0.95
GSEA_THREADS=1
# Permutation tail model below the empirical floor: gpd (default, Knijnenburg 2009
# Generalized-Pareto hybrid, estimates the tail shape from the data) or empirical.
TAIL="gpd"
# GSEA is computed with GSEApy; it must be installed in your active Python env
# (set SKIP_GSEA=1 below to run only the permutation/bootstrap table without it).

# Set to 0 if you also want the broad Reactome permutation/GSEA/ORA outputs.
ONLY_KNOWN_LUAD=1

# Set to 1 to skip GSEA and only run the permutation/bootstrap table.
SKIP_GSEA=0

ARGS=(
  --config "$CONFIG"
  --dir "$PATHWAY_DIR"
  --msigdb-dir "$MSIGDB_DIR"
  --n-perm "$N_PERM"
  --n-boot "$N_BOOT"
  --seed "$SEED"
  --panel-min-members "$PANEL_MIN_MEMBERS"
  --ci-level "$CI_LEVEL"
  --gsea-threads "$GSEA_THREADS"
  --tail "$TAIL"
)

if [[ "$ONLY_KNOWN_LUAD" == "1" ]]; then
  ARGS+=(--only-known-luad)
fi

if [[ "$SKIP_GSEA" == "1" ]]; then
  ARGS+=(--skip-gsea)
fi

python -m joint_fusion.testing.pathway_tests "${ARGS[@]}"
