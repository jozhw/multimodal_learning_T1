#!/bin/bash

set -euo pipefail

# Discovery tests for REACTOME. Thin wrapper: it pins this collection and delegates to the
# shared engine run_discovery_pathway_tests.sh (so the threading / N_PERM / arg logic lives in
# one place). Run directly (./run_discovery_reactome.sh) or via qsub_discovery_reactome.sh.
# N_PERM / SEED / TAIL / SKIP_GSEA / SKIP_GSEA_PLOTS / JOBS are still overridable, e.g. N_PERM=1000 ./this.

cd "$(dirname "$0")/../.."
export COLLECTIONS="c2.cp.reactome"
export OUT_NAME="pathway_interpret_reactome"
export MIN_MEMBERS="${MIN_MEMBERS:-10}"
exec ./joint_fusion/run/run_discovery_pathway_tests.sh
