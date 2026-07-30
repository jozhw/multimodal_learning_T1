#!/bin/bash

set -euo pipefail

# Discovery tests for HALLMARK. Thin wrapper: it pins this collection and delegates to the
# shared engine run_discovery_pathway_tests.sh (so the threading / N_PERM / arg logic lives in
# one place). Run directly (./run_discovery_hallmark.sh) or via qsub_discovery_hallmark.sh.
# N_PERM / SEED / TAIL / SKIP_GSEA / JOBS are still overridable, e.g. N_PERM=1000 ./this.

cd "$(dirname "$0")/../.."
export COLLECTIONS="h.all"
export OUT_NAME="pathway_interpret_hallmark"
export MIN_MEMBERS="${MIN_MEMBERS:-10}"
exec ./joint_fusion/run/run_discovery_pathway_tests.sh
