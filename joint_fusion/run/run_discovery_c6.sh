#!/bin/bash

set -euo pipefail

# Discovery tests for C6. Thin wrapper: it pins this collection and delegates to the shared
# engine run_discovery_pathway_tests.sh (so the threading / N_PERM / arg logic lives in one
# place). Run directly (./run_discovery_c6.sh) or via qsub_discovery_c6.sh. C6 sets are
# directional (UP/DN) -- read gsea_prerank.csv / ora_up / ora_down as the primary result.
# N_PERM / SEED / TAIL / SKIP_GSEA / JOBS are still overridable, e.g. N_PERM=1000 ./this.

cd "$(dirname "$0")/../.."
export COLLECTIONS="c6.all"
export OUT_NAME="pathway_interpret_c6"
export MIN_MEMBERS="${MIN_MEMBERS:-10}"
exec ./joint_fusion/run/run_discovery_pathway_tests.sh
