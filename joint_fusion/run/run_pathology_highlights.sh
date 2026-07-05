#!/bin/bash

set -euo pipefail

# Make sure to call from the repo root.
#
# Emits the pathologist-facing ROI overlays: each slide gets a *_boxes.png with
# an unfilled square drawn around every high-scoring tile, placed at its true
# position on the slide. By default (BOXES_ONLY=true) the heatmap and
# heatmap-overlay are skipped. Hand the *_boxes.png files (in
# <OUTPUT_DIR>/overlays) to the pathologist.

# Choose exactly one input mode:
# 1) Directory mode: point to a folder of .npz tile-score files
TILE_SCORES_DIR="checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1/saliency_tile_scores"

# 2) Single-file mode: set a specific .npz file and leave TILE_SCORES_DIR empty
TILE_SCORES_FILE=""

# Output folder for generated *_boxes.png ROI overlays (and heatmap/overlay if
# BOXES_ONLY=false).
OUTPUT_DIR="checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1/pathology_highlights"

# Directory containing full-slide images such as assets/TCGA-50-5044-01Z-00-DX1....png
SLIDE_ASSETS_DIR="assets/png_files_all_samples_per_patient"

# CSV with per-slide full-resolution Width/Height (openslide_getstats.py output).
# REQUIRED for the boxes: they are placed at absolute slide positions, so there
# is no tissue-bbox fallback (a misplaced box is worse than none).
SLIDE_STATS_CSV="assets/slide_statistics_all.csv"

# Tile edge length used when placing each tile on the heatmap canvas
TILE_SIZE=256

# Which scores drive the ROIs. "saliency_scores" for the saliency ROIs, or
# "attention_scores" to box the attention hotspots instead.
SCORE_KEY="saliency_scores"

# ----------------------------------------------------------------------------
# ROI selection (how tiles get boxed)
# ----------------------------------------------------------------------------
# Default: rank-based top (100 - percentile)% of tiles per slide. This gives a
# bounded, consistent number of boxes on EVERY slide (e.g. 90 -> top 10%),
# which is what you want for a comparable pathologist workload.
#   90 -> top 10%   |   95 -> top 5%   |   98 -> top 2%
HIGHLIGHT_PERCENTILE=90

# Optional absolute cutoff on the within-slide min-max-normalized score, in
# [0, 1]. If set (non-empty), it OVERRIDES the percentile. Note this is not
# comparable across slides (each slide's max is 1.0 by construction) and tends
# to be sparse/variable, so prefer the percentile unless you have a reason.
# Example: HIGHLIGHT_THRESHOLD=0.8
HIGHLIGHT_THRESHOLD=""

# Box appearance
BOX_COLOR="lime"
BOX_LINEWIDTH=1.5

# Emit ONLY the *_boxes.png ROI overlays (skip the heatmap and heatmap-overlay).
# Set to false to also produce the heatmap + heatmap-overlay alongside the boxes.
BOXES_ONLY=true

CMD=(
  python -m joint_fusion.testing.generate_slide_heatmap_overlays
  --output-dir "$OUTPUT_DIR"
  --slide-assets-dir "$SLIDE_ASSETS_DIR"
  --slide-stats-csv "$SLIDE_STATS_CSV"
  --tile-size "$TILE_SIZE"
  --highlight-boxes
  --box-color "$BOX_COLOR"
  --box-linewidth "$BOX_LINEWIDTH"
)

if [[ "$BOXES_ONLY" == "true" ]]; then
  CMD+=(--boxes-only)
fi

if [[ -n "$TILE_SCORES_FILE" ]]; then
  CMD+=(--tile-scores "$TILE_SCORES_FILE")
elif [[ -n "$TILE_SCORES_DIR" ]]; then
  CMD+=(--tile-scores-dir "$TILE_SCORES_DIR")
else
  echo "Set either TILE_SCORES_FILE or TILE_SCORES_DIR." >&2
  exit 1
fi

if [[ -n "$SCORE_KEY" ]]; then
  CMD+=(--score-key "$SCORE_KEY")
fi

# Absolute threshold overrides the percentile when provided.
if [[ -n "$HIGHLIGHT_THRESHOLD" ]]; then
  CMD+=(--highlight-threshold "$HIGHLIGHT_THRESHOLD")
else
  CMD+=(--highlight-percentile "$HIGHLIGHT_PERCENTILE")
fi

"${CMD[@]}"
