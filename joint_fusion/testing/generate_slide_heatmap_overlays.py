import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from joint_fusion.testing.reconstruct_slide_heatmap import (
    build_heatmap,
    infer_score_key,
)


def args():
    parser = argparse.ArgumentParser(
        description="Generate slide-level heatmaps and overlays from saved tile-score .npz files."
    )
    parser.add_argument(
        "--tile-scores",
        type=str,
        default=None,
        help="Path to a single attention_scores_*.npz or saliency_scores_*.npz file.",
    )
    parser.add_argument(
        "--tile-scores-dir",
        type=str,
        default=None,
        help="Directory containing tile-score .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where heatmaps and overlays will be written.",
    )
    parser.add_argument(
        "--slide-assets-dir",
        type=str,
        default="assets",
        help="Directory containing full-slide images named with the TCGA slide prefix.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile edge length in pixels.",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default=None,
        help="Optional score key override. Defaults to attention_scores or saliency_scores.",
    )
    parser.add_argument(
        "--save-array",
        action="store_true",
        help="Also save the heatmap canvas as a .npy file.",
    )
    parser.add_argument(
        "--slide-stats-csv",
        type=str,
        default=None,
        help="Path to slide_statistics_all.csv (openslide_getstats.py output). Its "
        "per-slide full-resolution Width/Height let the heatmap be placed at its "
        "true absolute position on the thumbnail (no rescaling guess).",
    )
    parser.add_argument(
        "--allow-tissue-fallback",
        action="store_true",
        help="If a slide's dimensions are not found in the stats CSV, fall back to "
        "approximate tissue-bbox registration instead of skipping its overlay.",
    )
    return parser.parse_args()


def infer_slide_key_from_tile_name(tile_name):
    tile_stem = Path(str(tile_name)).stem
    parts = tile_stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Could not infer slide key from tile name: {tile_name}")
    return "-".join(parts[:3])


def resolve_slide_asset(slide_assets_dir, slide_key):
    assets_dir = Path(slide_assets_dir)
    matches = sorted(assets_dir.glob(f"{slide_key}-*.png"))
    if not matches:
        matches = sorted(assets_dir.glob(f"{slide_key}-*"))
    if not matches:
        return None

    for match in matches:
        if "-DX1" in match.name:
            return match
    return matches[0]


def render_heatmap(heatmap, output_path, title):
    plt.figure(figsize=(12, 12))
    plt.imshow(heatmap, cmap="hot", origin="upper")
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_slide_dimensions(stats_csv_path):
    """Map slide identifiers -> (full_res_width, full_res_height) from the
    openslide_getstats.py CSV.

    Returns two dicts: one keyed by the full file stem (e.g.
    ``TCGA-05-4244-01Z-00-DX1.<uuid>``) for exact matches, and one keyed by the
    3-token slide key (``TCGA-05-4244``) as a fallback, preferring the DX1 slide.
    """
    by_stem = {}
    by_key = {}
    with open(stats_csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = row.get("File", "") or ""
            try:
                width = int(float(row["Width"]))
                height = int(float(row["Height"]))
            except (KeyError, ValueError, TypeError):
                continue

            stem = Path(file_name).stem  # strips .svs
            by_stem[stem] = (width, height)

            parts = stem.split("-")
            if len(parts) >= 3:
                key = "-".join(parts[:3])
                if key not in by_key or "-DX1" in stem:
                    by_key[key] = (width, height)
    return by_stem, by_key


def lookup_slide_dimensions(dims_by_stem, dims_by_key, slide_asset_path, slide_key):
    if dims_by_stem is None:
        return None
    dims = dims_by_stem.get(Path(slide_asset_path).stem)
    if dims is None:
        dims = dims_by_key.get(slide_key)
    return dims


def render_overlay_faithful(
    heatmap, heatmap_extent, slide_asset_path, slide_w, slide_h, output_path, title
):
    """Overlay the heatmap at its true absolute position on the slide.

    The thumbnail is drawn spanning the full-resolution slide coordinate frame
    ``(0, slide_w, slide_h, 0)``, and the heatmap is drawn at its own level-0
    ``heatmap_extent``. Both layers therefore share one coordinate system, so
    the mapping is a single uniform scale (the thumbnail's downsample) with no
    per-axis stretching or tissue-detection heuristic.
    """
    slide_image = plt.imread(slide_asset_path)

    plt.figure(figsize=(12, 12))
    plt.imshow(slide_image, origin="upper", extent=(0, slide_w, slide_h, 0))
    plt.imshow(
        heatmap,
        cmap="hot",
        origin="upper",
        extent=heatmap_extent,
        alpha=0.45,
    )
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def detect_tissue_bbox(slide_image, dark_threshold=150):
    """Bounding box (x0, y0, x1, y1) of tissue in the slide thumbnail.

    Mirrors the tissue test used in preprocessing/remove_background.py: pixels
    darker than ``dark_threshold`` (on a 0-255 scale) count as tissue. Returns
    the full-image box if no tissue is found.
    """
    arr = np.asarray(slide_image)
    if arr.ndim == 3:
        gray = arr[..., :3].mean(axis=2)
    else:
        gray = arr
    # matplotlib returns floats in [0, 1] for PNGs; rescale the threshold then.
    threshold = dark_threshold / 255.0 if gray.max() <= 1.0 else dark_threshold

    mask = gray < threshold
    height, width = gray.shape
    if not mask.any():
        return 0, 0, width, height

    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def render_overlay_tissue(heatmap, slide_asset_path, output_path, title):
    """Approximate fallback when slide dimensions are unavailable.

    Registers the heatmap's bounding box to the thumbnail's detected tissue box.
    This is only an estimate of the slide downsample (it scales x and y
    independently and depends on tissue detection), so it can skew the geometry;
    prefer render_overlay_faithful whenever slide dimensions are known.
    """
    slide_image = plt.imread(slide_asset_path)
    slide_h, slide_w = slide_image.shape[:2]
    x0, y0, x1, y1 = detect_tissue_bbox(slide_image)

    plt.figure(figsize=(12, 12))
    plt.imshow(
        slide_image,
        origin="upper",
        extent=(0, slide_w, slide_h, 0),
    )
    plt.imshow(
        heatmap,
        cmap="hot",
        origin="upper",
        extent=(x0, x1, y1, y0),
        alpha=0.45,
    )
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def iter_tile_score_files(tile_scores, tile_scores_dir):
    if tile_scores is not None:
        yield Path(tile_scores)
        return

    if tile_scores_dir is None:
        raise ValueError("Provide either --tile-scores or --tile-scores-dir.")

    for path in sorted(Path(tile_scores_dir).glob("*.npz")):
        yield path


def process_tile_score_file(
    path, output_dir, slide_assets_dir, tile_size, score_key, save_array,
    dims_by_stem=None, dims_by_key=None, allow_tissue_fallback=False,
):
    payload = np.load(path, allow_pickle=True)
    resolved_score_key = infer_score_key(payload, score_key)
    heatmap, heatmap_extent = build_heatmap(payload, resolved_score_key, tile_size)
    title = resolved_score_key.replace("_", " ")

    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    heatmap_output = heatmap_dir / f"{path.stem}_heatmap.png"
    render_heatmap(heatmap, heatmap_output, title)

    if save_array:
        np.save(heatmap_output.with_suffix(".npy"), heatmap)

    tile_names = payload.get("tile_names")
    if tile_names is None or len(tile_names) == 0:
        return

    slide_key = infer_slide_key_from_tile_name(tile_names[0])
    slide_asset_path = resolve_slide_asset(slide_assets_dir, slide_key)
    if slide_asset_path is None:
        print(f"[warn] no slide asset found for {slide_key}")
        return

    overlay_output = overlay_dir / f"{path.stem}_overlay.png"
    dims = lookup_slide_dimensions(
        dims_by_stem, dims_by_key, slide_asset_path, slide_key
    )

    if dims is not None:
        slide_w, slide_h = dims
        render_overlay_faithful(
            heatmap,
            heatmap_extent,
            slide_asset_path,
            slide_w,
            slide_h,
            overlay_output,
            title,
        )
        mode = "faithful"
    elif allow_tissue_fallback:
        render_overlay_tissue(heatmap, slide_asset_path, overlay_output, title)
        mode = "tissue-fallback"
    else:
        print(
            f"[warn] no slide dimensions for {slide_key} in stats CSV; "
            f"skipping overlay (use --allow-tissue-fallback to approximate)"
        )
        return

    print(
        f"[ok] {path.name} -> {heatmap_output.name}, {overlay_output.name} ({mode})"
    )


def main():
    opt = args()
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dims_by_stem = dims_by_key = None
    if opt.slide_stats_csv:
        dims_by_stem, dims_by_key = load_slide_dimensions(opt.slide_stats_csv)
        print(
            f"[info] loaded full-resolution dimensions for {len(dims_by_stem)} "
            f"slides from {opt.slide_stats_csv}"
        )
    else:
        print(
            "[warn] no --slide-stats-csv provided; overlays need slide dimensions "
            "for faithful placement (use --allow-tissue-fallback to approximate)"
        )

    for path in iter_tile_score_files(opt.tile_scores, opt.tile_scores_dir):
        process_tile_score_file(
            path=path,
            output_dir=output_dir,
            slide_assets_dir=opt.slide_assets_dir,
            tile_size=opt.tile_size,
            score_key=opt.score_key,
            save_array=opt.save_array,
            dims_by_stem=dims_by_stem,
            dims_by_key=dims_by_key,
            allow_tissue_fallback=opt.allow_tissue_fallback,
        )


if __name__ == "__main__":
    main()
