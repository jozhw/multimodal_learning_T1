import argparse
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


def render_overlay(heatmap, slide_asset_path, output_path, title):
    slide_image = plt.imread(slide_asset_path)
    height, width = heatmap.shape

    plt.figure(figsize=(12, 12))
    plt.imshow(
        slide_image,
        origin="upper",
        extent=(0, width, height, 0),
    )
    plt.imshow(
        heatmap,
        cmap="hot",
        origin="upper",
        extent=(0, width, height, 0),
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


def process_tile_score_file(path, output_dir, slide_assets_dir, tile_size, score_key, save_array):
    payload = np.load(path, allow_pickle=True)
    resolved_score_key = infer_score_key(payload, score_key)
    heatmap = build_heatmap(payload, resolved_score_key, tile_size)
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
    render_overlay(heatmap, slide_asset_path, overlay_output, title)
    print(f"[ok] {path.name} -> {heatmap_output.name}, {overlay_output.name}")


def main():
    opt = args()
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in iter_tile_score_files(opt.tile_scores, opt.tile_scores_dir):
        process_tile_score_file(
            path=path,
            output_dir=output_dir,
            slide_assets_dir=opt.slide_assets_dir,
            tile_size=opt.tile_size,
            score_key=opt.score_key,
            save_array=opt.save_array,
        )


if __name__ == "__main__":
    main()
