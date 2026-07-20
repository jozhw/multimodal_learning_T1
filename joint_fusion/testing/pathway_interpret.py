"""
pathway_interpret.py

Pathway-level interpretability for the joint-fusion LUAD survival model, following
Steyaert et al. (2023, Commun Med 3:44): aggregate the model's per-gene gradients
onto MSigDB Reactome pathways, rank them, and draw the figures.

Three per-gene signals, written per patient by test.py into IG_6sep/:
  integrated_grads   attribution magnitude          -> separate IG magnitude table
  path_gradients     signed effect on risk          -> IG-consistent comparison plot
  vanilla_gradients  raw endpoint gradient           -> primary Steyaert-style plot

Direction comes from the gradient, NOT the sign of IG: the IG baseline is cohort-mean
expression, so IG's sign describes the patient (above/below average), not the gene, and
cancels across the cohort. Magnitude survives, direction does not. Derivation:
literature/ig_pathway_design_notes.md.

Database-only: gene sets and genes come from MSigDB; the primary analysis uses
Reactome only, matching Steyaert's pathway universe. Optional all-collections output
can be written for sensitivity analysis. Lung/NSCLC relevance is flagged solely by
reading MSigDB's own gene-set names (LUNG_NAME_TOKENS), never hand-curated and never
used to rank or test.

Layers:
  A  score each pathway = mean gradient over its member genes; rank by summed
     |vanilla gradient| for the primary Steyaert-style result. IG magnitude is exported
     as a separate baseline-aware attribution table/plot.
  B  drill-down: each top pathway's member genes, ranked by their own attribution.

Optional statistical evidence (permutation null, GSEA prerank, hypergeometric ORA)
lives in pathway_tests.py -- run it on the saved bundle when you want p-values.

Inputs (offline):
  <output_base_dir>/IG_6sep/   integrated_grads_*, path_gradients_*, vanilla_gradients_*,
                               omic_input_*  (test.py)
  assets/msigdb/               .gmt gene sets + Ensembl->symbol .chip
                               (fetch once: python -m joint_fusion.fetch_msigdb)

Outputs (<output_base_dir>/pathway_interpret/): pathway_scores.csv (+ lung-named subset),
pathway_ig_magnitude_scores.csv, per-patient and per-gene beeswarms, member-gene table,
direction/gradient plots, the score matrices, pathway_analysis_bundle.npz
(for pathway_tests.py), run_metadata.json.

Run:
  python -m joint_fusion.testing.pathway_interpret \
      --config=joint_fusion/config/config_checkpoint_2026-04-07-04-58-17_fold1.yaml
"""

import argparse
import glob
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from joint_fusion.config.config_manager import ConfigManager
from joint_fusion.testing.interpret_omics import (
    annotate_genes,
    clean_ensembl_id,
    load_gene_names,
)
from joint_fusion.utils.logging import setup_logging

logger = logging.getLogger(__name__)

MSIGDB_DIR = "assets/msigdb"
GENE_AXIS_CACHE = "assets/gene_axis.txt"

# Primary collection: Reactome C2:CP, matching the Steyaert pathway-gradient figure.
PRIMARY_COLLECTIONS = ("c2.cp.reactome",)

# Sensitivity universe: keep broader pathway/signature collections out of the headline
# plot, but make it easy to generate a supplemental table.
ALL_COLLECTIONS = (
    "h.all",
    "c2.cp.reactome",
    "c2.cp.kegg_legacy",
    "c2.cp.kegg_medicus",
    "c6.all",
)

DEFAULT_COLLECTIONS = PRIMARY_COLLECTIONS

RANKING_COLUMNS = {
    "vanilla_gradient_abs": (
        "summed_abs_vanilla_gradient",
        "sum over patients of |mean member-gene vanilla gradient|",
    ),
    "path_gradient_abs": (
        "summed_abs_path_gradient",
        "sum over patients of |mean member-gene path-averaged gradient|",
    ),
    "ig_abs": (
        "summed_abs_ig",
        "sum over patients of |mean member-gene IG attribution|",
    ),
}

# Lung/NSCLC flag reads MSigDB's own gene-set names -- no hand-authored biology.
# Reactome-primary runs may have few or no literal lung-named pathways; the
# all-collections supplement can include KEGG lung cancer maps and C6 KRAS.LUNG sets.
LUNG_NAME_TOKENS = ("LUNG", "NSCLC", "ADENOCARCINOMA")

# Confirmatory panel (NOT discovery): the KEGG NSCLC driver network nt06266 (hsa05223),
# decomposed into its N-numbered driver perturbation modules as packaged in MSigDB
# KEGG_MEDICUS. Reports how the model's attributions land on the established NSCLC
# drivers; it never enters the Reactome discovery ranking. (N00041 EGFR- and N00022
# ERBB2-overexpression -> RAS-ERK have no MSigDB set, so they are omitted.)
KEGG_NSCLC_PANEL = {
    "KEGG_MEDICUS_VARIANT_MUTATION_ACTIVATED_EGFR_TO_RAS_ERK_SIGNALING_PATHWAY": "N00014 EGFR->RAS-ERK",
    "KEGG_MEDICUS_VARIANT_MUTATION_ACTIVATED_MET_TO_RAS_ERK_SIGNALING_PATHWAY": "N01062 MET->RAS-ERK",
    "KEGG_MEDICUS_VARIANT_RET_FUSION_KINASE_TO_RAS_ERK_SIGNALING_PATHWAY": "N00008 RET-fusion->RAS-ERK",
    "KEGG_MEDICUS_VARIANT_EML4_ALK_FUSION_KINASE_TO_RAS_ERK_SIGNALING_PATHWAY": "N00007 EML4-ALK->RAS-ERK",
    "KEGG_MEDICUS_VARIANT_MUTATION_ACTIVATED_KRAS_NRAS_TO_ERK_SIGNALING_PATHWAY": "N00012 KRAS/NRAS->ERK",
    "KEGG_MEDICUS_VARIANT_MUTATION_ACTIVATED_EGFR_TO_PI3K_SIGNALING_PATHWAY": "N00036 EGFR->PI3K",
    "KEGG_MEDICUS_VARIANT_MUTATION_ACTIVATED_MET_TO_PI3K_SIGNALING_PATHWAY": "N01063 MET->PI3K",
    "KEGG_MEDICUS_VARIANT_EML4_ALK_FUSION_KINASE_TO_PI3K_SIGNALING_PATHWAY": "N00047 EML4-ALK->PI3K",
    "KEGG_MEDICUS_VARIANT_MUTATION_ACTIVATED_KRAS_NRAS_TO_PI3K_SIGNALING_PATHWAY": "N00032 KRAS/NRAS->PI3K",
    "KEGG_MEDICUS_VARIANT_LOSS_OF_RASSF1_TO_RAS_RASSF1_SIGNALING_PATHWAY": "N00097 RASSF1-loss",
    "KEGG_MEDICUS_VARIANT_EML4_ALK_FUSION_KINASE_TO_JAK_STAT_SIGNALING_PATHWAY": "N00105 EML4-ALK->JAK-STAT",
    "KEGG_MEDICUS_VARIANT_MUTATION_ACTIVATED_EGFR_TO_PLCG_ERK_SIGNALING_PATHWAY": "N00024 EGFR->PLCG-ERK",
    "KEGG_MEDICUS_VARIANT_EML4_ALK_FUSION_KINASE_TO_PLCG_ERK_SIGNALING_PATHWAY": "N00025 EML4-ALK->PLCG-ERK",
    "KEGG_MEDICUS_REFERENCE_P16_CELL_CYCLE_G1_S": "N00070 p16/CDKN2A->G1/S",
    "KEGG_MEDICUS_VARIANT_MUTATION_INACTIVATED_TP53_TO_TRANSCRIPTION": "N00115 TP53->transcription",
}


# ---------------------------------------------------------------------------
# Gene sets and the Ensembl -> symbol map (MSigDB, offline)
# ---------------------------------------------------------------------------


def load_gmt(path):
    """Parse an MSigDB .gmt into {pathway_name: set(gene_symbols)}."""
    gene_sets = {}
    with open(path) as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            name, _url, *members = fields
            members = {m.strip().upper() for m in members if m.strip()}
            if members:
                gene_sets[name] = members
    logger.info(f"{os.path.basename(path)}: {len(gene_sets)} gene sets")
    return gene_sets


def load_collections(msigdb_dir=MSIGDB_DIR, collections=DEFAULT_COLLECTIONS):
    """Load every requested collection, tagging each set with its source collection."""
    gene_sets, source = {}, {}
    for prefix in collections:
        matches = sorted(glob.glob(os.path.join(msigdb_dir, f"{prefix}.v*.symbols.gmt")))
        if not matches:
            raise FileNotFoundError(
                f"No .gmt for collection '{prefix}' in {msigdb_dir}. "
                "Run: python -m joint_fusion.fetch_msigdb"
            )
        for name, members in load_gmt(matches[-1]).items():
            gene_sets[name] = members
            source[name] = prefix
    logger.info(f"{len(gene_sets)} gene sets total from {len(collections)} collections")
    return gene_sets, source


def load_chip(msigdb_dir=MSIGDB_DIR):
    """Parse the MSigDB Ensembl .chip into {clean_ensembl_id: HGNC symbol}.

    MSigDB's own annotation file keeps the identifier map and gene sets on one release,
    and the run offline.
    """
    matches = sorted(glob.glob(os.path.join(msigdb_dir, "Human_Ensembl_Gene_ID_MSigDB.v*.chip")))
    if not matches:
        raise FileNotFoundError(
            f"No Ensembl .chip in {msigdb_dir}. Run: python -m joint_fusion.fetch_msigdb"
        )
    chip = pd.read_csv(matches[-1], sep="\t", dtype=str)
    mapping = {}
    for probe, symbol in zip(chip["Probe Set ID"], chip["Gene Symbol"]):
        if not isinstance(symbol, str) or not symbol.strip() or symbol.strip() == "---":
            continue
        mapping[str(probe).strip()] = symbol.strip().upper()
    logger.info(f"{os.path.basename(matches[-1])}: {len(mapping)} Ensembl -> symbol")
    return mapping, os.path.basename(matches[-1])


# ---------------------------------------------------------------------------
# Load the per-patient attribution matrices written by test.py
# ---------------------------------------------------------------------------


def load_matrix(ig_directory, prefix, n_genes, patient_ids=None):
    """Stack <prefix>_<TCGA>.npy into a (patients x genes) matrix.

    If ``patient_ids`` is given, load exactly those patients in that order so every
    matrix (IG, gradients, expression) is row-aligned; missing files are an error
    rather than a silent misalignment.
    """
    if patient_ids is None:
        files = sorted(glob.glob(os.path.join(ig_directory, f"{prefix}_*.npy")))
        if not files:
            return None, []
        patient_ids = [
            os.path.basename(f)[len(prefix) + 1 : -4] for f in files
        ]
    else:
        files = [
            os.path.join(ig_directory, f"{prefix}_{pid}.npy") for pid in patient_ids
        ]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            return None, []

    rows, kept = [], []
    for pid, path in zip(patient_ids, files):
        values = np.asarray(np.load(path)).flatten()
        if len(values) != n_genes:
            logger.warning(f"Skipping {prefix} for {pid}: {len(values)} != {n_genes}")
            continue
        rows.append(values)
        kept.append(pid)

    if not rows:
        return None, []
    return np.vstack(rows).astype(np.float64), kept


def load_attributions(ig_directory, n_genes, require_gradients=True):
    """Load the IG, path-gradient, vanilla-gradient and expression matrices, row-aligned.

    Gradients are written by the current test.py; an older IG run has only
    integrated_grads_*.npy, so direction is unavailable -- hence the hard failure unless
    require_gradients is False.
    """
    ig, patient_ids = load_matrix(ig_directory, "integrated_grads", n_genes)
    if ig is None:
        raise FileNotFoundError(f"No integrated_grads_*.npy in {ig_directory}")
    logger.info(f"IG: {ig.shape[0]} patients x {ig.shape[1]} genes")

    path_gradients, _ = load_matrix(
        ig_directory, "path_gradients", n_genes, patient_ids
    )
    vanilla_gradients, _ = load_matrix(
        ig_directory, "vanilla_gradients", n_genes, patient_ids
    )
    expression, _ = load_matrix(ig_directory, "omic_input", n_genes, patient_ids)

    missing = []
    if path_gradients is None:
        missing.append("path_gradients_*.npy")
    if vanilla_gradients is None:
        missing.append("vanilla_gradients_*.npy")
    if missing:
        message = (
            f"Missing {', '.join(missing)} in {ig_directory}. The direction of a gene's "
            "effect on risk cannot be recovered from signed cohort-mean-baseline IG; "
            "re-run test.py, which now exports path and vanilla gradients alongside "
            "the attributions."
        )
        if require_gradients:
            raise FileNotFoundError(message)
        logger.error(message + " Proceeding with --allow-missing-gradients: all "
                     "missing direction columns will be NaN and gradient plots/GSEA "
                     "may be skipped.")

    return ig, path_gradients, vanilla_gradients, expression, patient_ids


# ---------------------------------------------------------------------------
# Ensembl axis -> symbol axis
# ---------------------------------------------------------------------------


def collapse_to_symbols(matrices, ensembl_ids, chip):
    """Map the Ensembl gene axis onto MSigDB's HGNC-symbol axis.

    Genes with no symbol are dropped. Several Ensembl IDs mapping to one symbol are
    AVERAGED (summing would let a duplicated symbol dominate a pathway mean).
    Returns (collapsed_matrices, symbols, stats).
    """
    symbols = [chip.get(clean_ensembl_id(e)) for e in ensembl_ids]
    resolved = [i for i, s in enumerate(symbols) if s]

    groups = {}
    for i in resolved:
        groups.setdefault(symbols[i], []).append(i)

    ordered_symbols = sorted(groups)
    index_groups = [groups[s] for s in ordered_symbols]

    collapsed = {}
    for key, matrix in matrices.items():
        if matrix is None:
            collapsed[key] = None
            continue
        collapsed[key] = np.column_stack(
            [matrix[:, idx].mean(axis=1) for idx in index_groups]
        )

    stats = {
        "n_ensembl_ids": len(ensembl_ids),
        "n_unmapped_to_symbol": len(ensembl_ids) - len(resolved),
        "n_symbols": len(ordered_symbols),
        "n_symbols_from_multiple_ensembl": sum(1 for g in index_groups if len(g) > 1),
    }
    logger.info(
        f"gene axis: {stats['n_ensembl_ids']} Ensembl IDs -> {stats['n_symbols']} symbols "
        f"({stats['n_unmapped_to_symbol']} unmapped, dropped; "
        f"{stats['n_symbols_from_multiple_ensembl']} symbols collapsed from >1 ID)"
    )
    return collapsed, ordered_symbols, stats


# ---------------------------------------------------------------------------
# Layer A: pathway scores (the paper's step)
# ---------------------------------------------------------------------------


def build_membership(gene_sets, symbols, min_members=10):
    """Binary (pathway x gene) membership matrix, restricted to the measured axis.

    Sets with fewer than ``min_members`` measured genes are dropped (a mean over a few
    genes is noise). Coverage (measured / annotated members) is reported so thinly
    covered pathways are visible.
    """
    index = {s: i for i, s in enumerate(symbols)}
    names, rows, sizes, coverage = [], [], [], []

    for name, members in sorted(gene_sets.items()):
        hit = [index[m] for m in members if m in index]
        if len(hit) < min_members:
            continue
        row = np.zeros(len(symbols), dtype=np.float64)
        row[hit] = 1.0
        names.append(name)
        rows.append(row)
        sizes.append(len(hit))
        coverage.append(len(hit) / len(members))

    if not names:
        raise ValueError("No gene set met the minimum measured-member threshold.")

    membership = np.vstack(rows)
    logger.info(
        f"{len(names)} gene sets with >= {min_members} measured genes "
        f"(median size {int(np.median(sizes))}, median coverage {np.median(coverage):.0%})"
    )
    return membership, names, np.asarray(sizes), np.asarray(coverage)


def pathway_scores(matrix, membership, sizes):
    """score[patient, pathway] = mean of the matrix over the pathway's member genes.

    errstate: float32 matmul raises spurious warnings on some BLAS backends for finite
    inputs; the caller checks finiteness.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return (matrix @ membership.T) / sizes[None, :]


def direction_label(value):
    if value is None or pd.isna(value):
        return "n/a"
    if value > 0:
        return "risk-increasing"
    if value < 0:
        return "risk-decreasing"
    return "n/a"


def sign_agreement_by_column(a, b):
    """Fraction of rows where two signed matrices agree, column by column."""
    if a is None or b is None:
        return None
    sign_a = np.sign(a)
    sign_b = np.sign(b)
    valid = (sign_a != 0) & (sign_b != 0)
    denom = valid.sum(axis=0)
    numer = ((sign_a == sign_b) & valid).sum(axis=0)
    return np.divide(
        numer,
        denom,
        out=np.full(a.shape[1], np.nan, dtype=float),
        where=denom > 0,
    )


def score_luad_panel(
    msigdb_dir,
    symbols,
    ig_symbols,
    path_gradient_symbols,
    vanilla_gradient_symbols,
    output_dir,
    min_members=3,
    discovery_summed_abs=None,
):
    """Confirmatory KEGG NSCLC driver panel (nt06266) -- NOT part of discovery.

    Scores the KEGG_NSCLC_PANEL modules off the same per-patient attributions used for the
    Reactome discovery ranking, and writes known_luad_kegg_nsclc_scores.csv (rank of each
    driver module by summed |IG|, its direction, coverage, and its percentile against the
    Reactome discovery distribution). The discovery ranking is untouched. min_members is
    lowered (5) here because these driver modules are small (~9-20 genes).
    """
    try:
        gene_sets, _ = load_collections(msigdb_dir, ("c2.cp.kegg_medicus",))
    except FileNotFoundError:
        logger.warning("kegg_medicus not fetched; skipping the KEGG NSCLC panel.")
        return None

    missing = [name for name in KEGG_NSCLC_PANEL if name not in gene_sets]
    if missing:
        logger.warning(f"KEGG NSCLC panel: {len(missing)} module(s) absent from MSigDB.")

    index = {s: i for i, s in enumerate(symbols)}
    rows = []
    for name, node in KEGG_NSCLC_PANEL.items():
        annotated = gene_sets.get(name)
        if annotated is None:
            continue
        members = [index[m] for m in annotated if m in index]
        row = {
            "kegg_node": node,
            "pathway": name,
            "n_annotated_genes": len(annotated),
            "n_measured_genes": len(members),
            "coverage": len(members) / len(annotated) if annotated else np.nan,
            "scored": len(members) >= min_members,
        }
        if len(members) >= min_members:
            mem = np.zeros(len(symbols), dtype=np.float64)
            mem[members] = 1.0
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                ig_score = (ig_symbols @ mem) / len(members)
                row["summed_abs_ig"] = float(np.abs(ig_score).sum())
                if path_gradient_symbols is not None:
                    pg = float((path_gradient_symbols @ mem).mean() / len(members))
                    row["mean_path_gradient"] = pg
                    row["direction"] = direction_label(pg)
                if vanilla_gradient_symbols is not None:
                    row["mean_local_gradient"] = float(
                        (vanilla_gradient_symbols @ mem).mean() / len(members)
                    )
            if discovery_summed_abs is not None:
                row["percentile_vs_reactome"] = float(
                    np.mean(np.asarray(discovery_summed_abs) < row["summed_abs_ig"])
                )
        rows.append(row)

    if not rows:
        logger.warning("No KEGG NSCLC panel modules found; skipping.")
        return None

    table = pd.DataFrame(rows).sort_values(
        ["scored", "summed_abs_ig"], ascending=[False, False], na_position="last"
    )
    path = os.path.join(output_dir, "known_luad_kegg_nsclc_scores.csv")
    table.to_csv(path, index=False)
    n_scored = int(table["scored"].sum())
    logger.info(
        f"KEGG NSCLC driver panel (nt06266) -> {path} "
        f"({n_scored}/{len(table)} modules scored at min_members={min_members})"
    )
    return table


def add_lung_name_flag(table):
    """Flag pathways MSigDB itself names for lung/NSCLC, read from the gene-set name only.

    Subtype comes from the same name: NON_SMALL_CELL/NSCLC -> NSCLC (LUAD's superclass),
    SMALL_CELL -> SCLC (not LUAD), else "lung". Nothing hand-authored; all pathways are
    still scored. Adds columns lung_named_in_msigdb, matched_lung_name_tokens,
    lung_cancer_subtype_from_name.
    """

    def classify(name):
        upper = str(name).upper()
        tokens = [token for token in LUNG_NAME_TOKENS if token in upper]
        if not tokens:
            return tokens, ""
        # Order matters: NON_SMALL_CELL contains SMALL_CELL as a substring.
        if "NON_SMALL_CELL" in upper or "NSCLC" in upper:
            subtype = "NSCLC"
        elif "SMALL_CELL" in upper:
            subtype = "SCLC"
        else:
            subtype = "lung"
        return tokens, subtype

    classified = [classify(pathway) for pathway in table["pathway"]]
    out = table.reset_index(drop=True).copy()
    out["lung_named_in_msigdb"] = [bool(tok) for tok, _ in classified]
    out["matched_lung_name_tokens"] = ["; ".join(tok) for tok, _ in classified]
    out["lung_cancer_subtype_from_name"] = [subtype for _, subtype in classified]
    return out


# ---------------------------------------------------------------------------
# Layer B: member-gene drill-down
# ---------------------------------------------------------------------------


def member_gene_table(
    top_pathways,
    gene_sets,
    symbols,
    ig_symbols,
    path_gradient_symbols,
    vanilla_gradient_symbols,
    gene_details,
):
    """For each top pathway, its member genes ranked by their own attribution.

    The genes come from MSigDB (whatever it assigns to the pathway), so each gene arrives
    with the pathway that surfaced it -- no curated gene list anywhere.
    """
    index = {s: i for i, s in enumerate(symbols)}
    mean_abs = np.abs(ig_symbols).mean(axis=0)
    mean_signed = ig_symbols.mean(axis=0)
    frac_positive = (ig_symbols > 0).mean(axis=0)
    if path_gradient_symbols is not None:
        mean_path_gradient = path_gradient_symbols.mean(axis=0)
        frac_path_gradient_positive = (path_gradient_symbols > 0).mean(axis=0)
    else:
        mean_path_gradient = np.full(len(symbols), np.nan)
        frac_path_gradient_positive = np.full(len(symbols), np.nan)
    if vanilla_gradient_symbols is not None:
        mean_vanilla_gradient = vanilla_gradient_symbols.mean(axis=0)
        frac_vanilla_gradient_positive = (vanilla_gradient_symbols > 0).mean(axis=0)
    else:
        mean_vanilla_gradient = np.full(len(symbols), np.nan)
        frac_vanilla_gradient_positive = np.full(len(symbols), np.nan)

    gene_gradient_agreement = sign_agreement_by_column(
        path_gradient_symbols, vanilla_gradient_symbols
    )
    if gene_gradient_agreement is None:
        gene_gradient_agreement = np.full(len(symbols), np.nan)

    rows = []
    for pathway in top_pathways:
        members = [index[m] for m in gene_sets[pathway] if m in index]
        total = mean_abs[members].sum()
        for rank, i in enumerate(sorted(members, key=lambda i: -mean_abs[i]), start=1):
            info = gene_details.get(symbols[i], {})
            rows.append(
                {
                    "pathway": pathway,
                    "rank_in_pathway": rank,
                    "symbol": symbols[i],
                    "description": info.get("description"),
                    "mean_abs_ig": mean_abs[i],
                    "share_of_pathway_attribution": mean_abs[i] / total if total else np.nan,
                    # Aliases -> path gradient (the primary direction).
                    "mean_gradient": mean_path_gradient[i],
                    "direction": direction_label(mean_path_gradient[i]),
                    "mean_path_gradient": mean_path_gradient[i],
                    "path_direction": direction_label(mean_path_gradient[i]),
                    "frac_patients_path_gradient_positive": frac_path_gradient_positive[i],
                    "mean_vanilla_gradient": mean_vanilla_gradient[i],
                    "vanilla_direction": direction_label(mean_vanilla_gradient[i]),
                    "frac_patients_vanilla_gradient_positive": (
                        frac_vanilla_gradient_positive[i]
                    ),
                    "path_vanilla_sign_agreement": gene_gradient_agreement[i],
                    "mean_signed_ig_DO_NOT_USE_FOR_DIRECTION": mean_signed[i],
                    "frac_patients_ig_positive": frac_positive[i],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_beeswarm(
    scores,
    names,
    output_path,
    top_n=20,
    xlabel=None,
    title=None,
    color_scores=None,
    color_label=None,
    standardize_color_by_row=False,
    center_color_at_zero=True,
    color_limits=None,
):
    """The paper's Fig. 6 analog: one row per pathway, one dot per patient.

    x = the patient's mean pathway score. By default, colour = the same pathway score,
    matching the Steyaert/SHAP-style convention where red/blue reflects high/low
    gradient value. Pass expression scores explicitly for an expression-coloured
    supplemental view.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(range(min(top_n, len(names))))[::-1]
    rng = np.random.default_rng(0)
    colour_matrix = scores if color_scores is None else color_scores

    if color_limits is None:
        finite_colour = np.asarray(colour_matrix[:, : len(names)], dtype=float)
        finite_colour = finite_colour[np.isfinite(finite_colour)]
        if finite_colour.size:
            if center_color_at_zero:
                vmax = np.nanpercentile(np.abs(finite_colour), 98)
                if not np.isfinite(vmax) or vmax == 0:
                    vmax = np.nanmax(np.abs(finite_colour))
                color_limits = (-vmax, vmax)
            else:
                color_limits = (
                    np.nanpercentile(finite_colour, 2),
                    np.nanpercentile(finite_colour, 98),
                )
        else:
            color_limits = (-1, 1)

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 2))
    dots = None
    for y, j in enumerate(rows):
        values = scores[:, j]
        colour = colour_matrix[:, j]
        finite = np.isfinite(colour)
        if standardize_color_by_row and finite.any() and np.ptp(colour[finite]) > 0:
            colour = (colour - np.nanmean(colour)) / (np.nanstd(colour) + 1e-12)
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        dots = ax.scatter(
            values,
            np.full(len(values), y) + jitter,
            c=colour,
            cmap="coolwarm",
            s=13,
            alpha=0.85,
            linewidths=0,
            vmin=color_limits[0],
            vmax=color_limits[1],
        )

    ax.axvline(0, color="0.3", lw=0.8, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [names[j].replace("HALLMARK_", "").replace("REACTOME_", "")[:52] for j in rows],
        fontsize=8,
    )
    ax.set_xlabel(
        xlabel
        or (
            "Mean integrated-gradient attribution over pathway genes\n"
            "(per patient; positive = pushes THIS patient's predicted risk up)"
        )
    )
    ax.set_title(title or "Top pathways by summed |IG| attribution", fontsize=11)
    bar = fig.colorbar(dots, ax=ax, pad=0.01)
    bar.set_label(color_label or "Gradient value", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Beeswarm -> {output_path}")


def plot_direction(table, output_path, top_n=20):
    """Signed bar chart of the top pathways, using the gradient direction -- not signed
    IG, which cancels across the cohort and would draw near-zero bars regardless of
    importance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = table.head(top_n).iloc[::-1]
    if subset["mean_gradient"].isna().all():
        logger.warning("No gradients available; skipping the direction plot.")
        return

    labels = [
        n.replace("HALLMARK_", "").replace("REACTOME_", "")[:52] for n in subset["pathway"]
    ]
    values = subset["mean_gradient"].to_numpy()
    colours = ["#d7191c" if v > 0 else "#2c7bb6" for v in values]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(subset) + 2))
    ax.barh(labels, values, color=colours)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(
        "Mean path gradient over pathway genes\n"
        "(positive = raises predicted risk / poor prognosis; negative = protective)"
    )
    ax.tick_params(axis="y", labelsize=8)

    ax.set_title("Direction of the top pathways", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Direction plot -> {output_path}")


def plot_gradient_comparison(table, output_path, top_n=20):
    """Paired pathway-level path vs vanilla gradient means."""
    if (
        "mean_path_gradient" not in table
        or "mean_vanilla_gradient" not in table
        or table[["mean_path_gradient", "mean_vanilla_gradient"]].isna().all().all()
    ):
        logger.warning("Path/vanilla gradient columns unavailable; skipping comparison plot.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = table.head(top_n).iloc[::-1]
    labels = [
        n.replace("HALLMARK_", "")
        .replace("REACTOME_", "")
        .replace("KEGG_MEDICUS_", "")
        .replace("KEGG_", "")[:52]
        for n in subset["pathway"]
    ]
    y = np.arange(len(subset))

    fig, ax = plt.subplots(figsize=(9, 0.46 * len(subset) + 2.2))
    ax.axvline(0, color="0.25", lw=0.8, zorder=0)
    ax.scatter(
        subset["mean_path_gradient"],
        y + 0.08,
        label="path-averaged gradient",
        s=24,
        color="#b2182b",
    )
    ax.scatter(
        subset["mean_vanilla_gradient"],
        y - 0.08,
        label="vanilla gradient",
        s=24,
        color="#2166ac",
    )
    for yi, (_, row) in zip(y, subset.iterrows()):
        if pd.notna(row["mean_path_gradient"]) and pd.notna(row["mean_vanilla_gradient"]):
            ax.plot(
                [row["mean_path_gradient"], row["mean_vanilla_gradient"]],
                [yi, yi],
                color="0.75",
                lw=0.8,
                zorder=0,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean gradient over pathway genes (positive = raises predicted risk)")
    ax.set_title("Path-averaged vs vanilla gradient direction for top pathways", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gradient comparison plot -> {output_path}")


def plot_gene_beeswarm(
    top_pathways, gene_sets, symbols, gene_gradient, gene_magnitude, output_path, top_n=20
):
    """One row per pathway, one dot per MEMBER GENE -- the gene-level companion to
    plot_beeswarm (per patient). Shows how each pathway's own genes distribute their
    effect, the detail the pathway mean collapses.
      x      = gene's mean path gradient (signed direction on risk)
      colour = sign (red = raises risk, blue = protective)
      size   = gene's mean |IG| (how much the model leans on it)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured = set(symbols)
    pathways = list(top_pathways[:top_n])
    rows = list(range(len(pathways)))[::-1]
    rng = np.random.default_rng(0)

    all_mags = [
        gene_magnitude[g]
        for p in pathways
        for g in gene_sets[p]
        if g in measured
    ]
    max_mag = max(all_mags) if all_mags else 1.0

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 2))
    for y, j in enumerate(rows):
        members = [
            g for g in gene_sets[pathways[j]]
            if g in measured and np.isfinite(gene_gradient[g])
        ]
        if not members:
            continue
        x = np.array([gene_gradient[g] for g in members])
        dot_sizes = np.array([8 + 60 * (gene_magnitude[g] / max_mag) for g in members])
        colours = ["#d7191c" if v > 0 else "#2c7bb6" for v in x]
        jitter = rng.uniform(-0.18, 0.18, size=len(x))
        ax.scatter(
            x, np.full(len(x), y) + jitter, c=colours, s=dot_sizes,
            alpha=0.7, linewidths=0,
        )

    ax.axvline(0, color="0.3", lw=0.8, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [pathways[j].replace("HALLMARK_", "").replace("REACTOME_", "")[:52] for j in rows],
        fontsize=8,
    )
    ax.set_xlabel(
        "Mean path gradient of each member gene\n"
        "(one dot per gene; red = raises risk, blue = protective; dot size = mean |IG|)"
    )
    ax.set_title("Distribution of member-gene effects within each top pathway", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gene-level beeswarm -> {output_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def cache_gene_axis(gene_names, path=GENE_AXIS_CACHE):
    """Persist the model's ordered gene axis so later runs need no cluster data."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(gene_names) + "\n")
    logger.info(f"Gene axis ({len(gene_names)} IDs) cached -> {path}")


def resolve_gene_axis(mapping_file_path, gene_axis_path):
    """Read the gene axis from the cached list if present, else the mapping JSON."""
    if gene_axis_path and os.path.exists(gene_axis_path):
        with open(gene_axis_path) as fh:
            gene_names = [line.strip() for line in fh if line.strip()]
        logger.info(f"Gene axis from cache: {len(gene_names)} IDs ({gene_axis_path})")
        return gene_names

    gene_names = load_gene_names(mapping_file_path)
    cache_gene_axis(gene_names, gene_axis_path or GENE_AXIS_CACHE)
    return gene_names


ANALYSIS_BUNDLE_NAME = "pathway_analysis_bundle.npz"


def save_analysis_bundle(
    path,
    ig_symbols,
    path_gradient_symbols,
    membership,
    sizes,
    names,
    symbols,
    patient_ids,
    collections_per_pathway,
):
    """Persist everything Layer C needs so pathway_tests.py can reproduce the tests later.

    The tests are pure functions of the symbol-level attribution matrices and the
    membership matrix, so this bundle removes any need for the model, raw IG files, or
    MSigDB. Strings are stored as unicode arrays so it loads without pickle.
    """
    has_path_gradients = path_gradient_symbols is not None
    np.savez_compressed(
        path,
        ig_symbols=np.asarray(ig_symbols, dtype=np.float32),
        path_gradient_symbols=(
            np.asarray(path_gradient_symbols, dtype=np.float32)
            if has_path_gradients
            else np.zeros((0, 0), dtype=np.float32)
        ),
        has_path_gradients=np.array(1 if has_path_gradients else 0, dtype=np.int8),
        membership=np.asarray(membership, dtype=np.int8),
        sizes=np.asarray(sizes, dtype=np.int64),
        names=np.asarray(list(names), dtype=str),
        symbols=np.asarray(list(symbols), dtype=str),
        patient_ids=np.asarray([str(p) for p in patient_ids], dtype=str),
        collections=np.asarray(list(collections_per_pathway), dtype=str),
    )


def run(
    mapping_file_path,
    ig_directory,
    output_dir,
    gene_axis_path=GENE_AXIS_CACHE,
    msigdb_dir=MSIGDB_DIR,
    collections=DEFAULT_COLLECTIONS,
    min_members=10,
    panel_min_members=3,
    top_n=20,
    gene_info_cache_dir="assets/gene_info",
    require_gradients=True,
    ranking_statistic="vanilla_gradient_abs",
    write_figures=True,
    write_member_genes=True,
    write_bundle=True,
):
    os.makedirs(output_dir, exist_ok=True)

    gene_names = resolve_gene_axis(mapping_file_path, gene_axis_path)
    ig, path_gradients, vanilla_gradients, expression, patient_ids = load_attributions(
        ig_directory, len(gene_names), require_gradients
    )

    gene_sets, collection_of = load_collections(msigdb_dir, collections)
    chip, chip_name = load_chip(msigdb_dir)

    collapsed, symbols, axis_stats = collapse_to_symbols(
        {
            "ig": ig,
            "path_gradients": path_gradients,
            "vanilla_gradients": vanilla_gradients,
            "expression": expression,
        },
        gene_names,
        chip,
    )
    ig_symbols = collapsed["ig"]
    path_gradient_symbols = collapsed["path_gradients"]
    vanilla_gradient_symbols = collapsed["vanilla_gradients"]
    expression_symbols = collapsed["expression"]

    membership, names, sizes, coverage = build_membership(gene_sets, symbols, min_members)

    # --- Layer A: the paper's pathway score ---------------------------------------
    scores = pathway_scores(ig_symbols, membership, sizes)
    summed_abs_ig = np.abs(scores).sum(axis=0)

    if path_gradient_symbols is not None:
        path_gradient_scores = pathway_scores(path_gradient_symbols, membership, sizes)
        mean_path_gradient = path_gradient_scores.mean(axis=0)
        summed_abs_path_gradient = np.abs(path_gradient_scores).sum(axis=0)
        mean_abs_path_gradient = np.abs(path_gradient_scores).mean(axis=0)
    else:
        path_gradient_scores = None
        mean_path_gradient = np.full(len(names), np.nan)
        summed_abs_path_gradient = np.full(len(names), np.nan)
        mean_abs_path_gradient = np.full(len(names), np.nan)

    if vanilla_gradient_symbols is not None:
        vanilla_gradient_scores = pathway_scores(vanilla_gradient_symbols, membership, sizes)
        mean_vanilla_gradient = vanilla_gradient_scores.mean(axis=0)
        summed_abs_vanilla_gradient = np.abs(vanilla_gradient_scores).sum(axis=0)
        mean_abs_vanilla_gradient = np.abs(vanilla_gradient_scores).mean(axis=0)
    else:
        vanilla_gradient_scores = None
        mean_vanilla_gradient = np.full(len(names), np.nan)
        summed_abs_vanilla_gradient = np.full(len(names), np.nan)
        mean_abs_vanilla_gradient = np.full(len(names), np.nan)

    pathway_gradient_agreement = sign_agreement_by_column(
        path_gradient_scores, vanilla_gradient_scores
    )
    if pathway_gradient_agreement is None:
        pathway_gradient_agreement = np.full(len(names), np.nan)

    table = pd.DataFrame(
        {
            "pathway": names,
            "collection": [collection_of[n] for n in names],
            "n_measured_genes": sizes,
            "coverage": coverage,
            # Baseline-aware attribution magnitude; exported as a separate IG table.
            "summed_abs_ig": summed_abs_ig,
            "mean_abs_ig": np.abs(scores).mean(axis=0),
            # Primary Steyaert-style ranking statistic.
            "summed_abs_vanilla_gradient": summed_abs_vanilla_gradient,
            "mean_abs_vanilla_gradient": mean_abs_vanilla_gradient,
            "summed_abs_path_gradient": summed_abs_path_gradient,
            "mean_abs_path_gradient": mean_abs_path_gradient,
            "mean_path_gradient": mean_path_gradient,
            "path_direction": [direction_label(g) for g in mean_path_gradient],
            "mean_vanilla_gradient": mean_vanilla_gradient,
            "vanilla_direction": [direction_label(g) for g in mean_vanilla_gradient],
            "path_vanilla_sign_agreement": pathway_gradient_agreement,
            "mean_signed_ig_DO_NOT_USE_FOR_DIRECTION": scores.mean(axis=0),
        }
    )
    table = add_lung_name_flag(table)

    if ranking_statistic not in RANKING_COLUMNS:
        raise ValueError(
            f"Unknown ranking_statistic={ranking_statistic!r}. "
            f"Choose one of: {', '.join(RANKING_COLUMNS)}"
        )
    ranking_column, ranking_description = RANKING_COLUMNS[ranking_statistic]
    if table[ranking_column].isna().all():
        fallback_column, fallback_description = RANKING_COLUMNS["ig_abs"]
        logger.warning(
            f"Requested ranking column {ranking_column} is unavailable; falling back "
            f"to {fallback_column}."
        )
        ranking_statistic = "ig_abs"
        ranking_column = fallback_column
        ranking_description = fallback_description

    if ranking_statistic == "vanilla_gradient_abs":
        direction_column = "mean_vanilla_gradient"
        direction_source = "vanilla_gradient"
    elif ranking_statistic == "path_gradient_abs":
        direction_column = "mean_path_gradient"
        direction_source = "path_gradient"
    elif not table["mean_path_gradient"].isna().all():
        direction_column = "mean_path_gradient"
        direction_source = "path_gradient"
    else:
        direction_column = "mean_vanilla_gradient"
        direction_source = "vanilla_gradient"

    table["mean_gradient"] = table[direction_column]
    table["direction"] = [direction_label(g) for g in table[direction_column]]
    table["direction_source"] = direction_source

    table = table.sort_values(ranking_column, ascending=False, ignore_index=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    table.insert(1, "ranking_statistic", ranking_statistic)
    logger.info(
        f"Pathways ranked by {ranking_column} ({ranking_description}); median size of "
        f"the top 20 = {int(table.head(20)['n_measured_genes'].median())} genes"
    )
    table.to_csv(os.path.join(output_dir, "pathway_scores.csv"), index=False)
    logger.info(f"Pathway scores -> {output_dir}/pathway_scores.csv")
    lung_table = table[table["lung_named_in_msigdb"]].copy()
    lung_table.to_csv(
        os.path.join(output_dir, "lung_named_pathway_scores.csv"), index=False
    )
    subtype_counts = lung_table["lung_cancer_subtype_from_name"].value_counts().to_dict()
    logger.info(
        f"Lung/NSCLC-named MSigDB subset -> "
        f"{output_dir}/lung_named_pathway_scores.csv "
        f"({len(lung_table)} lung-named pathways; subtypes from name: {subtype_counts}. "
        f"SCLC is not LUAD -- filter it out with lung_cancer_subtype_from_name != 'SCLC')"
    )

    ig_table = table.sort_values("summed_abs_ig", ascending=False, ignore_index=True).copy()
    ig_table.insert(0, "ig_magnitude_rank", range(1, len(ig_table) + 1))
    ig_table.to_csv(
        os.path.join(output_dir, "pathway_ig_magnitude_scores.csv"), index=False
    )
    logger.info(
        f"IG magnitude-ranked pathway scores -> "
        f"{output_dir}/pathway_ig_magnitude_scores.csv"
    )

    # Confirmatory KEGG NSCLC driver panel (nt06266) -- separate from the Reactome
    # discovery ranking; reports how the model behaved for the established NSCLC drivers.
    score_luad_panel(
        msigdb_dir,
        symbols,
        ig_symbols,
        path_gradient_symbols,
        vanilla_gradient_symbols,
        output_dir,
        min_members=panel_min_members,
        discovery_summed_abs=table["summed_abs_ig"].to_numpy(),
    )

    top_pathways = table["pathway"].head(top_n).tolist()
    order = [names.index(p) for p in top_pathways]
    ig_top_pathways = ig_table["pathway"].head(top_n).tolist()
    ig_order = [names.index(p) for p in ig_top_pathways]
    pd.DataFrame(scores[:, order], index=patient_ids, columns=top_pathways).to_csv(
        os.path.join(output_dir, "pathway_scores_matrix.csv")
    )
    pd.DataFrame(scores[:, ig_order], index=patient_ids, columns=ig_top_pathways).to_csv(
        os.path.join(output_dir, "pathway_ig_magnitude_scores_matrix.csv")
    )
    if path_gradient_scores is not None:
        pd.DataFrame(
            path_gradient_scores[:, order], index=patient_ids, columns=top_pathways
        ).to_csv(os.path.join(output_dir, "pathway_path_gradient_scores_matrix.csv"))
    if vanilla_gradient_scores is not None:
        pd.DataFrame(
            vanilla_gradient_scores[:, order], index=patient_ids, columns=top_pathways
        ).to_csv(os.path.join(output_dir, "pathway_vanilla_gradient_scores_matrix.csv"))

    # --- Layer B: member genes of the top pathways --------------------------------
    if write_member_genes:
        top_symbols = sorted(
            {s for p in top_pathways for s in gene_sets[p] if s in set(symbols)}
        )
        ensembl_for_symbol = {}
        for eid in gene_names:
            symbol = chip.get(clean_ensembl_id(eid))
            if symbol in set(top_symbols) and symbol not in ensembl_for_symbol:
                ensembl_for_symbol[symbol] = eid

        details_by_ensembl, _, _ = annotate_genes(
            list(ensembl_for_symbol.values()), gene_info_cache_dir
        )
        gene_details = {
            symbol: details_by_ensembl.get(eid, {})
            for symbol, eid in ensembl_for_symbol.items()
        }

        members = member_gene_table(
            top_pathways,
            gene_sets,
            symbols,
            ig_symbols,
            path_gradient_symbols,
            vanilla_gradient_symbols,
            gene_details,
        )
        members.to_csv(os.path.join(output_dir, "pathway_member_genes.csv"), index=False)
        logger.info(f"Member genes -> {output_dir}/pathway_member_genes.csv")

    # --- Save the bundle for the optional pathway_tests.py evidence step -----------
    if write_bundle:
        bundle_path = os.path.join(output_dir, ANALYSIS_BUNDLE_NAME)
        save_analysis_bundle(
            bundle_path,
            ig_symbols,
            path_gradient_symbols,
            membership,
            sizes,
            names,
            symbols,
            patient_ids,
            [collection_of[n] for n in names],
        )
        logger.info(f"Analysis bundle (for pathway_tests.py) -> {bundle_path}")

    # --- Figures -------------------------------------------------------------------
    if write_figures:
        expression_top = (
            pathway_scores(expression_symbols, membership, sizes)[:, order]
            if expression_symbols is not None
            else None
        )

        if vanilla_gradient_scores is not None:
            vanilla_top = vanilla_gradient_scores[:, order]
            for filename in (
                "pathway_steyaert_vanilla_gradient_beeswarm.png",
                "pathway_vanilla_gradient_beeswarm.png",
            ):
                plot_beeswarm(
                    vanilla_top,
                    top_pathways,
                    os.path.join(output_dir, filename),
                    top_n,
                    xlabel=(
                        "Mean vanilla gradient over normalized pathway gene inputs\n"
                        "(per patient endpoint; positive = higher predicted risk)"
                    ),
                    title="Steyaert-style pathway gradients",
                    color_scores=vanilla_top,
                    color_label="Gradient value",
                )

        if path_gradient_scores is not None:
            path_top = path_gradient_scores[:, order]
            for filename in (
                "pathway_steyaert_path_gradient_beeswarm.png",
                "pathway_path_gradient_beeswarm.png",
            ):
                plot_beeswarm(
                    path_top,
                    top_pathways,
                    os.path.join(output_dir, filename),
                    top_n,
                    xlabel=(
                        "Mean path-averaged gradient over normalized pathway gene inputs\n"
                        "(per patient; positive = higher predicted risk)"
                    ),
                    title="Pathway sensitivity by path-averaged gradients",
                    color_scores=path_top,
                    color_label="Gradient value",
                )

        plot_beeswarm(
            scores[:, order],
            top_pathways,
            os.path.join(output_dir, "pathway_beeswarm.png"),
            top_n,
            xlabel=(
                "Mean integrated-gradient attribution over pathway genes\n"
                "(per patient; positive = pushes this patient's predicted risk up)"
            ),
            title="Baseline-aware pathway attribution for Steyaert-ranked pathways",
            color_scores=scores[:, order],
            color_label="IG attribution value",
        )
        plot_beeswarm(
            scores[:, ig_order],
            ig_top_pathways,
            os.path.join(output_dir, "pathway_ig_magnitude_beeswarm.png"),
            top_n,
            xlabel=(
                "Mean integrated-gradient attribution over pathway genes\n"
                "(per patient; positive = pushes this patient's predicted risk up)"
            ),
            title="Top pathways by summed |IG| attribution",
            color_scores=scores[:, ig_order],
            color_label="IG attribution value",
        )

        if expression_top is not None:
            plot_beeswarm(
                scores[:, order],
                top_pathways,
                os.path.join(output_dir, "pathway_expression_colored_ig_beeswarm.png"),
                top_n,
                xlabel=(
                    "Mean integrated-gradient attribution over pathway genes\n"
                    "(per patient; positive = pushes this patient's predicted risk up)"
                ),
                title="Expression-coloured IG attribution for Steyaert-ranked pathways",
                color_scores=expression_top,
                color_label="Pathway mean expression (z)",
                standardize_color_by_row=True,
                center_color_at_zero=True,
                color_limits=(-2, 2),
            )

        # The two requested headline beeswarms: x = per-patient IG attribution (which is
        # baseline-relative and carries the spread), coloured by the gradient DIRECTION --
        # local (endpoint) gradient on one, path-averaged gradient on the other. The
        # gradient is near-constant per patient, so each row's colour reads as that
        # pathway's true risk direction (red = raises risk, blue = protective), which the
        # baseline-relative IG x-position cannot show on its own.
        ig_x_label = (
            "Mean integrated-gradient attribution over pathway genes\n"
            "(per patient, relative to the training-mean baseline; "
            "positive = pushes this patient's predicted risk up)"
        )
        if vanilla_gradient_scores is not None:
            plot_beeswarm(
                scores[:, order],
                top_pathways,
                os.path.join(output_dir, "pathway_ig_beeswarm_local_gradient_color.png"),
                top_n,
                xlabel=ig_x_label,
                title="IG attribution (x) coloured by local gradient (direction)",
                color_scores=vanilla_gradient_scores[:, order],
                color_label="Local gradient (direction: red = raises risk)",
            )
        if path_gradient_scores is not None:
            plot_beeswarm(
                scores[:, order],
                top_pathways,
                os.path.join(output_dir, "pathway_ig_beeswarm_path_gradient_color.png"),
                top_n,
                xlabel=ig_x_label,
                title="IG attribution (x) coloured by path-integrated gradient (direction)",
                color_scores=path_gradient_scores[:, order],
                color_label="Path-integrated gradient (direction: red = raises risk)",
            )

        plot_direction(table, os.path.join(output_dir, "pathway_direction.png"), top_n)
        plot_gradient_comparison(
            table, os.path.join(output_dir, "pathway_gradient_comparison.png"), top_n
        )
        if write_member_genes and path_gradient_symbols is not None:
            # Gene-level companion: one dot per member gene (see plot_gene_beeswarm).
            gene_gradient = dict(zip(symbols, path_gradient_symbols.mean(axis=0)))
            gene_magnitude = dict(zip(symbols, np.abs(ig_symbols).mean(axis=0)))
            plot_gene_beeswarm(
                top_pathways,
                gene_sets,
                symbols,
                gene_gradient,
                gene_magnitude,
                os.path.join(output_dir, "pathway_gene_beeswarm.png"),
                top_n,
            )

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_patients": len(patient_ids),
        "analysis_bundle": ANALYSIS_BUNDLE_NAME if write_bundle else None,
        "attribution": {
            "primary_method": "vanilla gradients",
            "comparison_method": "path-averaged gradients",
            "ig_method": "integrated gradients",
            "baseline": "mean RNA-seq expression over the training split",
            "direction_statistic": f"cohort-mean {direction_source.replace('_', ' ')}",
            "direction_source": direction_source,
            "local_comparison_statistic": "cohort-mean vanilla gradient",
            "ranking_statistic": ranking_statistic,
            "ranking_column": ranking_column,
            "ranking_description": ranking_description,
            "ig_magnitude_output": "pathway_ig_magnitude_scores.csv",
            "primary_figure": "pathway_steyaert_vanilla_gradient_beeswarm.png",
            "path_gradient_companion_figure": "pathway_steyaert_path_gradient_beeswarm.png",
            "note": (
                "The primary pathway plot follows Steyaert et al. by using local "
                "endpoint gradients and Reactome pathways. IG is exported separately "
                "as a baseline-aware attribution magnitude layer. The sign of a "
                "cohort-mean-baseline IG is patient-relative and should not be used "
                "as cohort-level direction. See literature/ig_pathway_design_notes.md."
            ),
        },
        "gene_axis": axis_stats,
        "msigdb": {
            "collections": list(collections),
            "chip": chip_name,
            "version_file": os.path.join(msigdb_dir, "VERSION"),
            "n_gene_sets_scored": len(names),
            "min_measured_members": min_members,
        },
        "lung_name_flag": {
            "method": (
                "database-only: a pathway is flagged if its MSigDB gene-set NAME "
                "contains a lung/NSCLC token. No hand-authored biology or "
                "keyword-to-pathway curation is used."
            ),
            "tokens": list(LUNG_NAME_TOKENS),
            "used_for_ranking_or_testing": False,
            "output": "lung_named_pathway_scores.csv",
            "note": (
                "The flag identifies the MSigDB sets the database itself names for "
                "lung/NSCLC when those sets are present in the selected collections. "
                "All pathways are still scored against the selected universe; the "
                "flag does not choose, rank, or test anything."
            ),
        },
        "top_pathways": top_pathways,
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    log_summary(table, top_n, ranking_column, ranking_description)
    return table


def log_summary(table, top_n, ranking_column, ranking_description):
    logger.info("\n" + "=" * 92)
    logger.info(f"TOP {top_n} PATHWAYS BY {ranking_column} ({ranking_description})")
    logger.info("=" * 92)
    logger.info(f"{'Rank':<5}{'Pathway':<58}{'RankVal':<10}{'Dir':<6}Genes")
    for _, row in table.head(top_n).iterrows():
        name = row["pathway"].replace("HALLMARK_", "").replace("REACTOME_", "")[:56]
        arrow = "up" if row["mean_gradient"] > 0 else "down"
        logger.info(
            f"{row['rank']:<5}{name:<58}{row[ranking_column]:<10.4f}{arrow:<6}"
            f"{int(row['n_measured_genes'])}"
        )


def main(config, opt):
    base_dir = Path(config.testing.output_base_dir)
    primary_collections = tuple(c.strip() for c in opt.collections.split(",") if c.strip())
    supplemental_collections = tuple(
        c.strip() for c in opt.supplemental_collections.split(",") if c.strip()
    )
    run(
        mapping_file_path=opt.mapping_file or config.data.json_file,
        ig_directory=opt.ig_dir or str(base_dir / "IG_6sep"),
        output_dir=str(base_dir / "pathway_interpret"),
        gene_axis_path=opt.gene_axis,
        msigdb_dir=opt.msigdb_dir,
        collections=primary_collections,
        min_members=opt.min_members,
        panel_min_members=opt.panel_min_members,
        top_n=opt.top_n,
        gene_info_cache_dir=opt.gene_info_dir,
        require_gradients=not opt.allow_missing_gradients,
        ranking_statistic=opt.ranking_statistic,
        write_figures=True,
        write_member_genes=True,
        write_bundle=True,
    )
    if not opt.no_supplemental_all and set(supplemental_collections) != set(primary_collections):
        run(
            mapping_file_path=opt.mapping_file or config.data.json_file,
            ig_directory=opt.ig_dir or str(base_dir / "IG_6sep"),
            output_dir=str(base_dir / "pathway_interpret_supplemental_all"),
            gene_axis_path=opt.gene_axis,
            msigdb_dir=opt.msigdb_dir,
            collections=supplemental_collections,
            min_members=opt.min_members,
            panel_min_members=opt.panel_min_members,
            top_n=opt.top_n,
            gene_info_cache_dir=opt.gene_info_dir,
            require_gradients=not opt.allow_missing_gradients,
            ranking_statistic=opt.ranking_statistic,
            write_figures=False,
            write_member_genes=False,
            write_bundle=False,
        )
    logger.info(
        "\nDone. Scores + figures + pathway_analysis_bundle.npz written. For "
        "p-values / GSEA / ORA, run joint_fusion.testing.pathway_tests on the bundle."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pathway-level IG interpretability.")
    parser.add_argument(
        "--config",
        type=str,
        default="joint_fusion/config/config_checkpoint_2026-04-07-04-58-17_fold1.yaml",
        help="Defaults to the fold-1 config; pass another to use a different fold.",
    )
    parser.add_argument("--ig-dir", type=str, default=None,
                        help="Defaults to <output_base_dir>/IG_6sep.")
    parser.add_argument("--mapping-file", type=str, default=None,
                        help="Mapping JSON. Defaults to config.data.json_file.")
    parser.add_argument("--gene-axis", type=str, default=GENE_AXIS_CACHE,
                        help="Cached ordered gene axis; written on first run.")
    parser.add_argument("--msigdb-dir", type=str, default=MSIGDB_DIR)
    parser.add_argument(
        "--collections",
        type=str,
        default=",".join(DEFAULT_COLLECTIONS),
        help=(
            "Primary pathway collections. Defaults to Reactome only, matching "
            "Steyaert et al.'s pathway-gradient figure."
        ),
    )
    parser.add_argument(
        "--supplemental-collections",
        type=str,
        default=",".join(ALL_COLLECTIONS),
        help=(
            "Collections for the lightweight supplemental sensitivity table. "
            "Defaults to Hallmark + Reactome + KEGG + C6."
        ),
    )
    parser.add_argument(
        "--no-supplemental-all",
        action="store_true",
        help="Skip the lightweight all-collections sensitivity output directory.",
    )
    parser.add_argument(
        "--ranking-statistic",
        choices=sorted(RANKING_COLUMNS),
        default="vanilla_gradient_abs",
        help=(
            "Primary pathway ranking. Default is summed absolute vanilla gradients "
            "to reproduce Steyaert-style local-gradient pathway ranking."
        ),
    )
    parser.add_argument("--min-members", type=int, default=10,
                        help="Drop discovery gene sets with fewer measured genes than this.")
    parser.add_argument("--panel-min-members", type=int, default=3,
                        help="Min measured genes to SCORE a KEGG NSCLC driver module "
                             "(lower than discovery, since driver modules are small; "
                             "modules below this still appear in the panel CSV, flagged "
                             "scored=False).")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Pathways to plot and to drill into.")
    parser.add_argument("--gene-info-dir", type=str, default="assets/gene_info")
    parser.add_argument("--allow-missing-gradients", action="store_true",
                        help="Run without path_gradients_*.npy. Direction columns become "
                             "NaN and GSEA is skipped -- direction is NOT recoverable "
                             "from signed IG under a cohort-mean baseline.")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    config = ConfigManager.load_config(opt.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(
        log_dir=config.testing.output_base_dir,
        log_level=config.logging.log_level,
        log_name=f"pathway_interpret_{timestamp}.log",
    )

    main(config, opt)
