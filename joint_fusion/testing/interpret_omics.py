"""
interpret_omics.py

Rank genes by their Integrated Gradients (IG) attribution across patients using
two complementary rankings, then annotate the top genes from their Ensembl IDs
and assess how related they are to LUAD (lung adenocarcinoma):

  - SIGNED: mean signed IG identifies features associated with increased
    (most positive) or decreased (most negative) predicted risk.
  - MAGNITUDE: mean absolute IG quantifies attribution magnitude independent of
    direction.

The deliverable is a table (CSV + JSON) of the top genes with:
  - symbol / description / biotype (from the gene-info cache, gget.info for misses)
  - mean & std signed IG, mean & std absolute IG, and the number of patients
  - a LUAD driver-gene flag
  - the LUAD-related pathways each gene appears in (from Enrichr)

Outputs (under <output_base_dir>/interpret_omics/):
  - top_genes.csv               one row per top gene (both directions)
  - top_genes.json              same content, nested + LUAD pathway context
  - pathway_enrichment.csv      Enrichr terms for the top gene set, LUAD-flagged

Run from the repo root:
  python -m joint_fusion.testing.interpret_omics --config=<config.yaml> --top-n=20
"""

import argparse
import glob
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import gget
import numpy as np
import pandas as pd

from joint_fusion.config.config_manager import ConfigManager
from joint_fusion.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LUAD reference sets
# ---------------------------------------------------------------------------


# LUAD driver / recurrently altered genes.
#
# Main justification:
# 1) TCGA LUAD landmark paper:
#    The Cancer Genome Atlas Research Network. "Comprehensive molecular profiling of lung adenocarcinoma."
#    Nature 511, 543–550 (2014).
#    https://www.nature.com/articles/nature13385
#
#    This paper identifies recurrent/significantly altered LUAD genes including:
#    TP53, KRAS, EGFR, STK11, KEAP1, NF1, BRAF, PIK3CA, MET, RB1, SMARCA4, CDKN2A,
#    and discusses LUAD oncogenic alterations involving ALK, ROS1, RET, ERBB2, MET, etc.
#
# 2) NCCN NSCLC Guidelines / biomarker framework:
#    NCCN NSCLC guidelines recommend molecular biomarker testing and targeted therapy selection
#    for actionable NSCLC/LUAD alterations such as EGFR, ALK, ROS1, BRAF, MET, RET, ERBB2/HER2,
#    KRAS, etc.
#    https://jnccn.org/view/journals/jnccn/24/4/article-e260017.xml
#
# 3) OncoKB precision oncology knowledge base:
#    Clinically actionable NSCLC oncogenes include EGFR, ALK, ROS1, RET, MET, BRAF, ERBB2/HER2,
#    KRAS, and others.
#    https://www.oncokb.org/
#
# Notes:
# - NKX2-1 is also known as TTF-1. Use "NKX2-1" as the gene symbol; "TTF1" is the protein/pathology alias.
# - MYC is less LUAD-specific than EGFR/KRAS/ALK/etc., but MYC amplification/pathway activation is recurrent
#   in LUAD and is discussed in the TCGA LUAD paper.
# - TP53, STK11, KEAP1, NF1, RB1, SMARCA4, and CDKN2A are mostly tumor suppressor/co-driver genes,
#   not targetable oncogene drivers in the same sense as EGFR/ALK/ROS1/RET/MET/BRAF/ERBB2.
LUAD_RELATED_GENES = {
    "EGFR",  # Canonical actionable LUAD oncogene; recurrent EGFR mutations in TCGA LUAD and NCCN/OncoKB actionable NSCLC biomarker.
    "KRAS",  # Canonical LUAD oncogene; TCGA reports KRAS as one of the most frequent LUAD mutations; KRAS G12C is actionable in NSCLC.
    "TP53",  # Major tumor suppressor driver/co-driver; highly recurrently mutated in TCGA LUAD.
    "STK11",  # LKB1 tumor suppressor; recurrent LUAD alteration in TCGA; major KRAS-associated LUAD co-driver.
    "KEAP1",  # Recurrent LUAD tumor suppressor/co-driver affecting oxidative-stress/NRF2 biology; identified in TCGA LUAD.
    "BRAF",  # Recurrent LUAD oncogene; actionable alteration in NSCLC, especially BRAF V600E.
    "MET",  # Actionable LUAD/NSCLC driver; MET exon 14 skipping and amplification discussed in TCGA and NCCN/OncoKB.
    "ALK",  # Canonical LUAD fusion driver; actionable NSCLC biomarker.
    "ROS1",  # Canonical LUAD fusion driver; actionable NSCLC biomarker.
    "RET",  # Canonical LUAD fusion driver; actionable NSCLC biomarker.
    "ERBB2",  # HER2; recurrent/actionable LUAD alteration, including mutation/amplification.
    "PIK3CA",  # Recurrent PI3K-pathway alteration in TCGA LUAD; less LUAD-specific but driver-relevant.
    "NF1",  # Recurrent tumor suppressor alteration in TCGA LUAD; RAS pathway negative regulator.
    "RB1",  # Tumor suppressor driver/co-driver; recurrently altered in TCGA LUAD.
    "SMARCA4",  # Chromatin-remodeling tumor suppressor; recurrently altered in LUAD.
    "NKX2-1",  # TTF-1 lineage-survival oncogene/lineage factor; recurrent LUAD amplification/lineage dependency.
    "CDKN2A",  # Tumor suppressor/cell-cycle regulator; recurrently altered in TCGA LUAD.
    "MYC",  # Recurrent amplification/pathway activation in LUAD; driver-like but not LUAD-specific.
}

# Substrings (lower-cased) used to flag an Enrichr term as plausibly LUAD-related.
LUAD_TERM_KEYWORDS = [
    "lung",
    "luad",
    "nsclc",
    "non-small cell",
    "adenocarcinoma",
    "alveolar",
    "pulmonary",
    "surfactant",
    "egfr",
    "kras",
    "erbb",
    "pi3k",
    "mapk",
    "ras ",
    "p53",
    "cell cycle",
    "wnt",
]

# Enrichr gene-set libraries queried for the top genes. KEGG/Hallmark/Reactome
# cover canonical pathways; the disease/cell-type libraries surface LUAD- and
# lung-specific signatures directly.
DEFAULT_ENRICHR_LIBRARIES = [
    "KEGG_2021_Human",
    "MSigDB_Hallmark_2020",
    "Reactome_2022",
    "GO_Biological_Process_2021",
    "PanglaoDB_Augmented_2021",
    "Human_Gene_Atlas",
]

# On-disk per-gene info cache: one <clean_ensembl_id>.csv per gene, the same
# layout get_gene_info.py writes to assets/gene_info.
GENE_INFO_CACHE_DIR = "assets/gene_info"


def _clean_ensembl_id(ensembl_id):
    """Strip the Ensembl version suffix (ENSG000....17 -> ENSG000...)."""
    return str(ensembl_id).split(".")[0]


# ---------------------------------------------------------------------------
# Load IG attributions + gene names
# ---------------------------------------------------------------------------


def load_gene_names(mapping_file_path):
    """Return the ordered list of gene Ensembl IDs from the mapping JSON.

    Validates that every sample has the same RNA-seq gene order, because IG
    vectors are interpreted positionally.
    """
    logger.info(f"Loading gene names from {mapping_file_path}")
    mapping_df = pd.read_json(mapping_file_path, orient="index")

    first_sample_id = mapping_df.index[0]
    gene_names = list(mapping_df.loc[first_sample_id, "rnaseq_data"].keys())

    for sample_id in mapping_df.index:
        observed = list(mapping_df.loc[sample_id, "rnaseq_data"].keys())
        if observed != gene_names:
            raise ValueError(
                f"Gene order mismatch for sample {sample_id}. "
                "Cannot safely align IG vector positions to gene IDs."
            )

    logger.info(
        f"Found {len(gene_names)} genes with validated order across "
        f"{len(mapping_df)} samples"
    )
    return gene_names


def load_ig_matrix(ig_directory, n_genes):
    """Load every integrated_grads_<TCGA>.npy file into a (patients x genes) matrix.

    Returns:
        (ig_matrix, patient_ids). Files whose length does not match ``n_genes``
        are skipped with a warning.
    """
    ig_files = sorted(glob.glob(os.path.join(ig_directory, "integrated_grads_*.npy")))
    if not ig_files:
        raise FileNotFoundError(f"No integrated_grads_*.npy files in {ig_directory}")

    rows, patient_ids = [], []
    for file_path in ig_files:
        tcga_id = (
            os.path.basename(file_path)
            .replace("integrated_grads_", "")
            .replace(".npy", "")
        )
        ig_values = np.asarray(np.load(file_path)).flatten()

        if len(ig_values) != n_genes:
            logger.warning(
                f"Skipping {tcga_id}: {len(ig_values)} IG values != {n_genes} genes"
            )
            continue

        rows.append(ig_values)
        patient_ids.append(tcga_id)

    if not rows:
        raise ValueError(f"No IG file matched the expected gene count ({n_genes}).")

    logger.info(f"Loaded IG for {len(patient_ids)} patients x {n_genes} genes")
    return np.vstack(rows), patient_ids


# ---------------------------------------------------------------------------
# Rank genes by mean IG
# ---------------------------------------------------------------------------


def compute_gene_stats(ig_matrix, gene_names):
    """Per-gene attribution statistics across patients.

    Carries both the SIGNED mean IG (direction: positive -> increased predicted
    risk, negative -> decreased) and the mean ABSOLUTE IG (attribution magnitude
    independent of direction). ``std_*`` are the matching across-patient spreads.
    Medians add an outlier-robust view, and ``frac_positive``/``frac_negative``
    show how consistent the attribution sign is across patients.
    """
    abs_matrix = np.abs(ig_matrix)
    return pd.DataFrame(
        {
            "ensembl_id": gene_names,
            "mean_ig": ig_matrix.mean(axis=0),
            "std_ig": ig_matrix.std(axis=0),
            "mean_abs_ig": abs_matrix.mean(axis=0),
            "std_abs_ig": abs_matrix.std(axis=0),
            "median_ig": np.median(ig_matrix, axis=0),
            "median_abs_ig": np.median(abs_matrix, axis=0),
            "frac_positive": (ig_matrix > 0).mean(axis=0),
            "frac_negative": (ig_matrix < 0).mean(axis=0),
            "n_patients": ig_matrix.shape[0],
        }
    )


def rank_top_genes(stats, top_n):
    """Rank genes by SIGNED mean IG and return the top / bottom ``top_n``.

    Identifies features associated with increased (most positive) or decreased
    (most negative) predicted risk.

    Returns:
        (top_positive, top_negative) DataFrames, each with columns
        [rank, direction, ensembl_id, mean_ig, std_ig, mean_abs_ig, std_abs_ig,
        n_patients], ordered from strongest attribution outward.
    """
    signed = stats.sort_values("mean_ig", ascending=False, ignore_index=True)

    top_positive = signed.head(top_n).copy()
    top_positive.insert(0, "direction", "positive")
    top_positive.insert(0, "rank", range(1, len(top_positive) + 1))

    # Most negative first.
    top_negative = signed.tail(top_n).iloc[::-1].reset_index(drop=True)
    top_negative.insert(0, "direction", "negative")
    top_negative.insert(0, "rank", range(1, len(top_negative) + 1))

    return top_positive, top_negative


def rank_top_magnitude_genes(stats, top_n):
    """Rank genes by mean ABSOLUTE IG to quantify attribution magnitude
    independent of direction, and return the top ``top_n``.

    Returns:
        A DataFrame with the same columns as ``rank_top_genes`` output but with
        ``direction == "magnitude"``, ordered from largest magnitude outward.
    """
    magnitude = stats.sort_values("mean_abs_ig", ascending=False, ignore_index=True)

    top_magnitude = magnitude.head(top_n).copy()
    top_magnitude.insert(0, "direction", "magnitude")
    top_magnitude.insert(0, "rank", range(1, len(top_magnitude) + 1))

    return top_magnitude


# ---------------------------------------------------------------------------
# Annotate genes from Ensembl IDs (cache first, gget.info for misses)
# ---------------------------------------------------------------------------

# Column preference order within a cached gene CSV.
_SYMBOL_COLS = ("primary_gene_name", "ensembl_gene_name", "gene_name", "symbol")
_DESC_COLS = (
    "ensembl_description",
    "ncbi_description",
    "uniprot_description",
    "description",
)


def _extract_field(row, candidate_cols):
    for col in candidate_cols:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col].strip()
    return None


def load_or_fetch_gene_csv(ensembl_id, cache_dir=GENE_INFO_CACHE_DIR):
    """Return the full gene-info row (as a 1-row DataFrame) for one gene.

    Checks ``cache_dir/<clean_id>.csv`` first; on a miss, fetches the gene with
    ``gget.info`` and writes ``<clean_id>.csv`` exactly like get_gene_info.py
    (raw gget.info frame, ``index=False``) so both share one cache format.
    Returns None if the gene cannot be resolved.
    """
    clean = _clean_ensembl_id(ensembl_id)
    fp = os.path.join(cache_dir, f"{clean}.csv")

    if os.path.exists(fp):
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                return df
        except Exception:
            pass  # unreadable -> refetch below

    # Miss: fetch a single gene and cache it, mirroring get_gene_info.py.
    try:
        info = gget.info(clean)
    except Exception as e:
        logger.exception(f"gget.info failed for {clean}: {e}")
        return None

    if info is None or info.empty:
        logger.warning(f"No gene info returned for {clean}")
        return None

    info.to_csv(fp, index=False)
    logger.info(f"Fetched and cached gene info -> {fp}")
    return info


def _row_to_details(row):
    """Pull the display fields (symbol/description/biotype) from one gene row."""
    symbol = _extract_field(row, _SYMBOL_COLS)
    if symbol and symbol.startswith("ENSG"):
        symbol = None  # not a real HGNC symbol
    return {
        "symbol": symbol,
        "description": _extract_field(row, _DESC_COLS),
        "biotype": _extract_field(row, ("biotype",)),
    }


def annotate_genes(ensembl_ids, cache_dir=GENE_INFO_CACHE_DIR):
    """Resolve full specifications for each gene from the on-disk cache.

    For every Ensembl ID: read ``cache_dir/<clean_id>.csv`` if present, otherwise
    request it via gget.info and save it back (same per-gene CSV format as
    get_gene_info.py).

    Returns:
        (details, full_info_df) where ``details`` maps each original ensembl_id
        to {symbol, description, biotype}, and ``full_info_df`` concatenates the
        complete gget.info specification rows for the resolved genes.
    """
    os.makedirs(cache_dir, exist_ok=True)

    details = {}
    full_rows = []
    n_hit = 0
    for eid in ensembl_ids:
        fp = os.path.join(cache_dir, f"{_clean_ensembl_id(eid)}.csv")
        cached = os.path.exists(fp)
        df = load_or_fetch_gene_csv(eid, cache_dir)
        if df is None or df.empty:
            details[eid] = {"symbol": None, "description": None, "biotype": None}
            continue
        n_hit += 1 if cached else 0
        full_rows.append(df)
        details[eid] = _row_to_details(df.iloc[0])

    logger.info(
        f"gene-info cache ({cache_dir}): {n_hit} hit, "
        f"{len(ensembl_ids) - n_hit} fetched/missing"
    )

    full_info_df = (
        pd.concat(full_rows, ignore_index=True) if full_rows else pd.DataFrame()
    )
    return details, full_info_df


# ---------------------------------------------------------------------------
# LUAD relevance via Enrichr pathway enrichment
# ---------------------------------------------------------------------------


def _term_col(df):
    return next(
        (c for c in ("path_name", "term_name", "term", "Term") if c in df.columns),
        None,
    )


def _genes_col(df):
    return next(
        (c for c in ("overlapping_genes", "genes", "Genes") if c in df.columns),
        None,
    )


def run_enrichment(gene_symbols, libraries=None):
    """Run gget.enrichr across libraries and return one LUAD-flagged DataFrame.

    Adds a ``library`` column and a boolean ``luad_related`` column (term keyword
    match OR overlap with a known LUAD driver gene). Empty DataFrame on no input.
    """
    if not gene_symbols:
        logger.info("No resolvable gene symbols; skipping enrichment.")
        return pd.DataFrame()

    libraries = libraries or DEFAULT_ENRICHR_LIBRARIES
    logger.info(f"Enrichr for {len(gene_symbols)} symbols: {gene_symbols}")

    frames = []
    for library in libraries:
        try:
            res = gget.enrichr(gene_symbols, database=library)
        except Exception as e:
            logger.exception(f"  enrichr failed for '{library}': {e}")
            continue
        if res is None or len(res) == 0:
            continue
        res = pd.DataFrame(res)
        res.insert(0, "library", library)
        frames.append(res)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    term_col, genes_col = _term_col(combined), _genes_col(combined)

    def _is_luad(row):
        term = str(row[term_col]).lower() if term_col else ""
        if any(kw in term for kw in LUAD_TERM_KEYWORDS):
            return True
        if genes_col is not None:
            genes = row[genes_col]
            if isinstance(genes, str):
                genes = genes.replace(";", ",").split(",")
            genes = {str(g).strip().upper() for g in (genes or [])}
            if genes & LUAD_RELATED_GENES:
                return True
        return False

    combined["luad_related"] = combined.apply(_is_luad, axis=1)
    logger.info(
        f"  {int(combined['luad_related'].sum())}/{len(combined)} terms LUAD-flagged"
    )
    return combined


def luad_pathways_per_gene(enrich_df):
    """Map each gene symbol -> sorted list of LUAD-related terms it appears in."""
    if enrich_df is None or enrich_df.empty or "luad_related" not in enrich_df.columns:
        return {}
    term_col, genes_col = _term_col(enrich_df), _genes_col(enrich_df)
    if term_col is None or genes_col is None:
        return {}

    mapping = {}
    for _, row in enrich_df[enrich_df["luad_related"]].iterrows():
        genes = row[genes_col]
        if isinstance(genes, str):
            genes = genes.replace(";", ",").split(",")
        term = str(row[term_col])
        for g in genes or []:
            mapping.setdefault(str(g).strip().upper(), set()).add(term)
    return {g: sorted(terms) for g, terms in mapping.items()}


# ---------------------------------------------------------------------------
# Assemble + export
# ---------------------------------------------------------------------------


def build_table(ranking_frames, details, gene_pathways):
    """Return a single tidy DataFrame (one row per top gene, all rankings).

    ``ranking_frames`` is an iterable of the per-direction DataFrames
    (positive / negative / magnitude); each row keeps its ``direction`` and
    within-direction ``rank``.
    """
    rows = []
    for df in ranking_frames:
        for _, r in df.iterrows():
            eid = r["ensembl_id"]
            info = details.get(eid, {})
            symbol = info.get("symbol")
            luad_terms = gene_pathways.get(str(symbol).upper(), []) if symbol else []
            rows.append(
                {
                    "direction": r["direction"],
                    "rank": int(r["rank"]),
                    "ensembl_id": eid,
                    "symbol": symbol,
                    "description": info.get("description"),
                    "biotype": info.get("biotype"),
                    "mean_ig": round(float(r["mean_ig"]), 6),
                    "std_ig": round(float(r["std_ig"]), 6),
                    "mean_abs_ig": round(float(r["mean_abs_ig"]), 6),
                    "std_abs_ig": round(float(r["std_abs_ig"]), 6),
                    "median_ig": round(float(r["median_ig"]), 6),
                    "median_abs_ig": round(float(r["median_abs_ig"]), 6),
                    "frac_positive": round(float(r["frac_positive"]), 4),
                    "frac_negative": round(float(r["frac_negative"]), 4),
                    "n_patients": int(r["n_patients"]),
                    "luad_driver": bool(str(symbol).upper() in LUAD_RELATED_GENES),
                    "n_luad_pathways": len(luad_terms),
                    "luad_pathways": "; ".join(luad_terms),
                }
            )
    return pd.DataFrame(rows)


def export_results(table, enrich_df, full_info_df, output_dir, top_n):
    """Write top_genes.csv/json, the full-spec CSV, and pathway_enrichment.csv."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "top_genes.csv")
    table.to_csv(csv_path, index=False)
    logger.info(f"Top gene table -> {csv_path}")

    # Full gget.info specifications for the top genes (every column), concatenated
    # from the per-gene cache files -- one combined table of gene specs.
    if full_info_df is not None and not full_info_df.empty:
        info_path = os.path.join(output_dir, "top_genes_gene_info.csv")
        full_info_df.to_csv(info_path, index=False)
        logger.info(f"Top gene full specifications -> {info_path}")

    json_path = os.path.join(output_dir, "top_genes.json")
    bundle = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top_n_per_direction": top_n,
        "ranking_metrics": {
            "signed": "mean signed IG (positive -> increased predicted risk, "
            "negative -> decreased)",
            "magnitude": "mean absolute IG (attribution magnitude independent "
            "of direction)",
        },
        "top_positive_genes": table[table["direction"] == "positive"].to_dict(
            "records"
        ),
        "top_negative_genes": table[table["direction"] == "negative"].to_dict(
            "records"
        ),
        "top_magnitude_genes": table[table["direction"] == "magnitude"].to_dict(
            "records"
        ),
    }
    with open(json_path, "w") as fh:
        json.dump(bundle, fh, indent=2)
    logger.info(f"Top gene bundle -> {json_path}")

    if enrich_df is not None and not enrich_df.empty:
        enrich_path = os.path.join(output_dir, "pathway_enrichment.csv")
        enrich_df.to_csv(enrich_path, index=False)
        logger.info(f"Pathway enrichment -> {enrich_path}")


def log_table(table):
    """Print a compact human-readable summary of the top genes."""
    # (direction, headline, the metric column the ranking is sorted by).
    sections = [
        (
            "positive",
            "MOST POSITIVE GENES (increased predicted risk, by mean signed IG)",
            "mean_ig",
        ),
        (
            "negative",
            "MOST NEGATIVE GENES (decreased predicted risk, by mean signed IG)",
            "mean_ig",
        ),
        ("magnitude", "LARGEST-MAGNITUDE GENES (by mean absolute IG)", "mean_abs_ig"),
    ]
    for direction, headline, metric_col in sections:
        subset = table[table["direction"] == direction]
        if subset.empty:
            continue
        metric_label = "Mean |IG|" if metric_col == "mean_abs_ig" else "Mean IG"
        logger.info("\n" + "=" * 88)
        logger.info(f"TOP {len(subset)} {headline}")
        logger.info("=" * 88)
        logger.info(
            f"{'Rank':<5}{'Symbol':<14}{metric_label:<12}{'LUAD drv':<9}"
            f"{'LUAD paths':<11}Description"
        )
        for _, r in subset.iterrows():
            desc = (r["description"] or "")[:40]
            logger.info(
                f"{r['rank']:<5}{str(r['symbol'] or r['ensembl_id'])[:13]:<14}"
                f"{r[metric_col]:<12.6f}{('yes' if r['luad_driver'] else '-'):<9}"
                f"{r['n_luad_pathways']:<11}{desc}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    mapping_file_path,
    ig_directory,
    output_dir,
    top_n,
    gene_info_cache_dir=GENE_INFO_CACHE_DIR,
    libraries=None,
):
    """Full pipeline: rank -> annotate -> LUAD relevance -> export."""
    gene_names = load_gene_names(mapping_file_path)
    ig_matrix, _ = load_ig_matrix(ig_directory, len(gene_names))

    stats = compute_gene_stats(ig_matrix, gene_names)
    # Signed ranking: features tied to increased / decreased predicted risk.
    top_positive, top_negative = rank_top_genes(stats, top_n)
    # Magnitude ranking: attribution size independent of direction.
    top_magnitude = rank_top_magnitude_genes(stats, top_n)
    ranking_frames = [top_positive, top_negative, top_magnitude]

    # De-duplicate Ensembl IDs across rankings for annotation/enrichment
    # (magnitude genes often overlap the signed extremes), preserving order.
    ensembl_ids = list(
        dict.fromkeys(eid for frame in ranking_frames for eid in frame["ensembl_id"])
    )
    details, full_info_df = annotate_genes(ensembl_ids, gene_info_cache_dir)

    symbols = [d["symbol"] for d in details.values() if d["symbol"]]
    enrich_df = run_enrichment(symbols, libraries)
    gene_pathways = luad_pathways_per_gene(enrich_df)

    table = build_table(ranking_frames, details, gene_pathways)
    log_table(table)
    export_results(table, enrich_df, full_info_df, output_dir, top_n)
    return table


def main(config, opt):
    base_dir = Path(config.testing.output_base_dir)
    ig_directory = opt.ig_dir or str(base_dir / "IG_6sep")
    output_dir = str(base_dir / "interpret_omics")
    run(
        mapping_file_path=opt.mapping_file or config.data.json_file,
        ig_directory=ig_directory,
        output_dir=output_dir,
        top_n=opt.top_n,
        gene_info_cache_dir=opt.gene_info_dir,
    )
    logger.info("\nDone. See top_genes.csv / top_genes.json for the table.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank top IG genes and assess LUAD relevance."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/checkpoints/checkpoint_2026-02-14-05-44-19/config.yaml",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of most-positive and most-negative genes to report (each).",
    )
    parser.add_argument(
        "--ig-dir",
        type=str,
        default=None,
        help="IG directory. Defaults to <output_base_dir>/IG_6sep.",
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        default=None,
        help="Mapping JSON. Defaults to config.data.json_file.",
    )
    parser.add_argument(
        "--gene-info-dir",
        type=str,
        default=GENE_INFO_CACHE_DIR,
        help="Per-gene <ensembl_id>.csv cache; misses are fetched via gget.info "
        "and written back here.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    config = ConfigManager.load_config(opt.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(
        log_dir=config.testing.output_base_dir,
        log_level=config.logging.log_level,
        log_name=f"interpret_omics_{timestamp}.log",
    )

    main(config, opt)
