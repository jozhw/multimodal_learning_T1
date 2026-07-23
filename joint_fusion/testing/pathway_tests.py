"""
pathway_tests.py

Reads the pathway_analysis_bundle.npz written by pathway_interpret.py and runs three tests --
gene-label permutation null, GSEA prerank, hypergeometric ORA -- writing their
p-values / FDR / leading-edge tables. The discovery tests need only the bundle (symbol
attribution matrices + membership); the known-LUAD panel additionally loads the KEGG
Medicus NSCLC modules from MSigDB, since it is scored outside the discovery universe.

--tail sets how the permutation p is extrapolated below the empirical floor: gpd
(default; Knijnenburg 2009 GPD hybrid, tail shape estimated from the data) or empirical.

Run (bare uses the fold-1 config's output folder; override with --dir or --bundle):
    python -m joint_fusion.testing.pathway_tests
    python -m joint_fusion.testing.pathway_tests --dir <output_base_dir>/pathway_interpret
    python -m joint_fusion.testing.pathway_tests --bundle path/to/pathway_analysis_bundle.npz
    python -m joint_fusion.testing.pathway_tests --only-known-luad --n-perm 10000
    python -m joint_fusion.testing.pathway_tests --tail empirical   # floored ground truth

    Outputs (written next to the bundle):
        pathway_permutation_stats.csv   per-pathway summed|IG|, perm_z, p, empirical p,
                                        FDR, perm_tail (the tail model used)
        pathway_scores_with_stats.csv   pathway_scores.csv + the permutation columns
                                        (only if pathway_scores.csv is present)
        gsea_prerank.csv                GSEApy prerank NES / p / FDR / leading edge
        ora_{magnitude,up,down}.csv     hypergeometric enrichment of the top genes
        known_luad_panel_stats.csv      targeted KEGG NSCLC module permutation p/FDR
                                        plus bootstrap CIs
        known_luad_gsea_prerank.csv     targeted KEGG NSCLC module GSEA, if gradients exist
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto, hypergeom
from statsmodels.stats.multitest import multipletests

from joint_fusion.testing.pathway_interpret import (
    ANALYSIS_BUNDLE_NAME,
    KEGG_NSCLC_PANEL,
    MSIGDB_DIR,
    direction_label,
    load_collections,
    pathway_scores,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer C: permutation null on the pathway statistic
# ---------------------------------------------------------------------------

# GPD tail (Knijnenburg 2009): keep the empirical p where >= GPD_MIN_EXCEED null draws
# exceed the observed; otherwise fit a GPD to the top GPD_N_EXCEED null values.
GPD_MIN_EXCEED = 10
GPD_N_EXCEED = 250


def _gpd_tail_pvalues(
    null, observed, exceed, min_exceed=GPD_MIN_EXCEED, n_exceed=GPD_N_EXCEED
):
    """Peaks-over-threshold permutation p-values (Knijnenburg et al. 2009).

    ``null`` is the (n_perm, n_pathways) null matrix, ``observed`` the per-pathway
    statistic, ``exceed`` the count of null draws >= observed. Where >= min_exceed draws
    exceed the observed the empirical p is reliable and kept; otherwise (the floored
    pathways) a GPD is fitted to the top n_exceed null values and its tail probability
    used. By Pickands-Balkema-de Haan the threshold exceedances converge to a GPD for any
    parent null, so no shape is assumed. A failed/degenerate fit falls back to the
    floored empirical p.
    """
    n_perm = null.shape[0]
    p = (exceed + 1) / (n_perm + 1)  # empirical default for every pathway
    if n_perm < 100:
        logger.warning(
            "GPD tail needs ~100+ permutations to populate the tail; "
            "falling back to empirical p-values."
        )
        return p

    n_tail = min(n_exceed, n_perm - 1)
    tiny = np.finfo(np.float64).tiny
    for j in np.flatnonzero(exceed < min_exceed):
        tail_sorted = np.sort(null[:, j])[-n_tail:]  # ascending top n_tail
        threshold = float(tail_sorted[0])
        if float(observed[j]) <= threshold:
            continue  # observed sits where the empirical p is already resolved
        excess = (tail_sorted - threshold).astype(np.float64)
        try:
            xi, _, beta = genpareto.fit(excess, floc=0.0)
            if not (np.isfinite(xi) and np.isfinite(beta) and beta > 0):
                continue
            tail_p = float(
                genpareto.sf(float(observed[j]) - threshold, xi, loc=0.0, scale=beta)
            )
        except Exception:
            continue
        if not np.isfinite(tail_p):
            continue
        # tail_p == 0: observed beyond the bounded (xi<0) GPD support; floor final p > 0.
        p[j] = max((n_tail / n_perm) * tail_p, tiny)
    return p


def permutation_null(ig_symbols, membership, sizes, n_perm=1000, seed=0, tail="gpd"):
    """Is a pathway's summed |mean IG| bigger than a size-matched random set? Shuffle
    gene labels (each set keeps its size) n_perm times and recompute.

    The empirical p is floored at 1/(n_perm+1), which sits above the BH threshold once
    ~1e3 sets are tested, so alone it calls everything non-significant. ``tail`` sets the
    extrapolation below that floor:
      "gpd"       (default) Knijnenburg 2009 hybrid; the tail shape is estimated from the
                  data (GPD), not assumed. Parametric fits (log-normal/gamma) were dropped
                  as their only support was one calibration run, not a tail shape.
      "empirical" the raw floored p, no extrapolation.

    Returns ``p_value`` (chosen tail) and ``p_empirical`` (always). ``z`` is a
    size-corrected ranking effect size (log-space standardised observed vs null), not a
    p-value; reported for any ``tail``.
    """
    observed = np.abs(pathway_scores(ig_symbols, membership, sizes)).sum(axis=0)
    if not np.isfinite(observed).all():
        raise ValueError("Non-finite pathway score; check the IG arrays for NaN/Inf.")

    rng = np.random.default_rng(seed)
    n_genes = ig_symbols.shape[1]
    n_pathways = len(sizes)

    ig32 = ig_symbols.astype(np.float32)
    membership32 = membership.astype(np.float32)
    sizes32 = sizes.astype(np.float32)
    observed32 = observed.astype(np.float32)

    # Full null so empirical and GPD tails read the same draws (~8 MB at 1000 x 1900).
    null = np.empty((n_perm, n_pathways), dtype=np.float32)
    for i in range(n_perm):
        permuted = membership32[:, rng.permutation(n_genes)]
        null[i] = np.abs(pathway_scores(ig32, permuted, sizes32)).sum(axis=0)
        if (i + 1) % 200 == 0:
            logger.info(f"  pathway permutation {i + 1}/{n_perm}")

    null_mean = null.mean(axis=0, dtype=np.float64)
    exceed = (null >= observed32).sum(axis=0)
    p_empirical = (exceed + 1) / (n_perm + 1)

    # Log-space z: size-corrected ranking effect size (not a p-value).
    log_null = np.log(np.maximum(null, np.float32(1e-30)))
    log_mean = log_null.mean(axis=0, dtype=np.float64)
    log_sd = log_null.std(axis=0, dtype=np.float64)
    log_sd[log_sd == 0] = np.nan
    z_scores = (np.log(np.maximum(observed, 1e-300)) - log_mean) / log_sd

    if tail == "empirical":
        p_value = p_empirical
    elif tail == "gpd":
        p_value = _gpd_tail_pvalues(null, observed, exceed)
    else:
        raise ValueError(f"Unknown tail model: {tail!r} (use 'gpd' or 'empirical').")

    return {
        "observed": observed,
        "null_mean": null_mean,
        "z": z_scores,
        "p_value": p_value,
        "p_empirical": p_empirical,
        "tail": tail,
    }


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg step-up FDR (statsmodels) on the finite p-values only.

    Unresolved p-values (NaN/Inf) are dropped and returned as NaN, so they neither get a
    q-value nor inflate the test count n.
    """
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.any():
        q[finite] = multipletests(p[finite], method="fdr_bh")[1]
    return q


# ---------------------------------------------------------------------------
# Layer C: GSEA prerank
# ---------------------------------------------------------------------------


def _membership_to_gene_sets(membership, names, symbols):
    """Convert a membership matrix into the dict expected by GSEApy."""
    symbols = np.asarray(symbols)
    return {name: symbols[membership[i] > 0].tolist() for i, name in enumerate(names)}


def _leading_edge_size(value):
    if pd.isna(value) or value == "":
        return 0
    return len(str(value).split(";"))


def gsea_prerank(
    rank_metric,
    membership,
    names,
    n_perm=1000,
    seed=0,
    symbols=None,
    threads=1,
):
    """GSEA prerank via GSEApy (Subramanian et al. 2005; Fang et al. 2023).

    The bundle membership already applied the intended size filter, so GSEApy's min_size
    is set to 1 to avoid dropping small pre-specified driver modules.
    """
    try:
        import gseapy as gp
    except ImportError as e:
        raise ImportError(
            "GSEApy is required for the GSEA prerank step. Install it in this "
            "environment, or rerun with --skip-gsea to omit GSEA."
        ) from e

    if symbols is None:
        symbols = [str(i) for i in range(len(rank_metric))]
    rank_metric = np.asarray(rank_metric, dtype=float)
    symbols = np.asarray(symbols)
    finite = np.isfinite(rank_metric)
    if not finite.any():
        raise ValueError("GSEA rank metric contains no finite values.")

    ranks = pd.DataFrame(
        {
            "gene_name": symbols[finite].astype(str),
            "rank_metric": rank_metric[finite],
        }
    )
    ranks = ranks.sort_values("rank_metric", ascending=False, ignore_index=True)
    gene_sets = _membership_to_gene_sets(membership[:, finite], names, symbols[finite])

    result = gp.prerank(
        rnk=ranks,
        gene_sets=gene_sets,
        outdir=None,
        min_size=1,
        max_size=len(ranks),
        permutation_num=n_perm,
        weight=1.0,
        ascending=False,
        threads=threads,
        seed=seed,
        no_plot=True,
        verbose=False,
    )
    table = result.res2d.rename(
        columns={
            "Term": "pathway",
            "ES": "es",
            "NES": "nes",
            "NOM p-val": "p_value",
            "FDR q-val": "fdr_q",
            "FWER p-val": "fwer_p_value",
            "Lead_genes": "leading_edge_genes",
            "Tag %": "tag_percent",
            "Gene %": "gene_percent",
        }
    )
    for column in ("es", "nes", "p_value", "fdr_q", "fwer_p_value"):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    sizes = dict(zip(names, membership.sum(axis=1).astype(int)))
    table["size"] = table["pathway"].map(sizes)
    table["leading_edge_size"] = table["leading_edge_genes"].map(_leading_edge_size)
    out_cols = [
        "pathway",
        "es",
        "nes",
        "p_value",
        "fdr_q",
        "fwer_p_value",
        "size",
        "leading_edge_size",
        "leading_edge_genes",
        "tag_percent",
        "gene_percent",
    ]
    out_cols = [c for c in out_cols if c in table.columns]
    return table[out_cols].sort_values(
        ["fdr_q", "p_value", "nes"],
        ascending=[True, True, False],
        ignore_index=True,
        na_position="last",
    )


# ---------------------------------------------------------------------------
# Layer C: hypergeometric ORA
# ---------------------------------------------------------------------------


def over_representation(hits, gene_sets, background, min_members=10):
    """Hypergeometric ORA of a gene list against the gene sets.

    Background is the genes the model saw, not the genome: a whole-genome background
    (Enrichr's default) inflates p-values by counting genes that could never be selected.
    """
    background = set(background)
    hits = [g for g in hits if g in background]
    universe = len(background)
    n_drawn = len(hits)
    hit_set = set(hits)

    rows = []
    for name, members in sorted(gene_sets.items()):
        in_universe = members & background
        if len(in_universe) < min_members:
            continue
        overlap = hit_set & in_universe
        if not overlap:
            continue
        p = hypergeom.sf(len(overlap) - 1, universe, len(in_universe), n_drawn)
        expected = n_drawn * len(in_universe) / universe
        rows.append(
            {
                "pathway": name,
                "overlap_k": len(overlap),
                "pathway_size_K": len(in_universe),
                "list_size_n": n_drawn,
                "universe_N": universe,
                "fold_enrichment": len(overlap) / expected if expected else np.nan,
                "p_value": p,
                "overlap_genes": ";".join(sorted(overlap)),
            }
        )

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    table["fdr_q"] = benjamini_hochberg(table["p_value"].to_numpy())
    return table.sort_values("fdr_q", ignore_index=True)


# ---------------------------------------------------------------------------
# Bundle I/O and the targeted known-LUAD/NSCLC driver panel
# ---------------------------------------------------------------------------


def load_bundle(bundle_path):
    """Load the symbol-level matrices + membership the three tests need."""
    d = np.load(bundle_path, allow_pickle=False)
    has_pg = int(d["has_path_gradients"]) == 1
    return {
        "ig_symbols": d["ig_symbols"].astype(np.float64),
        "path_gradient_symbols": (
            d["path_gradient_symbols"].astype(np.float64) if has_pg else None
        ),
        "membership": d["membership"].astype(np.float64),
        "sizes": d["sizes"].astype(np.float64),
        "names": [str(x) for x in d["names"]],
        "symbols": [str(x) for x in d["symbols"]],
        "collections": [str(x) for x in d["collections"]],
    }


def measured_gene_sets(membership, names, symbols):
    """Rebuild {pathway: set(measured member symbols)} from the membership matrix.

    Membership already restricts each pathway to the measured gene axis, so this gives
    exactly the sets ORA uses with the measured-gene background.
    """
    symbols = np.asarray(symbols)
    return {
        name: set(symbols[membership[i] > 0].tolist()) for i, name in enumerate(names)
    }


def known_luad_panel_membership(symbols, msigdb_dir=MSIGDB_DIR, min_members=3):
    """Build the pre-specified KEGG Medicus NSCLC driver panel on the bundle gene axis."""
    gene_sets, _ = load_collections(msigdb_dir, ("c2.cp.kegg_medicus",))
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}

    rows = []
    membership_rows = []
    scored_names = []
    scored_nodes = []
    scored_sizes = []

    for pathway, node in KEGG_NSCLC_PANEL.items():
        annotated = gene_sets.get(pathway)
        if annotated is None:
            rows.append(
                {
                    "kegg_node": node,
                    "pathway": pathway,
                    "n_annotated_genes": 0,
                    "n_measured_genes": 0,
                    "coverage": np.nan,
                    "scored": False,
                    "skip_reason": "missing_from_msigdb",
                }
            )
            continue

        members = [symbol_index[g] for g in annotated if g in symbol_index]
        scored = len(members) >= min_members
        rows.append(
            {
                "kegg_node": node,
                "pathway": pathway,
                "n_annotated_genes": len(annotated),
                "n_measured_genes": len(members),
                "coverage": len(members) / len(annotated) if annotated else np.nan,
                "scored": scored,
                "skip_reason": (
                    "" if scored else f"fewer_than_{min_members}_measured_genes"
                ),
            }
        )
        if scored:
            row = np.zeros(len(symbols), dtype=np.float64)
            row[members] = 1.0
            membership_rows.append(row)
            scored_names.append(pathway)
            scored_nodes.append(node)
            scored_sizes.append(len(members))

    membership = (
        np.vstack(membership_rows)
        if membership_rows
        else np.zeros((0, len(symbols)), dtype=np.float64)
    )
    return (
        pd.DataFrame(rows),
        membership,
        scored_names,
        scored_nodes,
        np.asarray(scored_sizes),
    )


def bootstrap_panel_scores(
    ig_scores, path_gradient_scores=None, n_boot=2000, seed=0, ci_level=0.95
):
    """Patient bootstrap CIs for panel effect-size summaries.

    Complements the permutation test's competitive p-value: measures how stable the panel
    score is across the patient cohort.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    n_patients, n_pathways = ig_scores.shape
    alpha = (1.0 - ci_level) / 2.0

    boot_mean_abs_ig = np.empty((n_boot, n_pathways), dtype=np.float64)
    boot_mean_gradient = (
        np.empty((n_boot, n_pathways), dtype=np.float64)
        if path_gradient_scores is not None
        else None
    )

    for i in range(n_boot):
        idx = rng.integers(0, n_patients, size=n_patients)
        boot_mean_abs_ig[i] = np.abs(ig_scores[idx]).mean(axis=0)
        if path_gradient_scores is not None:
            boot_mean_gradient[i] = path_gradient_scores[idx].mean(axis=0)

    out = {
        "mean_abs_ig_boot_mean": boot_mean_abs_ig.mean(axis=0),
        "mean_abs_ig_ci_low": np.quantile(boot_mean_abs_ig, alpha, axis=0),
        "mean_abs_ig_ci_high": np.quantile(boot_mean_abs_ig, 1.0 - alpha, axis=0),
    }
    if boot_mean_gradient is not None:
        out.update(
            {
                "mean_path_gradient_boot_mean": boot_mean_gradient.mean(axis=0),
                "mean_path_gradient_ci_low": np.quantile(
                    boot_mean_gradient, alpha, axis=0
                ),
                "mean_path_gradient_ci_high": np.quantile(
                    boot_mean_gradient, 1.0 - alpha, axis=0
                ),
                "bootstrap_frac_risk_increasing": (boot_mean_gradient > 0).mean(axis=0),
            }
        )
    return out


def run_known_luad_panel(
    bundle,
    output_dir,
    msigdb_dir=MSIGDB_DIR,
    n_perm=1000,
    n_boot=2000,
    seed=0,
    min_members=3,
    ci_level=0.95,
    skip_gsea=False,
    gsea_threads=1,
    tail="gpd",
):
    """Targeted evidence layer for the pre-specified KEGG NSCLC driver panel."""
    ig_symbols = bundle["ig_symbols"]
    path_gradient_symbols = bundle["path_gradient_symbols"]
    symbols = bundle["symbols"]

    panel_base, membership, names, nodes, sizes = known_luad_panel_membership(
        symbols, msigdb_dir=msigdb_dir, min_members=min_members
    )
    if len(names) == 0:
        panel_path = os.path.join(output_dir, "known_luad_panel_stats.csv")
        panel_base.to_csv(panel_path, index=False)
        logger.info(f"Known LUAD panel: no modules scored -> {panel_path}")
        return

    logger.info(
        f"Known LUAD/NSCLC panel permutation for {len(names)} scored modules "
        f"({n_perm} permutations)"
    )
    null = permutation_null(ig_symbols, membership, sizes, n_perm, seed, tail=tail)
    ig_scores = pathway_scores(ig_symbols, membership, sizes)
    path_scores = (
        pathway_scores(path_gradient_symbols, membership, sizes)
        if path_gradient_symbols is not None
        else None
    )
    boot = bootstrap_panel_scores(
        ig_scores,
        path_scores,
        n_boot=n_boot,
        seed=seed,
        ci_level=ci_level,
    )

    scored = pd.DataFrame(
        {
            "pathway": names,
            "kegg_node": nodes,
            "summed_abs_ig": null["observed"],
            "mean_abs_ig": np.abs(ig_scores).mean(axis=0),
            "perm_z": null["z"],
            "perm_null_mean": null["null_mean"],
            "perm_p_value": null["p_value"],
            "perm_p_empirical": null["p_empirical"],
            "perm_fdr_q": benjamini_hochberg(null["p_value"]),
            "perm_tail": null["tail"],
            "mean_path_gradient": (
                path_scores.mean(axis=0)
                if path_scores is not None
                else np.full(len(names), np.nan)
            ),
            "path_direction": (
                [direction_label(x) for x in path_scores.mean(axis=0)]
                if path_scores is not None
                else ["unknown"] * len(names)
            ),
        }
    )
    for col, values in boot.items():
        scored[col] = values

    table = panel_base.merge(scored, on=["pathway", "kegg_node"], how="left")
    table = table.sort_values(
        ["scored", "perm_fdr_q", "summed_abs_ig"],
        ascending=[False, True, False],
        na_position="last",
        ignore_index=True,
    )
    panel_path = os.path.join(output_dir, "known_luad_panel_stats.csv")
    table.to_csv(panel_path, index=False)
    n_sig = int((table["perm_fdr_q"] < 0.05).sum())
    logger.info(
        f"Known LUAD panel -> {panel_path} "
        f"({n_sig} modules at targeted panel FDR < 0.05)"
    )

    if path_gradient_symbols is not None and not skip_gsea:
        logger.info(f"Known LUAD GSEA prerank via GSEApy ({n_perm} permutations)")
        gsea = gsea_prerank(
            path_gradient_symbols.mean(axis=0),
            membership,
            names,
            n_perm,
            seed,
            symbols,
            threads=gsea_threads,
        )
        gsea["kegg_node"] = gsea["pathway"].map(dict(zip(names, nodes)))
        gsea_path = os.path.join(output_dir, "known_luad_gsea_prerank.csv")
        gsea.to_csv(gsea_path, index=False)
        n_gsea_sig = int((gsea["fdr_q"] < 0.25).sum())
        logger.info(
            f"Known LUAD GSEA -> {gsea_path} " f"({n_gsea_sig} modules at FDR < 0.25)"
        )


# ---------------------------------------------------------------------------
# Orchestration and CLI
# ---------------------------------------------------------------------------


def run_layer_c(
    bundle,
    output_dir,
    n_perm=1000,
    seed=0,
    ora_top_n=100,
    min_members=10,
    skip_gsea=False,
    msigdb_dir=MSIGDB_DIR,
    panel_min_members=3,
    n_boot=2000,
    ci_level=0.95,
    skip_known_luad=False,
    gsea_threads=1,
    tail="gpd",
):
    ig_symbols = bundle["ig_symbols"]
    path_gradient_symbols = bundle["path_gradient_symbols"]
    membership = bundle["membership"]
    sizes = bundle["sizes"]
    names = bundle["names"]
    symbols = bundle["symbols"]
    collection_of = dict(zip(names, bundle["collections"]))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Permutation null on the paper's own pathway statistic.
    logger.info(f"Permutation null for {len(names)} pathways ({n_perm} permutations)")
    null = permutation_null(ig_symbols, membership, sizes, n_perm, seed, tail=tail)
    stats = pd.DataFrame(
        {
            "pathway": names,
            "collection": [collection_of[n] for n in names],
            "n_measured_genes": sizes.astype(int),
            "summed_abs_ig": null["observed"],
            "perm_z": null["z"],
            "perm_null_mean": null["null_mean"],
            "perm_p_value": null["p_value"],
            "perm_p_empirical": null["p_empirical"],
            "perm_fdr_q": benjamini_hochberg(null["p_value"]),
            "perm_tail": null["tail"],
        }
    ).sort_values("summed_abs_ig", ascending=False, ignore_index=True)
    stats_path = os.path.join(output_dir, "pathway_permutation_stats.csv")
    stats.to_csv(stats_path, index=False)
    logger.info(f"Permutation stats -> {stats_path}")

    # Merge the permutation columns back into pathway_scores.csv if it is present.
    scores_csv = os.path.join(output_dir, "pathway_scores.csv")
    if os.path.exists(scores_csv):
        base = pd.read_csv(scores_csv)
        stat_cols = [
            "perm_z",
            "perm_null_mean",
            "perm_p_value",
            "perm_p_empirical",
            "perm_fdr_q",
            "perm_tail",
        ]
        base = base.drop(
            columns=[c for c in stat_cols if c in base.columns], errors="ignore"
        )
        merged = base.merge(stats[["pathway"] + stat_cols], on="pathway", how="left")
        merged_path = os.path.join(output_dir, "pathway_scores_with_stats.csv")
        merged.to_csv(merged_path, index=False)
        logger.info(f"Scores + permutation stats -> {merged_path}")

    # 2. GSEA prerank on the path gradients (direction-aware, threshold-free).
    if path_gradient_symbols is not None and not skip_gsea:
        logger.info(f"GSEA prerank via GSEApy ({n_perm} permutations)")
        gsea = gsea_prerank(
            path_gradient_symbols.mean(axis=0),
            membership,
            names,
            n_perm,
            seed,
            symbols,
            threads=gsea_threads,
        )
        gsea["collection"] = gsea["pathway"].map(collection_of)
        gsea_path = os.path.join(output_dir, "gsea_prerank.csv")
        gsea.to_csv(gsea_path, index=False)
        n_sig = int((gsea["fdr_q"] < 0.25).sum())
        logger.info(f"GSEA -> {gsea_path} ({n_sig} pathways at FDR < 0.25)")
    else:
        logger.info("GSEA skipped (bundle has no path gradients, or --skip-gsea).")

    # 3. Hypergeometric ORA of the top genes, background = the measured gene axis.
    gene_sets = measured_gene_sets(membership, names, symbols)
    mean_abs = np.abs(ig_symbols).mean(axis=0)
    ora_lists = {"magnitude": [symbols[i] for i in np.argsort(-mean_abs)[:ora_top_n]]}
    if path_gradient_symbols is not None:
        mean_gradient = path_gradient_symbols.mean(axis=0)
        ora_lists["up"] = [symbols[i] for i in np.argsort(-mean_gradient)[:ora_top_n]]
        ora_lists["down"] = [symbols[i] for i in np.argsort(mean_gradient)[:ora_top_n]]

    for label, hits in ora_lists.items():
        result = over_representation(hits, gene_sets, symbols, min_members)
        if result.empty:
            logger.info(f"ORA ({label}): no pathway had any overlap.")
            continue
        result["collection"] = result["pathway"].map(collection_of)
        ora_path = os.path.join(output_dir, f"ora_{label}.csv")
        result.to_csv(ora_path, index=False)
        n_sig = int((result["fdr_q"] < 0.05).sum())
        logger.info(
            f"ORA ({label}, top {ora_top_n}) -> {ora_path} ({n_sig} at FDR < 0.05)"
        )

    # 4. Targeted known-LUAD/NSCLC panel: FDR over the pre-specified modules only,
    # separate from the discovery universe.
    if not skip_known_luad:
        try:
            run_known_luad_panel(
                bundle,
                output_dir,
                msigdb_dir=msigdb_dir,
                n_perm=n_perm,
                n_boot=n_boot,
                seed=seed,
                min_members=panel_min_members,
                ci_level=ci_level,
                skip_gsea=skip_gsea,
                gsea_threads=gsea_threads,
                tail=tail,
            )
        except FileNotFoundError as e:
            logger.warning(f"Known LUAD panel skipped: {e}")


DEFAULT_CONFIG = "joint_fusion/config/config_checkpoint_2026-04-07-04-58-17_fold1.yaml"


def resolve_bundle_path(opt):
    if opt.bundle:
        return Path(opt.bundle)
    if opt.dir:
        return Path(opt.dir) / ANALYSIS_BUNDLE_NAME
    # Neither given: mirror pathway_interpret's output folder so a bare run matches it.
    from joint_fusion.config.config_manager import ConfigManager

    config = ConfigManager.load_config(opt.config)
    base = Path(config.testing.output_base_dir) / "pathway_interpret"
    return base / ANALYSIS_BUNDLE_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Fold-1 config by default; used to locate the output folder "
        "when --dir/--bundle are not given.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="pathway_interpret output folder holding the bundle "
        "(overrides --config).",
    )
    parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Path to pathway_analysis_bundle.npz (overrides --dir).",
    )
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument(
        "--tail",
        choices=("gpd", "empirical"),
        default="gpd",
        help="Tail model for the permutation p-value below the empirical floor. "
        "gpd (default) is the Knijnenburg 2009 Generalized-Pareto hybrid, which estimates "
        "the tail shape from the data (no parametric assumption); empirical is the "
        "floored ground-truth p, kept for comparison.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Patient bootstrap resamples for known_luad_panel_stats.csv.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Bootstrap confidence level for the known LUAD panel.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ora-top-n", type=int, default=100)
    parser.add_argument("--min-members", type=int, default=10)
    parser.add_argument(
        "--msigdb-dir",
        type=str,
        default=MSIGDB_DIR,
        help="MSigDB asset folder; needed for the KEGG NSCLC/LUAD panel.",
    )
    parser.add_argument(
        "--panel-min-members",
        type=int,
        default=3,
        help="Min measured genes to score a KEGG NSCLC driver module.",
    )
    parser.add_argument(
        "--gsea-threads",
        type=int,
        default=1,
        help="Worker threads passed to GSEApy prerank.",
    )
    parser.add_argument("--skip-gsea", action="store_true")
    parser.add_argument(
        "--skip-known-luad",
        action="store_true",
        help="Skip the targeted known LUAD/NSCLC KEGG panel.",
    )
    parser.add_argument(
        "--only-known-luad",
        action="store_true",
        help="Run only the targeted known LUAD/NSCLC KEGG panel.",
    )
    opt = parser.parse_args()
    if opt.only_known_luad and opt.skip_known_luad:
        parser.error("--only-known-luad cannot be combined with --skip-known-luad.")

    bundle_path = resolve_bundle_path(opt)
    if not bundle_path.exists():
        raise SystemExit(
            f"Bundle not found: {bundle_path}\n"
            "Run joint_fusion.testing.pathway_interpret first (it writes the bundle)."
        )
    logger.info(f"Loading bundle: {bundle_path}")
    bundle = load_bundle(bundle_path)
    if opt.only_known_luad:
        run_known_luad_panel(
            bundle,
            output_dir=str(bundle_path.parent),
            msigdb_dir=opt.msigdb_dir,
            n_perm=opt.n_perm,
            n_boot=opt.n_boot,
            seed=opt.seed,
            min_members=opt.panel_min_members,
            ci_level=opt.ci_level,
            skip_gsea=opt.skip_gsea,
            gsea_threads=opt.gsea_threads,
            tail=opt.tail,
        )
        logger.info(
            "\nDone. See known_luad_panel_stats.csv and, unless skipped, "
            "known_luad_gsea_prerank.csv."
        )
    else:
        run_layer_c(
            bundle,
            output_dir=str(bundle_path.parent),
            n_perm=opt.n_perm,
            seed=opt.seed,
            ora_top_n=opt.ora_top_n,
            min_members=opt.min_members,
            skip_gsea=opt.skip_gsea,
            msigdb_dir=opt.msigdb_dir,
            panel_min_members=opt.panel_min_members,
            n_boot=opt.n_boot,
            ci_level=opt.ci_level,
            skip_known_luad=opt.skip_known_luad,
            gsea_threads=opt.gsea_threads,
            tail=opt.tail,
        )
        logger.info(
            "\nDone. See pathway_scores_with_stats.csv, gsea_prerank.csv, "
            "ora_*.csv, and known_luad_panel_stats.csv."
        )


if __name__ == "__main__":
    main()
