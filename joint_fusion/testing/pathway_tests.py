"""
pathway_tests.py

Reads the collection-independent gene_attribution_bundle.npz written by pathway_interpret.py,
rebuilds the gene-set membership for the requested --collections on the bundle's gene axis,
and runs three tests -- gene-label permutation null, GSEA prerank, hypergeometric ORA --
writing their p-values / FDR / leading-edge tables. Because membership is rebuilt at test
time, ONE gene bundle serves every collection (Reactome, Hallmark, C6, ...); run one
collection per invocation so each keeps its own FDR family (do not pool).

--tail sets how the permutation p is extrapolated below the empirical floor: gpd
(default; peaks-over-threshold with the tail shape fixed to xi=0, i.e. the exponential
member of the GPD family -- conservative under the statistic's bounded support and free of
the free-shape MLE's endpoint ill-conditioning) or empirical (raw floored p, no
extrapolation).

Run (bare uses the fold-1 config's output folder; --out-dir sets the per-collection folder,
--gene-bundle the shared bundle, --collections which universe):
    python -m joint_fusion.testing.pathway_tests --collections c2.cp.reactome \
        --out-dir <output_base_dir>/pathway_interpret_reactome
    python -m joint_fusion.testing.pathway_tests --collections h.all \
        --out-dir <output_base_dir>/pathway_interpret_hallmark
    python -m joint_fusion.testing.pathway_tests --tail empirical   # floored ground truth

    Outputs (written to --out-dir):
        pathway_permutation_stats.csv   per-pathway summed|IG|, perm_z, p, empirical p,
                                        FDR, perm_tail (the tail model used),
                                        perm_gpd_shape (diagnostic free-shape GPD xi)
        pathway_scores_with_stats.csv   pathway_scores.csv + the permutation columns
                                        (only if pathway_scores.csv is present)
        gsea_prerank.csv                GSEApy prerank NES / p / FDR / leading edge
        ora_{magnitude,up,down}.csv     hypergeometric enrichment of the top genes
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto, hypergeom
from statsmodels.stats.multitest import multipletests

from joint_fusion.testing.pathway_interpret import (
    GENE_BUNDLE_NAME,
    MSIGDB_DIR,
    build_membership,
    direction_label,
    load_collections,
    load_gene_bundle,
    pathway_scores,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer C: permutation null on the pathway statistic
# ---------------------------------------------------------------------------

# Peaks-over-threshold tail (Knijnenburg 2009): keep the empirical p where
# >= GPD_MIN_EXCEED null draws exceed the observed; otherwise extrapolate with a
# fixed-shape (xi=0) exponential tail fitted to the top GPD_N_EXCEED null values.
GPD_MIN_EXCEED = 10
GPD_N_EXCEED = 250


def _gpd_tail_pvalues(
    null, observed, exceed, min_exceed=GPD_MIN_EXCEED, n_exceed=GPD_N_EXCEED
):
    """Peaks-over-threshold permutation p-values with the tail shape fixed to xi=0 (the
    exponential member of the GPD family). See literature/gpd_permutation_derivation.tex.

    ``null`` is the (n_perm, n_pathways) null matrix, ``observed`` the per-pathway
    statistic, ``exceed`` the count of null draws >= observed. Where >= min_exceed draws
    exceed the observed the empirical p is reliable and kept as-is. Otherwise the pathway
    is below the empirical resolution floor 1/(n_perm+1) and we extrapolate its upper tail
    from the top n_exceed null values, whose smallest sets the threshold u.

    Why the exponential (xi=0), not the free-shape GPD. By Pickands-Balkema-de-Haan the
    threshold exceedances converge to a GPD for any parent null; the summed-|IG| statistic
    is a bounded sum over a finite gene pool, so its null has *bounded support* and hence a
    light tail, xi <= 0. Two problems make the free-shape MLE xi unusable here:
      * ill-conditioning -- for xi<0 the GPD has a finite endpoint u-beta/xi, and the tail
        probability near that endpoint is pathologically sensitive to xi: a ~0.03 change in
        an xi estimated from ~250 noisy draws moves the extrapolated p by tens of decades,
        and the endpoint-overshoot case (observed beyond u-beta/xi) makes genpareto.sf
        return exactly 0. This produced non-monotone, ~1e-300 "p-values".
      * variance -- estimating a second tail parameter from a small exceedance sample is
        high-variance regardless of conditioning.
    Fixing xi=0 (exponential) is the boundary member of the same family. Under bounded
    support (xi<=0) the exponential tail is the *heaviest admissible* tail, so its tail
    probability is an upper bound on the truth -- the p is conservative (never falsely
    tiny) -- while removing the shape-estimation variance and the endpoint cliff. Its MLE
    scale is the mean excess, so sf(ex) = exp(-ex / mean_excess), always positive until
    float underflow. The free-shape GPD is still fitted, but only as a *reported
    diagnostic* (``shape``); it does not enter the p-value.

    Returns ``(p, method, shape)`` where ``method[j]`` is how pathway j's p was obtained
    ("empirical", "exponential", or "degenerate" -> exponential unusable, empirical kept)
    and ``shape[j]`` is the diagnostic free-shape GPD xi for the extrapolated pathways
    (NaN where the empirical p was kept). shape<0 across the extrapolated sets is the
    empirical check on the bounded-support premise.
    """
    n_perm = null.shape[0]
    p = (exceed + 1) / (n_perm + 1)  # empirical default for every pathway
    method = np.array(["empirical"] * len(observed), dtype=object)
    shape = np.full(len(observed), np.nan, dtype=float)
    if n_perm < 100:
        logger.warning(
            "Tail extrapolation needs ~100+ permutations to populate the tail; "
            "falling back to empirical p-values."
        )
        return p, method, shape

    n_tail = min(n_exceed, n_perm - 1)
    tiny = np.finfo(np.float64).tiny
    for j in np.flatnonzero(exceed < min_exceed):
        tail_sorted = np.sort(null[:, j])[-n_tail:]  # ascending top n_tail
        threshold = float(tail_sorted[0])
        ex = float(observed[j]) - threshold
        if ex <= 0:
            continue  # observed sits inside the bulk; empirical p already resolves it
        excess = (tail_sorted - threshold).astype(np.float64)

        # Free-shape GPD fit: reported diagnostic only (xi<0 supports bounded support).
        try:
            xi, _, _ = genpareto.fit(excess, floc=0.0)
            if np.isfinite(xi):
                shape[j] = xi
        except Exception:
            pass

        # Exponential (xi=0) tail probability -- the value actually used. MLE scale =
        # mean excess; conservative under bounded support and free of the endpoint cliff.
        beta0 = float(np.mean(excess))
        if np.isfinite(beta0) and beta0 > 0:
            sf = float(np.exp(-ex / beta0))
            if np.isfinite(sf) and sf > 0:
                p[j] = max((n_tail / n_perm) * sf, tiny)
                method[j] = "exponential"
                continue

        method[j] = "degenerate"  # exponential unusable -> keep the empirical p in p[j]
    return p, method, shape


# ---------------------------------------------------------------------------
# Parallel permutation workers (module level so ProcessPoolExecutor can pickle them).
# Each worker holds the read-only IG/membership/sizes arrays in a module global (shipped
# once at pool start) and computes a batch of independent permutations.
# ---------------------------------------------------------------------------
_PERM_ARRAYS = {}


def _perm_worker_init(ig, membership, sizes):
    _PERM_ARRAYS["ig"] = ig
    _PERM_ARRAYS["membership"] = membership
    _PERM_ARRAYS["sizes"] = sizes


def _perm_worker_chunk(seed_seqs):
    ig = _PERM_ARRAYS["ig"]
    membership = _PERM_ARRAYS["membership"]
    sizes = _PERM_ARRAYS["sizes"]
    n_genes = membership.shape[1]
    out = np.empty((len(seed_seqs), membership.shape[0]), dtype=np.float32)
    for k, ss in enumerate(seed_seqs):
        rng = np.random.default_rng(ss)
        permuted = membership[:, rng.permutation(n_genes)]
        out[k] = np.abs(pathway_scores(ig, permuted, sizes)).sum(axis=0)
    return out


def permutation_null(
    ig_symbols, membership, sizes, n_perm=1000, seed=0, tail="gpd", n_jobs=1
):
    """Is a pathway's summed |mean IG| bigger than a size-matched random set? Shuffle
    gene labels (each set keeps its size) n_perm times and recompute.

    ``n_jobs`` parallelises the (independent) permutations across processes: 1 = serial,
    -1 = all cores, N = N workers. A per-permutation RNG stream (SeedSequence.spawn) makes
    the null identical for a given seed regardless of worker count -- but it differs from
    the pre-parallel single-stream draws, so p-values shift slightly (within Monte-Carlo
    error) on the first run after this change, then stay reproducible.

    The empirical p is floored at 1/(n_perm+1), which sits above the BH threshold once
    ~1e3 sets are tested, so alone it calls everything non-significant. ``tail`` sets the
    extrapolation below that floor:
      "gpd"       (default) peaks-over-threshold extrapolation with the tail shape fixed to
                  xi=0 (the exponential member of the GPD family). Conservative under the
                  statistic's bounded support and free of the free-shape MLE's endpoint
                  ill-conditioning; see _gpd_tail_pvalues and
                  literature/gpd_permutation_derivation.tex. (The flag value stays "gpd"
                  for back-compatibility; the free-shape xi is reported as a diagnostic.)
      "empirical" the raw floored p, no extrapolation.

    Returns ``p_value`` (chosen tail) and ``p_empirical`` (always). ``tail`` is a
    per-pathway array recording how each p was obtained ("empirical", "exponential", or
    "degenerate"), and ``shape`` the diagnostic free-shape GPD xi on the extrapolated
    pathways (NaN elsewhere). ``z`` is a size-corrected ranking effect size (log-space
    standardised observed vs null), not a p-value; reported for any ``tail``.
    """
    observed = np.abs(pathway_scores(ig_symbols, membership, sizes)).sum(axis=0)
    if not np.isfinite(observed).all():
        raise ValueError("Non-finite pathway score; check the IG arrays for NaN/Inf.")

    n_genes = ig_symbols.shape[1]
    n_pathways = len(sizes)

    ig32 = ig_symbols.astype(np.float32)
    membership32 = membership.astype(np.float32)
    sizes32 = sizes.astype(np.float32)
    observed32 = observed.astype(np.float32)

    # One independent RNG stream per permutation -> the null is identical whether run
    # serially or across any number of workers.
    seed_seqs = np.random.SeedSequence(seed).spawn(n_perm)

    # Full null so empirical and GPD tails read the same draws (~8 MB at 1000 x 1900).
    if n_jobs in (None, 0, 1):
        null = np.empty((n_perm, n_pathways), dtype=np.float32)
        for i in range(n_perm):
            rng = np.random.default_rng(seed_seqs[i])
            permuted = membership32[:, rng.permutation(n_genes)]
            null[i] = np.abs(pathway_scores(ig32, permuted, sizes32)).sum(axis=0)
            if (i + 1) % 200 == 0:
                logger.info(f"  pathway permutation {i + 1}/{n_perm}")
    else:
        n_cores = os.cpu_count() or 1
        workers = n_cores if n_jobs < 0 else min(n_jobs, n_cores)
        chunk = max(1, n_perm // (workers * 4))
        batches = [seed_seqs[i : i + chunk] for i in range(0, n_perm, chunk)]
        logger.info(
            f"  permuting with {workers} workers "
            f"({len(batches)} batches of ~{chunk}, n_perm={n_perm})"
        )
        # Pin BLAS to one thread per worker. Each permutation is one large matmul that
        # BLAS already multithreads across all cores, so the *serial* path is already
        # core-parallel. Without pinning, W workers x (BLAS threads each) oversubscribe
        # the cores and the parallel run is SEVERAL TIMES SLOWER than serial. Set before
        # the pool is created so spawned workers inherit it at their numpy import.
        thread_vars = (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        saved = {k: os.environ.get(k) for k in thread_vars}
        for k in thread_vars:
            os.environ[k] = "1"
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_perm_worker_init,
                initargs=(ig32, membership32, sizes32),
            ) as ex:
                parts = list(ex.map(_perm_worker_chunk, batches))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        null = np.vstack(parts)

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
        tail_method = np.array(["empirical"] * n_pathways, dtype=object)
        tail_shape = np.full(n_pathways, np.nan, dtype=float)
    elif tail == "gpd":
        p_value, tail_method, tail_shape = _gpd_tail_pvalues(null, observed, exceed)
    else:
        raise ValueError(f"Unknown tail model: {tail!r} (use 'gpd' or 'empirical').")

    return {
        "observed": observed,
        "null_mean": null_mean,
        "z": z_scores,
        "p_value": p_value,
        "p_empirical": p_empirical,
        "tail": tail_method,
        "shape": tail_shape,
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


def _bh_adjust_with_m(p_values, m):
    """Benjamini-Hochberg where the correction denominator is ``m`` -- the number of tests
    actually performed -- which may exceed ``len(p_values)``.

    ORA reports only the gene sets that overlap the hit list, but every size-eligible set
    was *tested* (the non-overlapping ones simply have p = 1). Correcting over the reported
    subset alone undercounts the tests and makes q anti-conservative; the honest denominator
    is the eligible-set count. Because the omitted sets all have p = 1 (the largest possible
    value) they occupy the top ranks and never lower another set's q, so passing ``m`` here
    is exactly equivalent to running BH over the full eligible set.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


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
    n_eligible = 0  # every size-eligible set is a test, whether or not it overlaps
    for name, members in sorted(gene_sets.items()):
        in_universe = members & background
        if len(in_universe) < min_members:
            continue
        n_eligible += 1
        overlap = hit_set & in_universe
        if not overlap:
            continue  # p would be 1.0; kept out of the table but counted in n_eligible
        p = hypergeom.sf(len(overlap) - 1, universe, len(in_universe), n_drawn)
        expected = n_drawn * len(in_universe) / universe
        rows.append(
            {
                "pathway": name,
                "overlap_k": len(overlap),
                "pathway_size_K": len(in_universe),
                "list_size_n": n_drawn,
                "universe_N": universe,
                "n_tests": n_eligible,  # BH denominator (filled after the loop)
                "fold_enrichment": len(overlap) / expected if expected else np.nan,
                "p_value": p,
                "overlap_genes": ";".join(sorted(overlap)),
            }
        )

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    table["n_tests"] = n_eligible
    # BH over ALL size-eligible sets, not just the overlapping ones that made the table.
    table["fdr_q"] = _bh_adjust_with_m(table["p_value"].to_numpy(), n_eligible)
    return table.sort_values("fdr_q", ignore_index=True)


# ---------------------------------------------------------------------------
# Gene bundle I/O and discovery membership helpers
# ---------------------------------------------------------------------------


def build_discovery_membership(symbols, collections, msigdb_dir=MSIGDB_DIR, min_members=10):
    """Build the (pathway x gene) membership for a collection on the gene bundle's axis.

    This is what makes the gene bundle collection-independent: membership is rebuilt here at
    test time for whatever ``collections`` is requested, rather than baked into the bundle,
    so one gene bundle serves Reactome, Hallmark, C6, ... Returns membership (float64),
    pathway names, sizes (float64), and a name->collection map.
    """
    gene_sets, collection_source = load_collections(msigdb_dir, collections)
    membership, names, sizes, _coverage = build_membership(gene_sets, symbols, min_members)
    collection_of = {name: collection_source[name] for name in names}
    return membership.astype(np.float64), names, sizes.astype(np.float64), collection_of


def measured_gene_sets(membership, names, symbols):
    """Rebuild {pathway: set(measured member symbols)} from the membership matrix.

    Membership already restricts each pathway to the measured gene axis, so this gives
    exactly the sets ORA uses with the measured-gene background.
    """
    symbols = np.asarray(symbols)
    return {
        name: set(symbols[membership[i] > 0].tolist()) for i, name in enumerate(names)
    }


# ---------------------------------------------------------------------------
# Orchestration and CLI
# ---------------------------------------------------------------------------


def run_layer_c(
    bundle,
    output_dir,
    collections=("c2.cp.reactome",),
    n_perm=1000,
    seed=0,
    ora_top_n=100,
    min_members=10,
    skip_gsea=False,
    msigdb_dir=MSIGDB_DIR,
    gsea_threads=1,
    tail="gpd",
    n_jobs=1,
):
    ig_symbols = bundle["ig_symbols"]
    path_gradient_symbols = bundle["path_gradient_symbols"]
    symbols = bundle["symbols"]
    # Rebuild membership for the requested collection on the gene bundle's axis (the bundle
    # is collection-independent; membership is never persisted). min_members is the single
    # size floor used both here and by ORA below.
    membership, names, sizes, collection_of = build_discovery_membership(
        symbols, collections, msigdb_dir, min_members
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        f"Built membership for {len(names)} sets from {', '.join(collections)} "
        f"(min_members={min_members}) on {len(symbols)} measured genes"
    )

    # 1. Permutation null on the paper's own pathway statistic.
    logger.info(f"Permutation null for {len(names)} pathways ({n_perm} permutations)")
    null = permutation_null(
        ig_symbols, membership, sizes, n_perm, seed, tail=tail, n_jobs=n_jobs
    )
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
            "perm_gpd_shape": null["shape"],
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
            "perm_gpd_shape",
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


DEFAULT_CONFIG = "joint_fusion/config/config_checkpoint_2026-04-07-04-58-17_fold1.yaml"


def resolve_gene_bundle_and_out(opt):
    """Resolve (gene_bundle_path, output_dir).

    The gene bundle is collection-independent and shared, so it lives one level above the
    per-collection output dir (<base>/gene_attribution_bundle.npz). --out-dir is the
    per-collection folder that holds this collection's pathway_scores.csv and receives the
    test outputs.
    """
    if opt.out_dir:
        out_dir = Path(opt.out_dir)
    else:
        from joint_fusion.config.config_manager import ConfigManager

        config = ConfigManager.load_config(opt.config)
        out_dir = Path(config.testing.output_base_dir) / "pathway_interpret"

    if opt.gene_bundle:
        gene_bundle = Path(opt.gene_bundle)
    else:
        # pathway_interpret writes the shared gene bundle to the base dir (out_dir's parent).
        gene_bundle = out_dir.parent / GENE_BUNDLE_NAME
    return gene_bundle, out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Fold-1 config by default; used to locate the output folder "
        "when --out-dir is not given.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Per-collection output folder (holds this collection's pathway_scores.csv "
        "and receives the test outputs). Overrides --config.",
    )
    parser.add_argument(
        "--gene-bundle",
        type=str,
        default=None,
        help="Path to the collection-independent gene_attribution_bundle.npz. Defaults to "
        "<out-dir>/../gene_attribution_bundle.npz (the shared base dir where "
        "pathway_interpret writes it).",
    )
    parser.add_argument(
        "--collections",
        type=str,
        default="c2.cp.reactome",
        help="Gene-set collection(s) to test, rebuilt on the gene bundle's axis at test "
        "time (comma-separated). Run ONE collection per invocation to keep each its own "
        "FDR family -- e.g. c2.cp.reactome, or h.all, or c6.all. Do not pool.",
    )
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument(
        "--tail",
        choices=("gpd", "empirical"),
        default="gpd",
        help="Tail model for the permutation p-value below the empirical floor. "
        "gpd (default) is the peaks-over-threshold extrapolation with the tail shape fixed "
        "to xi=0 (the exponential member of the GPD family): conservative under the "
        "statistic's bounded support and free of the free-shape MLE's endpoint "
        "ill-conditioning. empirical is the floored ground-truth p, kept for comparison.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ora-top-n", type=int, default=100)
    parser.add_argument("--min-members", type=int, default=10)
    parser.add_argument(
        "--msigdb-dir",
        type=str,
        default=MSIGDB_DIR,
        help="MSigDB asset folder holding the .gmt collections.",
    )
    parser.add_argument(
        "--gsea-threads",
        type=int,
        default=1,
        help="Worker threads passed to GSEApy prerank.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Processes for the permutation null (each permutation is independent). "
        "1 = serial (default), -1 = all cores, N = N workers. Note the serial default is "
        "ALREADY core-parallel: each permutation is one large matmul that BLAS "
        "multithreads across all cores, so serial is typically as fast as (or faster "
        "than) multiprocessing here. --jobs auto-pins BLAS to one thread per worker to "
        "avoid oversubscription; it is worth setting only if numpy is linked against a "
        "single-threaded BLAS.",
    )
    parser.add_argument("--skip-gsea", action="store_true")
    opt = parser.parse_args()

    collections = tuple(c.strip() for c in opt.collections.split(",") if c.strip())
    gene_bundle_path, out_dir = resolve_gene_bundle_and_out(opt)
    if not gene_bundle_path.exists():
        raise SystemExit(
            f"Gene bundle not found: {gene_bundle_path}\n"
            "Run joint_fusion.testing.pathway_interpret first (it writes the "
            "collection-independent gene_attribution_bundle.npz)."
        )
    logger.info(f"Loading gene bundle: {gene_bundle_path}")
    bundle = load_gene_bundle(gene_bundle_path)
    os.makedirs(out_dir, exist_ok=True)
    run_layer_c(
        bundle,
        output_dir=str(out_dir),
        collections=collections,
        n_perm=opt.n_perm,
        seed=opt.seed,
        ora_top_n=opt.ora_top_n,
        min_members=opt.min_members,
        skip_gsea=opt.skip_gsea,
        msigdb_dir=opt.msigdb_dir,
        gsea_threads=opt.gsea_threads,
        tail=opt.tail,
        n_jobs=opt.jobs,
    )
    logger.info(
        "\nDone. See pathway_scores_with_stats.csv, gsea_prerank.csv, and ora_*.csv."
    )


if __name__ == "__main__":
    main()
