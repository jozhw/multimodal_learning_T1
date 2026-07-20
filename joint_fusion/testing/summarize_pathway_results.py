"""
Summarize pathway_interpret.py outputs for manuscript review.

This is intentionally dependency-free: it reads pathway_scores.csv and the saved
per-patient score matrices with the Python standard library so it can run on a
local machine that does not have numpy/pandas/scipy installed.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DEFAULT_DIR = Path(
    "checkpoints/checkpoint_2026-04-07-04-58-17/test_results/"
    "best_model_fold_1/pathway_interpret"
)


LUAD_FAMILIES = {
    "cell_cycle_RB_E2F_G2M": {
        "terms": {
            "CELL_CYCLE",
            "E2F",
            "G2M",
            "MITOTIC",
            "MITOSIS",
            "CYCLIN",
            "CDK",
            "CHECKPOINT",
            "KINESIN",
            "KINESINS",
            "COHESIN",
            "REPLICATION",
            "REPLISOME",
            "TELOPHASE",
            "CYTOKINESIS",
            "MCM",
            "PLK",
            "POLO",
            "G0",
            "G1",
            "G2",
        },
        "description": "cell-cycle/proliferation axis, including RB/E2F/G2M programs",
    },
    "RTK_RAS_MAPK": {
        "terms": {
            "KRAS",
            "NRAS",
            "HRAS",
            "RAS",
            "MAPK",
            "RAF",
            "MEK",
            "ERK",
            "EGFR",
            "ERBB",
            "MET",
            "ALK",
            "RET",
            "ROS1",
            "NTRK",
            "RTK",
        },
        "description": "RTK/RAS/RAF/MAPK signaling axis",
    },
    "PI3K_AKT_mTOR": {
        "terms": {"PI3K", "PI_3_KINASE", "AKT", "MTOR", "MTORC1", "PTEN", "PIK3CA"},
        "description": "PI3K/AKT/mTOR signaling axis",
    },
    "TP53_DNA_damage_apoptosis": {
        "terms": {"P53", "TP53", "DNA_DAMAGE", "ATM", "ATR", "CHK1", "CHK2", "APOPTOSIS", "APOPTOTIC"},
        "description": "p53, DNA-damage checkpoint, and apoptosis programs",
    },
    "ECM_EMT_invasion": {
        "terms": {
            "EXTRACELLULAR_MATRIX",
            "ECM",
            "COLLAGEN",
            "FIBRONECTIN",
            "INTEGRIN",
            "SYNDECAN",
            "FOCAL_ADHESION",
            "ADHESION",
            "EMT",
            "MESENCHYMAL",
        },
        "description": "extracellular matrix, EMT, adhesion, and invasion biology",
    },
    "immune_inflammation": {
        "terms": {
            "IMMUNE",
            "IMMUNODEFICIENCY",
            "CHEMOKINE",
            "CHEMOKINES",
            "CYTOKINE",
            "CYTOKINES",
            "T_CELL",
            "B_CELL",
            "INTERFERON",
            "INFLAMMATION",
            "INFLAMMATORY",
            "TNF",
            "IL6",
            "ANTIGEN",
            "MHC",
            "COMPLEMENT",
        },
        "description": "immune, inflammatory, and antigen-presentation programs",
    },
    "lung_lineage_surfactant": {
        "terms": {"LUNG", "NSCLC", "ADENOCARCINOMA", "SURFACTANT", "ALVEOLAR"},
        "description": "lung/NSCLC/lung-lineage or surfactant biology",
    },
    "metabolism_STK11_AMPK_lipid": {
        "terms": {
            "LIPOPROTEIN",
            "CHOLESTEROL",
            "FATTY_ACID",
            "GLYCOLYSIS",
            "HYPOXIA",
            "OXIDATIVE",
            "METABOLISM",
            "AMPK",
            "LKB1",
            "STK11",
        },
        "description": "metabolic, lipid, hypoxia, STK11/LKB1/AMPK-related biology",
    },
    "MYC": {
        "terms": {"MYC"},
        "description": "MYC transcriptional programs",
    },
    "TGF_WNT_NOTCH_HIPPO": {
        "terms": {"TGF", "TGFB", "WNT", "BETA_CATENIN", "NOTCH", "HIPPO", "YAP", "TAZ"},
        "description": "TGF-beta, WNT/beta-catenin, Notch, and Hippo/YAP/TAZ signaling",
    },
    "KEAP1_NFE2L2_NRF2": {
        "terms": {"KEAP1", "NFE2L2", "NRF2"},
        "description": "KEAP1/NFE2L2/NRF2 oxidative-stress axis",
    },
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def norm_name(name):
    return str(name).upper().replace(".", "_").replace("-", "_").replace(" ", "_")


def matches_term(normalized, term):
    term = term.upper()
    if "_" in term:
        return term in normalized
    tokens = [token for token in normalized.split("_") if token]
    return term in tokens


def luad_families(pathway):
    normalized = norm_name(pathway)
    hits = []
    for family, spec in LUAD_FAMILIES.items():
        if any(matches_term(normalized, term) for term in spec["terms"]):
            hits.append(family)
    return hits


def read_matrix(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = header[1:]
        cols = {name: [] for name in names}
        for row in reader:
            for name, value in zip(names, row[1:]):
                try:
                    cols[name].append(float(value))
                except ValueError:
                    cols[name].append(float("nan"))
    return names, cols


def quantile(sorted_values, q):
    if not sorted_values:
        return None
    idx = round((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def distribution_stats(values):
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {}
    finite_sorted = sorted(finite)
    return {
        "n": len(finite),
        "frac_positive": sum(v > 0 for v in finite) / len(finite),
        "median": statistics.median(finite),
        "q1": quantile(finite_sorted, 0.25),
        "q3": quantile(finite_sorted, 0.75),
        "min": min(finite),
        "max": max(finite),
    }


def count_summary(rows, top_ns):
    summary = {}
    for top_n in top_ns:
        subset = rows[:top_n]
        strict_lung = [r for r in subset if r.get("lung_named_in_msigdb") == "True"]
        family_rows = [r for r in subset if r["luad_families"]]
        family_counts = {family: 0 for family in LUAD_FAMILIES}
        for row in subset:
            for family in row["luad_families"]:
                family_counts[family] += 1
        summary[f"top_{top_n}"] = {
            "n_pathways": len(subset),
            "strict_lung_named_count": len(strict_lung),
            "luad_family_annotated_count": len(family_rows),
            "family_counts": family_counts,
        }
    return summary


def write_annotation_csv(path, rows):
    fieldnames = [
        "rank",
        "ranking_statistic",
        "pathway",
        "collection",
        "direction",
        "summed_abs_vanilla_gradient",
        "mean_abs_vanilla_gradient",
        "summed_abs_path_gradient",
        "mean_abs_path_gradient",
        "summed_abs_ig",
        "mean_abs_ig",
        "mean_gradient",
        "mean_path_gradient",
        "mean_vanilla_gradient",
        "lung_named_in_msigdb",
        "lung_cancer_subtype_from_name",
        "luad_family_annotated",
        "luad_families",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ";".join(row[key])
                        if key == "luad_families"
                        else row.get(key, "")
                    )
                    for key in fieldnames
                }
            )


def markdown_table(rows, fields):
    lines = ["| " + " | ".join(label for label, _ in fields) + " |"]
    lines.append("|" + "|".join(["---"] * len(fields)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(fn(row)) for _, fn in fields) + " |")
    return "\n".join(lines)


def fmt_float(value, digits=3):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def write_markdown(path, rows, summary, dist_tables, output_dir, metadata):
    top20 = rows[:20]
    lung_rows = [r for r in rows if r.get("lung_named_in_msigdb") == "True"]
    attribution = metadata.get("attribution", {})
    ranking_column = attribution.get("ranking_column") or "summed_abs_ig"
    ranking_description = (
        attribution.get("ranking_description")
        or "sum over patients of |mean member-gene IG attribution|"
    )
    ranking_label = ranking_column.replace("_", " ")
    top_fields = [
        ("Rank", lambda r: r["rank"]),
        ("Pathway", lambda r: r["pathway"].replace("_", "\\_")),
        ("Direction", lambda r: r["direction"]),
        (ranking_label, lambda r: fmt_float(r.get(ranking_column), 4)),
        ("sum abs IG", lambda r: fmt_float(r.get("summed_abs_ig"), 4)),
        ("LUAD family annotation", lambda r: ", ".join(r["luad_families"]) or "-"),
    ]
    lung_fields = [
        ("Rank", lambda r: r["rank"]),
        ("Pathway", lambda r: r["pathway"].replace("_", "\\_")),
        ("Direction", lambda r: r["direction"]),
        ("sum abs IG", lambda r: fmt_float(r["summed_abs_ig"], 4)),
        ("Subtype token", lambda r: r.get("lung_cancer_subtype_from_name", "")),
    ]
    dist_fields = [
        ("Pathway", lambda r: r["pathway"].replace("_", "\\_")),
        ("IG frac +", lambda r: fmt_float(r["ig"]["frac_positive"], 3)),
        ("IG median", lambda r: fmt_float(r["ig"]["median"], 4)),
        ("Path grad frac +", lambda r: fmt_float(r["path_gradient"]["frac_positive"], 3)),
        ("Path grad median", lambda r: fmt_float(r["path_gradient"]["median"], 4)),
        ("Vanilla grad frac +", lambda r: fmt_float(r["vanilla_gradient"]["frac_positive"], 3)),
        ("Vanilla grad median", lambda r: fmt_float(r["vanilla_gradient"]["median"], 4)),
    ]

    lines = [
        "# Pathway Interpretability Results Review",
        "",
        f"Result directory: `{output_dir}`",
        "",
        "## What The Current Outputs Show",
        "",
        f"- `pathway_scores.csv` contains {len(rows)} scored pathways.",
        f"- Primary ranking uses `{ranking_column}`: {ranking_description}.",
        "- Direction is taken from path-averaged gradients, not signed IG.",
        "- `pathway_ig_magnitude_scores.csv` keeps the separate IG magnitude ranking when available.",
        "- The paper-style patient distribution is present: the pathway score matrices have one pathway score per patient, and the beeswarm PNGs plot those patient-level values.",
        "",
        "## LUAD-Specific Counts",
        "",
        "Two definitions are kept separate:",
        "",
        "- **Strict lung-named:** MSigDB pathway name itself contains lung/NSCLC/adenocarcinoma.",
        "- **Canonical LUAD-family annotation:** pathway name maps to a broad LUAD-relevant biology family such as cell cycle, RTK/RAS/MAPK, ECM/EMT, surfactant, immune, metabolism, etc. This is annotation only; it is not used for ranking.",
        "",
    ]
    for key, value in summary.items():
        lines.append(
            f"- {key.replace('_', ' ')}: strict lung-named "
            f"{value['strict_lung_named_count']}/{value['n_pathways']}; "
            f"canonical LUAD-family annotated "
            f"{value['luad_family_annotated_count']}/{value['n_pathways']}."
        )
    lines.extend(
        [
            "",
            "## Top 20 Pathways",
            "",
            markdown_table(top20, top_fields),
            "",
            "## Strict Lung/NSCLC-Named Pathways",
            "",
            markdown_table(lung_rows, lung_fields),
            "",
            "## Patient-Level Distribution Check",
            "",
            "For the original-paper analogy, the gradient distributions are the direct analog. In this repo:",
            "",
            "- `pathway_steyaert_vanilla_gradient_beeswarm.png` is the closest direct Steyaert-style plot.",
            "- `pathway_steyaert_path_gradient_beeswarm.png` is the IG-path direction companion.",
            "- `pathway_ig_magnitude_beeswarm.png` is the signed IG attribution distribution for the IG-ranked pathways.",
            "- `pathway_expression_colored_ig_beeswarm.png` is an extra expression-coloured view, not the Steyaert-style color convention.",
            "",
            markdown_table(dist_tables, dist_fields),
            "",
            "## Missing Statistical Evidence Layer",
            "",
            "`pathway_tests.py` has not been run for this output directory: `pathway_scores_with_stats.csv`, `gsea_prerank.csv`, and `ora_*.csv` are absent. Run on the remote environment with SciPy/NumPy installed:",
            "",
            "```bash",
            "python -m joint_fusion.testing.pathway_tests \\",
            f"  --dir {output_dir} \\",
            "  --n-perm 1000",
            "```",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--top-n", type=int, default=20)
    opt = parser.parse_args()

    output_dir = Path(opt.dir)
    rows = read_csv(output_dir / "pathway_scores.csv")
    metadata = read_json(output_dir / "run_metadata.json")
    for row in rows:
        families = luad_families(row["pathway"])
        row["luad_families"] = families
        row["luad_family_annotated"] = bool(families)

    summary = count_summary(rows, top_ns=(20, 50, 100, 200, 500, 1000))
    summary_path = output_dir / "pathway_luad_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_annotation_csv(output_dir / "pathway_luad_family_annotations.csv", rows)

    matrices = {
        "ig": "pathway_scores_matrix.csv",
        "path_gradient": "pathway_path_gradient_scores_matrix.csv",
        "vanilla_gradient": "pathway_vanilla_gradient_scores_matrix.csv",
    }
    matrix_stats = {}
    for label, filename in matrices.items():
        names, cols = read_matrix(output_dir / filename)
        matrix_stats[label] = {
            name: distribution_stats(cols[name])
            for name in names[: opt.top_n]
        }
    dist_tables = []
    for row in rows[: min(10, opt.top_n)]:
        pathway = row["pathway"]
        dist_tables.append(
            {
                "pathway": pathway,
                "ig": matrix_stats["ig"].get(pathway, {}),
                "path_gradient": matrix_stats["path_gradient"].get(pathway, {}),
                "vanilla_gradient": matrix_stats["vanilla_gradient"].get(pathway, {}),
            }
        )

    write_markdown(
        output_dir / "pathway_interpret_results_review.md",
        rows,
        summary,
        dist_tables,
        output_dir,
        metadata,
    )
    print(output_dir / "pathway_interpret_results_review.md")
    print(output_dir / "pathway_luad_family_annotations.csv")
    print(summary_path)


if __name__ == "__main__":
    main()
