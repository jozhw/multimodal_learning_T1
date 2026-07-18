"""
fetch_msigdb.py

Download the MSigDB gene-set (.gmt) and Ensembl-ID annotation (.chip) files that
the pathway interpretability step needs, into assets/msigdb/.

Run this ONCE on a machine with internet; the files are then vendored in the repo
and every downstream step (pathway_interpret.py) runs fully offline on the cluster.

Why a .chip file rather than gget: MSigDB gene sets are keyed on HGNC symbols while
the model's gene axis is versioned Ensembl IDs. The .chip file is MSigDB's own
Ensembl -> symbol map, so the mapping and the gene sets come from the same release
and cannot drift apart. Resolving ~9k genes one-at-a-time through gget.info instead
would take hours and mixes annotation sources.

  python -m joint_fusion.fetch_msigdb

Files (MSigDB 2026.1.Hs):
  h.all.*.symbols.gmt              Hallmark, 50 sets -- the readable headline
  c2.cp.reactome.*.symbols.gmt     Reactome -- the paper's collection (they used v7.5)
  c2.cp.kegg_legacy.*.symbols.gmt  KEGG legacy canonical pathways, including disease maps
  c2.cp.kegg_medicus.*.symbols.gmt KEGG Medicus disease/drug/pathway-oriented sets
  c6.all.*.symbols.gmt             oncogenic signatures (KRAS/EGFR/TP53 perturbations)
  Human_Ensembl_Gene_ID_MSigDB.*.chip   Ensembl -> HGNC symbol
"""

import argparse
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

MSIGDB_RELEASE = "2026.1.Hs"
GMT_BASE = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{MSIGDB_RELEASE}"
CHIP_BASE = "https://data.broadinstitute.org/gsea-msigdb/msigdb/annotations/human"

FILES = {
    f"h.all.v{MSIGDB_RELEASE}.symbols.gmt": f"{GMT_BASE}/h.all.v{MSIGDB_RELEASE}.symbols.gmt",
    f"c2.cp.reactome.v{MSIGDB_RELEASE}.symbols.gmt": f"{GMT_BASE}/c2.cp.reactome.v{MSIGDB_RELEASE}.symbols.gmt",
    f"c2.cp.kegg_legacy.v{MSIGDB_RELEASE}.symbols.gmt": f"{GMT_BASE}/c2.cp.kegg_legacy.v{MSIGDB_RELEASE}.symbols.gmt",
    f"c2.cp.kegg_medicus.v{MSIGDB_RELEASE}.symbols.gmt": f"{GMT_BASE}/c2.cp.kegg_medicus.v{MSIGDB_RELEASE}.symbols.gmt",
    f"c6.all.v{MSIGDB_RELEASE}.symbols.gmt": f"{GMT_BASE}/c6.all.v{MSIGDB_RELEASE}.symbols.gmt",
    f"Human_Ensembl_Gene_ID_MSigDB.v{MSIGDB_RELEASE}.chip": f"{CHIP_BASE}/Human_Ensembl_Gene_ID_MSigDB.v{MSIGDB_RELEASE}.chip",
}

DEFAULT_ASSET_DIR = "assets/msigdb"


def fetch(asset_dir=DEFAULT_ASSET_DIR, overwrite=False):
    os.makedirs(asset_dir, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(asset_dir, name)
        if os.path.exists(dest) and not overwrite:
            logger.info(f"exists, skipping: {dest}")
            continue
        logger.info(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
        logger.info(f"  -> {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")

    with open(os.path.join(asset_dir, "VERSION"), "w") as fh:
        fh.write(f"MSigDB {MSIGDB_RELEASE}\n")
    logger.info(f"MSigDB {MSIGDB_RELEASE} assets ready in {asset_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", default=DEFAULT_ASSET_DIR)
    parser.add_argument("--overwrite", action="store_true")
    opt = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch(opt.asset_dir, opt.overwrite)


if __name__ == "__main__":
    main()
