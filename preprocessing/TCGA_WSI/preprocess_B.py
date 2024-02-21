import wsi_preprocessing as pp
from pdb import set_trace

LEVEL = 2 # pyramid level for the WSI

# slides = pp.list_slides("/mnt/c/Users/tnandi/Downloads/sample_svs/")
slides = pp.list_slides("/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_svs/")
pp.save_slides_mpp_otsu(slides, "slides_mpp_otsu.csv", level=LEVEL)

# this may take a few minutes (depending on your local machine, of course)
pp.run_tiling("slides_mpp_otsu.csv", "tiles.csv")

# pp.calculate_filters("slides_mpp_otsu.csv", "", "tiles_filters.csv")
