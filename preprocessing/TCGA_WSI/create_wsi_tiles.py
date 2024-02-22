import wsi_preprocessing as pp
from pdb import set_trace

# using a modified version of https://github.com/lucasrla/wsi-preprocessing
# modified code available at C:\Users\tnandi\Downloads\wsi-preprocessing-master\wsi-preprocessing-master
# on Dell laptop use the wsi_env conda env with python3.6
# Note: the cleanup step (removing images with mostly background) takes too long, so it is done separately using remove_background.py

LEVEL = 2 # pyramid level for the WSI
# can set the tile size in tiles.py in the wsi_preprocessing codebase (do a pip install . after editing)

slides = pp.list_slides("/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_WSI_RNASeq_clinical/WSI/")
# slides = pp.list_slides("/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_svs/")
pp.save_slides_mpp_otsu(slides, "slides_mpp_otsu.csv", level=LEVEL)

# this may take a few minutes (depending on your local machine, of course)
pp.run_tiling("slides_mpp_otsu.csv", "tiles.csv")

# pp.calculate_filters("slides_mpp_otsu.csv", "", "tiles_filters.csv")
