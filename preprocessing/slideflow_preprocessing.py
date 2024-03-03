# Trying slideflow to do tiling, stain normalization etc for preparation for NN

import numpy as np
import slideflow as sf
sf.about()

work_dir = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/data_3'
slides_dir = '/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_WSI_RNASeq_clinical/WSI/'

# https://slideflow.dev/tutorial6/
# wsi = sf.WSI('/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_WSI_RNASeq_clinical/WSI/TCGA-05-4245-01Z-00-DX1.36ff5403-d4bb-4415-b2c5-7c750d655cde.svs',
#              tile_px=299,
#              tile_um=302)

# create a new project
P = sf.create_project(
    root=work_dir,
    # cfg=sf.project.LungAdenoSquam, # this throws error, so instead going directly with my own slides
    slides = slides_dir,
    download=True
)

# load an existing project
# P = sf.load_project(work_dir)

# prepare a dataset of image tiles
dataset = P.dataset(
    tile_px=256,
    tile_um='9.9x' # Most TCGA samples seem to be of x9.9
    # tile_um=128
)

print("current tiles: ", dataset.num_tiles)

# https://slideflow.dev/dataset/#slideflow.Dataset.extract_tiles
# https://github.com/jamesdolezal/slideflow/blob/master/slideflow/norm/__init__.py
dataset.extract_tiles(qc='both', #  ‘otsu’, ‘blur’, ‘both’, or None
                      save_tiles=True,
                      save_tfrecords=True,
                      skip_extracted=False,
                      max_tiles = 200,
                      normalizer='macenko_fast')



# P.extract_tiles(
#     tile_px=256,
#     tile_um='10x'
# )

dataset.summary()
