# Using slideflow to do tiling, stain normalization etc for preparation for NN
# on polaris: conda activate /lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/slideflow_env

import numpy as np
from pdb import set_trace
import slideflow as sf
# import openslide
from slideflow.slide import qc
sf.about()

#work_dir = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/data_3'
# work_dir = ('/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/')
work_dir = ('/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_20X_1000tiles/')

#slides_dir = '/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_WSI_RNASeq_clinical/WSI/'
slides_dir = ('/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_single_sample_per_patient/')
# slides_dir = ('/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_selected_samples/')

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
# since 20X is not available for most WSIs, we prescribe the tile size for that magnification level and downsample from 40X
dataset = P.dataset(
    tile_px=256,
    # tile_um='40x' #'9.9x' # Most TCGA samples seem to be of x9.9
    tile_um=128 # tile size in microns = mpp * pixel size [for 20X, tile_um = 0.5005 * 256 = 128.12]
)

print("current tiles: ", dataset.num_tiles)

# https://slideflow.dev/dataset/#slideflow.Dataset.extract_tiles
# https://github.com/jamesdolezal/slideflow/blob/master/slideflow/norm/__init__.py
dataset.extract_tiles(qc='otsu', #  ‘otsu’, ‘blur’, ‘both’, or None  # qc.GaussianV2()
                      save_tiles=True,
                      save_tfrecords=True,
                      skip_extracted=False,
                      max_tiles = 1000,) # was 200  # used 500 for the slides with pen marks to extract enough number of clean tiles
                      # normalizer='macenko_fast')

# whitespace_fraction,

dataset.summary()

