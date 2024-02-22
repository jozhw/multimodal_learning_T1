import openslide
import os
from pprint import pprint
from pdb import set_trace

# on Dell laptop, use the pytorch conda env with python3.10

svs_dir = '/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_svs/'
svs_files = [svs_dir + file for file in os.listdir(svs_dir) if file.endswith('.svs')]
# set_trace()

for file in svs_files:
    slide = openslide.open_slide(file)
    properties = slide.properties
    # pprint(properties)
    for key, value in properties.items():
        print(f"{key}: {value}")

    print("Number of levels: ", slide.level_count)
    print("Highest resolution (W, H), i.e., resolution at level 0: ", slide.dimensions)
    print("Resolution (W,H) at all levels: ", slide.level_dimensions)
    print("Downsample factors for each level: ", slide.level_downsamples)
    print("Optical magnification: x", properties['aperio.AppMag'])

    # get a thumbnail of our WSI (resized image)
    slide_thumb = slide.get_thumbnail(size=(400, 400))
    print("actual thumbnail size: ", slide_thumb.size)
    slide_thumb.show()

    # to choose an image size for a certain level
    # level3_img = slide.read_region((0, 0), 2, (1500, 1000)) # (location, level, size)
    # level3_img.size

    # magnification_levels_mpp = float(properties['aperio.MPP'])  # magnification level in microns per pixel
    # downsamples = [float(properties[f'openslide.level[{i}].downsample']) for i in range(int(properties['openslide.level-count']))]
    # #get the magnification at each level
    # magnifications = [magnification_levels_mpp / downsample for downsample in downsamples]
    # print("mpp (microns per pixel): ", magnification_levels_mpp)
    # for i, mag in enumerate(magnifications):
    #     print(f"Magnification at level {i}: {mag:.2f}x")
