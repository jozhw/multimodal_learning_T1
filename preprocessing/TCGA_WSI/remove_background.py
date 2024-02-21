import cv2
import numpy as np
import os
import shutil
from pdb import set_trace
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def process_images(source_dir, dest_dir, tissue_threshold=0.4):
    """
    Process images in the source directory, moving images with less than the specified
    foreground threshold of tissue to the destination directory.

    :param source_dir: Directory containing the images to process.
    :param dest_dir: Directory where images with insufficient tissue will be moved.
    :param tissue_threshold: Minimum fraction of the image that must be foreground (tissue).
    """
    os.makedirs(dest_dir, exist_ok=True)

    file_index = 0
    for filename in os.listdir(source_dir):
        if not filename.lower().endswith('.png'):
            continue

        file_path = os.path.join(source_dir, filename)

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # apply Gaussian blur to reduce noise and improve the thresholding
        # blurred_image = gray_image #cv2.GaussianBlur(gray_image, (5, 5), 0)
        # #use Otsu thresholding to separate the tissue from the background
        # # for this binary thresholding, the pixels above threshold will be set to 255, and those below threshold to 0
        # ret, otsu_thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #
        # # create a mask where the tissue regions are white and the background is black
        # mask = otsu_thresh == 255
        #
        # # create an output image that will contain the tissue with a white background
        # tissue = np.zeros_like(gray_image)
        # tissue[mask] = 255#image[mask]
        # cv2.imwrite(f"tissue_{file_index}.png", tissue)
        # # cv2.imshow('Tissue', tissue)
        # # set_trace()


        # calculate the tissue percentage
        # set_trace()
        dark_threshold = 150 # pixels with grayscale value below this are considered dark (representing tissues)
        mask = gray_image < dark_threshold
        tissue = 255*np.ones_like(gray_image)
        tissue[mask] = 5 # a low (dark) value
        # cv2.imwrite(f"tissue_{file_index}.png", tissue)

        tissue_pixel_count = np.sum(gray_image < dark_threshold)
        tissue_pct = tissue_pixel_count / gray_image.size

        # check if the image has sufficient fraction of tissue compared to the background
        print("++++ filename +++: ", filename)
        print("tissue_pixel_count: ", tissue_pixel_count)
        print("total_pixel_count: ", gray_image.size)
        print("tissue_pct:", tissue_pct)
        if tissue_pct > tissue_threshold:
            dest_path = os.path.join(dest_dir, filename)
            # shutil.move(file_path, dest_path)
            shutil.copy(file_path, dest_path)
            print(f"Moved: {filename}")
        file_index += 1
        # set_trace()

if __name__ == "__main__":
    base_dir = "/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/TCGA_WSI/"
    source_directory = "TCGA-78-7166-01Z-00-DX1.d19ad2d9-b006-4a13-ba54-2fec234c2373.svs_mpp-0.5_crop108-181_files/0/"
    destination_directory = "TCGA-78-7166-01Z-00-DX1.d19ad2d9-b006-4a13-ba54-2fec234c2373.svs_mpp-0.5_crop108-181_files/0_filtered/"
    process_images(base_dir + source_directory, base_dir + destination_directory)
