# code to keep 200 randomly chosen tiles from the ones extracted for WSIs with penmarks (since these were manually done, instead of WC filtering, the number of tiles can vary a lot)
import os
import shutil
import random
from tqdm import tqdm

base_dir = '/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/'
source_dir = base_dir + "256px_9.9x_clean_from_penmarks"
dest_dir = base_dir + "256px_9.9x_clean_from_penmarks_200tiles"
n_samples = 200

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    if os.path.isdir(subdir_path):
        # create the same subdirectory in the destination directory
        dest_subdir_path = os.path.join(dest_dir, subdir)
        if not os.path.exists(dest_subdir_path):
            os.makedirs(dest_subdir_path)

        jpg_files = [f for f in os.listdir(subdir_path) if f.endswith('.jpg')]

        # randomly select "n_samples" jpg files
        selected_files = random.sample(jpg_files, min(n_samples, len(jpg_files)))

        # copy selected files to the destination subdirectory
        # for file in selected_files:
        for file in tqdm(selected_files, desc=f'Copying files from {subdir}'):
            src_file_path = os.path.join(subdir_path, file)
            dest_file_path = os.path.join(dest_subdir_path, file)
            shutil.copy2(src_file_path, dest_file_path)

print("Completed sampling and transfer")
