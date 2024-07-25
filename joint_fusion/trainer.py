import pandas as pd
import torch
from pdb import set_trace
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
# import datasets
# from models import
# from train_test import train_nn
import argparse
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
import ast
from collections import Counter
# from data_mapping import create_data_mapping
# for profiling
# torch.autograd.profiler.profile(enabled=True)
from torch.profiler import profile, record_function, ProfilerActivity

# on Dell laptop (activate conda env 'pytorch_py3p10' and use 'python trainer.py')
# on Polaris, activate env /lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/pytorch_py3p10

parser = argparse.ArgumentParser()
parser.add_argument('--create_new_data_mapping', type=str, default=True, help="whether to create new data mapping or use existing one")
parser.add_argument('--input_mapping_data_path', type=str,
                    # default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/joint_fusion/', # on laptop
                    default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/joint_fusion/', # on Polaris
                    help='Path to input mapping data file')
parser.add_argument('--input_path', type=str,
                    # default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/', # on laptop
                    default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/', # on Polaris
                    help='Path to input data files')
parser.add_argument('--input_wsi_path', type=str,
                    # default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles/tiles/256px_9.9x/combined_tiles/', # on laptop
                    # default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_otsu_B/tiles/256px_9.9x/combined_tiles/', # on Polaris
                    default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/256px_9.9x/combined_tiles/',
                    help='Path to input WSI tiles')
parser.add_argument('--input_wsi_embeddings_path', type=str,
                    # default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/early_fusion_inputs/', # on laptop
                    default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_otsu_B/tiles/256px_9.9x/combined_tiles/', # on Polaris
                    help='Path to WSI embeddings generated from pretrained pathology foundation model')
parser.add_argument('--checkpoint_dir', type=str,
                    default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/joint_fusion/checkpoint_2024-04-20-08-43-52/',
                    help='Path to the checkpoint files from trained VAE for omic embedding generation')
# parser.add_argument('--output_path', type=str, default='results/output.txt', help='Path to output results file')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--lr_decay_iters', type=int, default=100, help='Learning rate decay steps')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--input_size_wsi', type=int, default=256, help="input_size for path images")
parser.add_argument('--embedding_dim_wsi', type=int, default=384, help="embedding dimension for WSI")
parser.add_argument('--embedding_dim_omic', type=int, default=256, help="embedding dimension for omic")
parser.add_argument('--input_mode', type=str, default="wsi_omic", help="wsi, omic, wsi_omic")
parser.add_argument('--fusion_type', type=str, default="joint_omic", help="early, late, joint, joint_omic, unimodal") # "joint_omic" only trains the omic embedding generator jointly with the downstream combined model
parser.add_argument('--profile', type=str, default=False, help="whether to profile or not")
parser.add_argument('--use_mixed_precision', type=str, default=False, help="whether to use mixed precision calculations")
parser.add_argument('--use_gradient_accumulation', type=str, default=False, help="whether to use gradient accumulation")

opt = parser.parse_args()

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
torch.backends.cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

if opt.create_new_data_mapping:
    # create data mappings
    # read the file containing gene expression and tile image locations for the TCGA-LUAD samples (mapped_data_16March)
    # mapping_df = pd.read_csv(opt.input_path + "mapped_data_21March.csv")

    # mapping_df = pd.read_json(opt.input_path + "mapped_data_8may.json", orient='index') # file generated by create_image_molecular_mapping.py
    mapping_df = pd.read_json(opt.input_path + "mapped_data_23july.json", orient='index')
    print("Total number of samples: ", mapping_df.shape[0])
    ids_with_wsi = mapping_df[mapping_df['tiles'].map(len) > 0].index.tolist()
    rnaseq_df = pd.DataFrame(mapping_df['rnaseq_data'].to_list(), index=mapping_df.index).transpose()
    print("Are there nans in rnaseq_df: ", rnaseq_df.isna().any().any())
    # df containing entries where both WSI and rnaseq data are available
    mapping_df = mapping_df.loc[ids_with_wsi]
    print("Total number of samples where both rnaseq and wsi data are available: ", mapping_df.shape[0])
    # set_trace()
    # remove rows where the number of tiles is different from the standard (to avoid length mismatch issues during batching)
    # edit it to keep only 200 random slides among all the available ones
    mask = mapping_df['tiles'].apply(len) == 200
    mapping_df = mapping_df[mask]

    # set_trace()
    # # check if there are empty (or unexpected number of) rnaseq entries
    # # check why this is happening
    # rnaseq_dict_sizes = mapping_df['rnaseq_data'].apply(lambda x: len(x))
    # print("rnaseq_dict_sizes.value_counts(): ", rnaseq_dict_sizes.value_counts())
    # mapping_df = mapping_df[mapping_df['rnaseq_data'].apply(lambda x: len(x) > 0)]
    # print(mapping_df)
    # save the df for using later, or for KM plots
    # mapping_df.to_csv('mapping_df.csv', index=False)

    # combine 'days_to_death' and 'days_to_last_followup' into a single column
    # assuming that for rows where 'days_to_death' is NaN, 'days_to_last_followup' contains the censoring time
    mapping_df['time'] = mapping_df['days_to_death'].fillna(mapping_df['days_to_last_followup'])
    # NOTE: TCGA-49-6742 seems to have both 'days_to_death' as well as 'days_to_last_followup' as None. So ignoring this for survival analysis
    mapping_df = mapping_df.dropna(subset=['time', 'event_occurred'])
    # remove that from the wsi and rnaseq combined embeddings too
    # rnaseq_df = rnaseq_df.drop(columns=['TCGA-49-6742'])
    mapping_df.to_json('mapping_df.json', orient='index')
    rnaseq_df.to_json('rnaseq_df.json', orient='index')
else:
    mapping_df = pd.read_json(opt.input_mapping_data_path + "mapping_df.json", orient='index')

set_trace()
from train_test import train_nn
# train the model
if opt.profile:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True,
                 use_cuda=True) as prof:
        with record_function("model_train"):
            model, optimizer = train_nn(opt, data, device)
    # torch.autograd.profiler.profile().export_chrome_trace("./profiling_results.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    prof.export_chrome_trace("trace.json")

# with torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/multimodal'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
# ) as prof:

else:
    model, optimizer = train_nn(opt, mapping_df, device)
# break
# set_trace()
