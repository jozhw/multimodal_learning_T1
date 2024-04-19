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
from train_test import train_nn
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

parser = argparse.ArgumentParser()
parser.add_argument('--create_new_data_mapping', type=str, default=True, help="whether to create new data mapping or use existing one")
parser.add_argument('--input_mapping_data_path', type=str,
                    default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/joint_fusion/', # on laptop
                    # default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/joint_fusion/', # on Polaris
                    help='Path to input mapping data file')
parser.add_argument('--input_path', type=str,
                    default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/', # on laptop
                    # default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/', # on Polaris
                    help='Path to input data files')
parser.add_argument('--input_wsi_path', type=str,
                    default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/TCGA_WSI/batch_corrected/processed_svs/tiles/256px_9.9x/combined_tiles/', # on laptop
                    # default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/batch_corrected/processed_svs/tiles/256px_9.9x/combined_tiles/', # on Polaris
                    help='Path to input WSI tiles')
# parser.add_argument('--output_path', type=str, default='results/output.txt', help='Path to output results file')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--lr_decay_iters', type=int, default=100, help='Learning rate decay steps')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--input_size_wsi', type=int, default=256, help="input_size for path images")
parser.add_argument('--embedding_dim_wsi', type=int, default=384, help="embedding dimension for WSI")
parser.add_argument('--embedding_dim_omic', type=int, default=256, help="embedding dimension for omic")
parser.add_argument('--input_mode', type=str, default="wsi", help="wsi, omic, wsi_omic")
parser.add_argument('--fusion_type', type=str, default="early", help="early, late, joint, unimodal")
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

    mapping_df = pd.read_json(opt.input_path + "mapped_data_9april.json", orient='index')


    # The rnaseq data are saved as string representations of dictionary, not actual dictionary object. Need to convert
    ### The below is not required if the input mapping file is json
    # print("Converting the rnaseq data to proper dicts (may take a min)")
    # mapping_df['rnaseq_data'] = mapping_df['rnaseq_data'].apply(eval)
    # # mapping_df['rnaseq_data'] = mapping_df['rnaseq_data'].apply(ast.literal_eval)
    # print("Conversion ongoing")
    # mapping_df['tiles'] = mapping_df['tiles'].apply(ast.literal_eval)  # convert strings to lists
    # print("Conversion over")
    # set_trace()
    # keep only rows that have both wsi and rnaseq data
    ids_with_wsi = mapping_df[mapping_df['tiles'].map(len) > 0].index.tolist()
    # extract df for training the rnaseq VAE: samples for which WSI are not available
    mask_training_vae = ~mapping_df.index.isin(ids_with_wsi)
    mapping_vae_training_df = mapping_df[mask_training_vae]
    rnaseq_df = pd.DataFrame(mapping_vae_training_df['rnaseq_data'].to_list(), index=mapping_vae_training_df.index).transpose()
    print("Are there nans in rnaseq_df: ", rnaseq_df.isna().any().any())
    # set_trace()
    # df containing entries where both WSI and rnaseq data are available
    mapping_df = mapping_df.loc[ids_with_wsi]

    # remove rows where the number of tiles is different from the standard (to avoid length mismatch issues during batching)
    mask = mapping_df['tiles'].apply(len) == 400
    mapping_df = mapping_df[mask]

    # # check if there are empty (or unexpected number of) rnaseq entries
    # # check why this is happening
    # rnaseq_dict_sizes = mapping_df['rnaseq_data'].apply(lambda x: len(x))
    # print("rnaseq_dict_sizes.value_counts(): ", rnaseq_dict_sizes.value_counts())
    # mapping_df = mapping_df[mapping_df['rnaseq_data'].apply(lambda x: len(x) > 0)]
    # print(mapping_df)

    # save the df for using later, or for KM plots
    # mapping_df.to_csv('mapping_df.csv', index=False)
    mapping_df.to_json('mapping_df.json', orient='index')
    rnaseq_df.to_json('rnaseq_df.json', orient='index')
else:
    mapping_df = pd.read_json(opt.input_mapping_data_path + "mapping_df.json", orient='index')


# set_trace()

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
