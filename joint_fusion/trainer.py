import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import datasets
import models
from train_test import train_nn, test
import argparse
from concurrent.futures import ThreadPoolExecutor
import pickle
from pdb import set_trace
import os
import ast
from collections import Counter
# from data_mapping import create_data_mapping

# for profiling
# torch.autograd.profiler.profile(enabled=True)
from torch.profiler import profile, record_function, ProfilerActivity

# on Dell laptop (activate conda env 'pytorch_py3p10' and use 'python trainer.py')
dataroot = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/data_from_pathomic_fusion/data/TCGA_GBMLGG/splits/'

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str,
                    default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/',
                    help='Path to input data files')
# parser.add_argument('--output_path', type=str, default='results/output.txt', help='Path to output results file')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--lr_decay_iters', type=int, default=100, help='Learning rate decay steps')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--input_size_wsi', type=int, default=1024, help="input_size for path images")
parser.add_argument('--embedding_dim_wsi', type=int, default=128, help="embedding dimension for WSI")
parser.add_argument('--embedding_dim_omic', type=int, default=128, help="embedding dimension for omic")
parser.add_argument('--input_modes', type=str, default="omic", help="wsi, omic, wsi_omic")
parser.add_argument('--fusion_type', type=str, default="early", help="early, late, joint")
parser.add_argument('--profile', type=str, default=False, help="whether to profile or not")
parser.add_argument('--use_mixed_precision', type=str, default=True, help="whether to use mixed precision calculations")
parser.add_argument('--use_gradient_accumulation', type=str, default=False, help="whether to use gradient accumulation")

opt = parser.parse_args()

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
torch.backends.cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

# create data mappings
# read the file containing gene expression and tile image locations for the TCGA-LUAD samples (mapped_data_16March)
mapping_df = pd.read_csv(opt.input_path + "mapped_data_21March.csv")

# The rnaseq data are saved as string representations of dictionary, not actual dictionary object. Need to convert
print("Converting the rnaseq data to proper dicts (may take a min)")
# mapping_df['rnaseq_data'] = mapping_df['rnaseq_data'].map(ast.literal_eval)
# mapping_df['rnaseq_data'] = ast.literal_eval(mapping_df['rnaseq_data'])
mapping_df['rnaseq_data'] = mapping_df['rnaseq_data'].apply(eval)
print("Conversion over")

create_data_mapping(opt)




# train the model
# set_trace()
# model, optimizer, metric_logger = train_nn(opt, data, device, cv_id)

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
    model, optimizer = train_nn(opt, data, device)
# break
# set_trace()
