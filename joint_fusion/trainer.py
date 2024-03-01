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
import pickle
from pdb import set_trace
import os
import ast

# for profiling
# torch.autograd.profiler.profile(enabled=True)
from torch.profiler import profile, record_function, ProfilerActivity

# on Polaris
# dataroot = '/lus/grand/projects/GeomicVar/tarak/multimodal_lucid/data/TCGA_GBMLGG/splits'

# on Dell laptop (activate conda env 'pytorch' and use 'python3.10 trainer.py')
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
parser.add_argument('--input_modes', type=str, default="wsi_omic", help="wsi, omic, wsi_omic")
parser.add_argument('--fusion_type', type=str, default="joint", help="early, late, joint")
parser.add_argument('--profile', type=str, default=False, help="whether to profile or not")
parser.add_argument('--use_mixed_precision', type=str, default=True, help="whether to use mixed precision calculations")
parser.add_argument('--use_gradient_accumulation', type=str, default=False, help="whether to use gradient accumulation")

opt = parser.parse_args()

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
torch.backends.cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

# create the data (Note: the dataset and dataloader are created within train_test.py)

# get the rnaseq data
data_rnaseq_all_genes_df = pd.read_csv(opt.input_path + 'combined_rnaseq_TCGA-LUAD.tsv', delimiter='\t')
# keep only the protein coding genes
data_rnaseq_df = data_rnaseq_all_genes_df[data_rnaseq_all_genes_df['gene_type'] == 'protein_coding']
# for some of the samples, there may be more than one rnaseq dataset (those with -01A- are from the primary tumor while those with -11A- are from tissue adjacent to the tumor (so, normal?)
# in such cases, we will keep only the tumor sample


# get the corresponding clinical data
data_clinical_df = pd.read_csv(opt.input_path + 'combined_clinical_TCGA-LUAD.tsv', delimiter='\t')
# extract 'days_to_death' and 'vital_status' into lists for each column
days_to_death_list = []
vital_status_list = []
for col in data_clinical_df.columns:
    print(data_clinical_df[col])
    # print(col)
    val = data_clinical_df[col].values[0]
    val = val.replace('nan', 'None')
    try:
        val_list = ast.literal_eval(val)
    except ValueError as e:
        print(f"Error processing column {col}: {e}")
        continue
    print(val_list)
    # replace None ('nan') with 1e10
    # if val_list[0] is None:
    #     val_list[0] = 1e10

    print(val_list)
    # set_trace()
    days_to_death = val_list[0]
    vital_status = val_list[1]
    days_to_death_list.append(days_to_death)
    vital_status_list.append(vital_status)

set_trace()

# create dictionary with WSI (paths to the tiles), rnaseq and clinical data



# train the model
# set_trace()
# model, optimizer, metric_logger = train_nn(opt, data, device, cv_id)

if opt.profile:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True,
                 use_cuda=True) as prof:
        with record_function("model_train"):
            model, optimizer = train_nn(opt, data, device, cv_id)
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
