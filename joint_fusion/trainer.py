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

# for profiling
# torch.autograd.profiler.profile(enabled=True)
from torch.profiler import profile, record_function, ProfilerActivity

# on Polaris
# dataroot = '/lus/grand/projects/GeomicVar/tarak/multimodal_lucid/data/TCGA_GBMLGG/splits'

# on Dell laptop (activate conda env 'pytorch' and use 'python3.10 trainer.py')
dataroot = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/data_from_pathomic_fusion/data/TCGA_GBMLGG/splits/'

parser = argparse.ArgumentParser()
# parser.add_argument('--input_path', type=str, default='data/input.txt', help='Path to input data file')
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
parser.add_argument('--profile', type=str, default=False, help="whether to profile or not")
parser.add_argument('--use_mixed_precision', type=str, default=True, help="whether to use mixed precision calculations")
parser.add_argument('--use_gradient_accumulation', type=str, default=False, help="whether to use gradient accumulation")

opt = parser.parse_args()

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
torch.backends.cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

# keep only the samples that have both WSI data and omic data, and use VGG for the image feature creation
ignore_missing_hist_subtype = 0  # grades (samples with missing grades are fine as we are only doing survival prediction; grades are neither input features nor target)
ignore_missing_mol_subtype = 1
use_vgg_features = 0
use_omic = '_rnaseq'  # pickle files with the rnaseq data included end with _rnaseq
# use_omic = ''

# use_patch, roi_dir = ('_patch_', 'all_st_patches_512')
use_patch, roi_dir = ('_patch_', 'all_st')

# set the path to the approriate pickle files
data_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (
dataroot, roi_dir, ignore_missing_mol_subtype, ignore_missing_hist_subtype, use_vgg_features, use_omic)
print("Loading %s" % data_path)

# load the pickle file (contains data splits corresponding to 15 fold CV)
data_cv = pickle.load(open(data_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
results = []

# Get predictions for each split (we will keep the best performing model)
for cv_id, data in data_cv_splits.items():
    print("************** SPLIT (%d/%d) **************" % (cv_id, len(data_cv_splits.items())))

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
        model, optimizer = train_nn(opt, data, device, cv_id)
    # break
    # set_trace()

#     # get the training and test losses
#     loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train = test(opt, model, data, 'train', device)
#     # loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data, 'test', device)
#
#     print("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
#     logging.info("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
#     print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
#     logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
#     results.append(grad_acc_test)
#
#     if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
# 	    model_state_dict = model.module.cpu().state_dict()
#     else:
# 	    model_state_dict = model.cpu().state_dict()
#
#     torch.save({
# 	    'split':k,
# 	    'opt': opt,
# 	    'epoch': opt.niter+opt.niter_decay,
# 	    'data': data,
# 	    'model_state_dict': model_state_dict,
# 	    'optimizer_state_dict': optimizer.state_dict(),
# 	    'metrics': metric_logger},
# 	    os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k)))
#
# 	# print()
#
#     pickle.dump(pred_train, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
#     pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))
#
#
# print('Split Results:', results)
# print("Average:", np.array(results).mean())
# pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))
#
