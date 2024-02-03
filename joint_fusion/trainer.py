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
from train_test import train, test
from pdb import set_trace

dataroot = '/lus/grand/projects/GeomicVar/tarak/multimodal_lucid/data/TCGA_GBMLGG/splits'

opt = parse_args()

ignore_missing_wsi = 1
ignore_missing_rnaseq = 1
use_vgg_features = 0
use_rnaseq = '_rnaseq'  # pickle files with rnaseq data end with _rnaseq

use_patch, roi_dir = ('_patch_', 'all_st_patches_512')

# set the path to the approriate pickle files
data_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (dataroot, roi_dir, ignore_missing_rnaseq, ignore_missing_wsi, use_vgg_features, use_rnaseq)
print("Loading %s" % data_cv_path)

# load the pickle file (contains data splits corresponding to 15 fold CV)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
results = []

for k, data in data_cv_splits.items():
    print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
    if os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d_patch_pred_train.pkl' % (opt.model_name, k))):
	    print("Train-Test Split already made.")
	    continue

    set_trace()
    # train the model
    model, optimizer, metric_logger = train(opt, data, device, k)

    # get the training and test losses
    loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train = test(opt, model, data, 'train', device)
    # loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data, 'test', device)

    print("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
    logging.info("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
    print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
    logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
    results.append(grad_acc_test)

    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
	    model_state_dict = model.module.cpu().state_dict()
    else:
	    model_state_dict = model.cpu().state_dict()
	
    torch.save({
	    'split':k,
	    'opt': opt,
	    'epoch': opt.niter+opt.niter_decay,
	    'data': data,
	    'model_state_dict': model_state_dict,
	    'optimizer_state_dict': optimizer.state_dict(),
	    'metrics': metric_logger}, 
	    os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k)))

	# print()

    pickle.dump(pred_train, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
    pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))


print('Split Results:', results)
print("Average:", np.array(results).mean())
pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))

