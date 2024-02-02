import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from data_loaders import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader
from networks import define_net, define_reg, define_optimizer, define_scheduler
from utils import unfreeze_unimodal, CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, mixed_collate, count_parameters

#from GPUtil import showUtilization as gpu_usage
import pdb
import pickle
import os


def train(opt, data, device, k):

    model     = create_model(opt, k)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    scheduler = scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)

    print(model)

    use_patch, roi_dir = ('_patch_', 'all_st_patches_512')

    custom_data_loader = custom_dataloader(opt, data, split='train', mode=opt.mode)
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=True, collate_fn=mixed_collate)

    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):

        model.train()
        loss_epoch, grad_acc_epoch = 0, 0

        for batch_idx, (x_wsi, x_rnaseq, grade) in enumerate(train_loader):
            grade = grade.to(device)
            _, pred = model(x_wsi = x_wsi.to(device), x_rnaseq = x_rnaseq.to(device))
            loss = F.nll_loss(pred, grade)
            loss_epoch += loss.data.item()

            optimizer.zero_grade()
            loss.backward()
            optimizer.step()


            

    

