import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler
# import torch.optim.lr_scheduler as lr_scheduler

from data_loader import custom_dataloader
from models import MultimodalNetwork
from utils import mixed_collate
# from utils import unfreeze_unimodal, CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, mixed_collate, count_parameters

#from GPUtil import showUtilization as gpu_usage
import pdb
import pickle
import os
from pdb import set_trace


def train(opt, data, device, cv_id):

    # first extract only the wsi data and the labeled grades
    # set_trace()
    model     = MultimodalNetwork(opt, cv_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)

    print(model)

    use_patch, roi_dir = ('_patch_', 'all_st_patches_512')

    custom_data_loader = custom_dataloader(opt, data, split='train', mode=opt.input_modes)
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=True, collate_fn=mixed_collate)

    for epoch in tqdm(range(1, opt.num_epochs)):

        model.train()
        loss_epoch, grad_acc_epoch = 0, 0

        for batch_idx, (x_wsi, x_omic, grade) in enumerate(train_loader):
            grade = grade.to(device)
            set_trace()
            _, pred = model(x_wsi = x_wsi.to(device), x_omic = x_omic.to(device))
            loss = F.nll_loss(pred, grade)
            loss_epoch += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = pred.argmax(dim=1, keepdim=True)
            grad_acc_epoch += pred.eq(grade.view_as(pred)).sum().item()

            scheduler.step()

    return model, optimizer
            

def test():
    pass
