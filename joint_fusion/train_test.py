import random
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler
# import torch.optim.lr_scheduler as lr_scheduler

from datasets import CustomDataset
from models import MultimodalNetwork
from utils import mixed_collate

import pdb
import pickle
import os
from pdb import set_trace

class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, log_risks, times, events):
        """

        :param log_risks: predictions from the NN
        :param times: observed survival time
        :param events: event (death) indicators (1/0)
        :return: Cox loss (scalar)
        """
        sorted_times, sorted_indices = torch.sort(times, descending=True)
        sorted_log_risks = log_risks[sorted_indices]
        sorted_events = events[sorted_indices]

        risk_set_sum = torch.cumsum(torch.exp(sorted_log_risks), dim=0)  # this is the term within summation for the second term on LHS

        event_mask = sorted_events.bool()
        cox_loss = -torch.sum(sorted_log_risks[event_mask] - torch.log(risk_set_sum))
        cox_loss /= torch.sum(events)  # should this be done?

        return cox_loss



def train(opt, data, device, cv_id):

    # first extract only the wsi data and the labeled grades
    # set_trace()
    model     = MultimodalNetwork(opt, cv_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)

    cox_loss = CoxLoss()

    print(model)

    use_patch, roi_dir = ('_patch_', 'all_st_patches_512')

    custom_dataset = CustomDataset(opt, data, split='train', mode=opt.input_modes)
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=mixed_collate)

    for epoch in tqdm(range(1, opt.num_epochs)):

        model.train()
        loss_epoch, grad_acc_epoch = 0, 0

        for batch_idx, (x_wsi, x_omic, censor, survtime, grade) in enumerate(train_loader):
            grade = grade.to(device)
            set_trace()
            _, pred = model(x_wsi = x_wsi.to(device), x_omic = x_omic.to(device))
            # loss = F.nll_loss(pred, grade)  # cross entropy for cancer grade classification
            loss = cox_loss(pred, t, e)                          # Cox partial likelihood loss for survival outcome prediction
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
