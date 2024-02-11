import random
from tqdm import tqdm
import numpy as np
import torch

from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler
# import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary

from datasets import CustomDataset
from models import MultimodalNetwork, print_model_summary
from utils import mixed_collate

import pdb
import pickle
import os
from pdb import set_trace


class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, log_risks, times, censor):
        """

        :param log_risks: predictions from the NN
        :param times: observed survival time
        :param censor: censor data, event (death) indicators (1/0)
        :return: Cox loss (scalar)
        """
        sorted_times, sorted_indices = torch.sort(times, descending=True)
        sorted_log_risks = log_risks[sorted_indices]
        sorted_censor = censor[sorted_indices]

        sorted_log_risks = sorted_log_risks - torch.max(sorted_log_risks)  # to avoid overflow
        cox_loss = sorted_log_risks - torch.log(torch.cumsum(torch.exp(sorted_log_risks), dim=0) + 1e-15)
        cox_loss = - cox_loss * sorted_censor

        # risk_set_sum = torch.cumsum(torch.exp(sorted_log_risks),
        #                             dim=0)  # this is the term within summation for the second term on LHS
        #
        # censor_mask = sorted_censor.bool()
        # cox_loss = -torch.sum(sorted_log_risks[censor_mask] - torch.log(risk_set_sum))
        # cox_loss /= torch.sum(events)  # should this be done?

        return cox_loss.mean()


def train_nn(opt, data, device, cv_id):
    model = MultimodalNetwork(embedding_dim_wsi=opt.embedding_dim_wsi,
                              embedding_dim_omic=opt.embedding_dim_omic,
                              mode=opt.input_modes)  # opt, cv_id)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    cox_loss = CoxLoss()

    print("WSINetwork Summary:")
    print_model_summary(model.wsi_net)
    print("\nOmicNetwork Summary:")
    print_model_summary(model.omic_net)
    print("\nMultimodalNetwork Summary:")
    print_model_summary(model)

    print("--------Model arch------------")
    print(model)
    print("--------WSI Model summary -----------")
    summary(model.wsi_net, input_size=(3, 1024, 1024))

    custom_dataset = CustomDataset(opt, data, split='train', mode=opt.input_modes)
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=opt.batch_size, shuffle=True,
                                               collate_fn=mixed_collate)

    for epoch in tqdm(range(1, opt.num_epochs)):
        print("epoch: ", epoch, " out of ", opt.num_epochs)

        model.train()  # set the model to train mode
        loss_epoch = 0

        for batch_idx, (x_wsi, x_omic, censor, survival_time, grade) in enumerate(train_loader):
            print("batch_index: ", batch_idx, " out of ", np.ceil(len(custom_dataset) / opt.batch_size))
            x_wsi = x_wsi.to(device)
            x_omic = x_omic.to(device)

            censor = censor.to(device)
            survival_time = survival_time.to(device)
            # grade = grade.to(device)
            optimizer.zero_grad()

            # model for survival outcome (uses Cox PH partial log likelihood as the loss function)
            # the model output should be considered as beta*X to be used in the Cox loss function
            predictions = model(x_wsi=x_wsi, x_omic=x_omic)

            # loss = F.nll_loss(predictions, grade)  # cross entropy for cancer grade classification
            loss = cox_loss(predictions.squeeze(), survival_time,
                            censor)  # Cox partial likelihood loss for survival outcome prediction
            # set_trace()
            print("loss: ", loss.data.item())
            loss_epoch += loss.data.item()
            loss.backward()
            optimizer.step()

            # Get the primary grade class
            # predictions = predictions.argmax(dim=1, keepdim=True)
            # grade_acc_epoch += predictions.eq(grade.view_as(predictions)).sum().item()

            scheduler.step()
            break
        print("epoch loss: ", loss_epoch)
    return model, optimizer


def test():
    pass


def validation():
    pass
