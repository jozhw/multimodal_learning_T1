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
from torch.cuda.amp import autocast, GradScaler
from datasets import CustomDataset
from models import MultimodalNetwork, print_model_summary
from utils import mixed_collate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pdb
import pickle
import os
from pdb import set_trace
torch.autograd.set_detect_anomaly(True)

class CoxLossOld(nn.Module):
    def __init__(self):
        super(CoxLossOld, self).__init__()

    def forward(self, log_risks, times, censor):
        """
        :param log_risks: predictions from the NN
        :param times: observed survival times (i.e. times to death) for the batch
        :param censor: censor data, event (death) indicators (1/0)
        :return: Cox loss (scalar)
        : NOTE: There's an issue with how the risk-set (inner sum in term 2) is calculated here

        """
        sorted_times, sorted_indices = torch.sort(times, descending=True)
        sorted_log_risks = log_risks[sorted_indices]
        sorted_censor = censor[sorted_indices]

        # sorted_log_risks = sorted_log_risks - torch.max(sorted_log_risks)  # to avoid overflow
        # Cox partial likelihood loss = log risk for each individual - cumulative risk (i.e., sum of risks of all at-risk individuals)
        # batching will prevent including all at-risk samples in the second term
        # if within a batch all samples are censored,  it will lead to a sum over an empty set for the second term
        # a small number is added to the term inside the log in the second term to handle such cases.

        cox_loss = sorted_log_risks - torch.log(torch.cumsum(torch.exp(sorted_log_risks), dim=0) + 1e-15)
        cox_loss = - cox_loss * sorted_censor
        # check the shape of sorted_log_risks

        # risk_set_sum = torch.cumsum(torch.exp(sorted_log_risks),
        #                             dim=0)  # this is the term within summation for the second term on LHS
        #
        # censor_mask = sorted_censor.bool()
        # cox_loss = -torch.sum(sorted_log_risks[censor_mask] - torch.log(risk_set_sum))
        # cox_loss /= torch.sum(events)  # should this be done?

        return cox_loss.mean()

class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, log_risks, times, censor):
        """
        :param log_risks: predictions from the NN
        :param times: observed survival times (i.e. times to death) for the batch
        :param censor: censor data, event (death) indicators (1/0)
        :return: Cox loss (scalar)
        """
        sorted_times, sorted_indices = torch.sort(times, descending=True)
        sorted_log_risks = log_risks[sorted_indices]
        sorted_censor = censor[sorted_indices]

        # precompute for using within the inner sum of term 2 in Cox loss
        exp_sorted_log_risks  = torch.exp(sorted_log_risks)

        # initialize all samples to be at-risk (will update it below)
        at_risk_mask = torch.ones_like(sorted_times, dtype=torch.bool)

        losses = []
        for time_index in range(len(sorted_times)):
            # include only the uncensored samples
            if sorted_censor[time_index] == 1:
                at_risk_mask = torch.arange(len(sorted_times)) <= time_index
                at_risk_mask = at_risk_mask.to(device)
                at_risk_sum = torch.sum(exp_sorted_log_risks[at_risk_mask]) # all are at-risk for the first sample (after arranged in descending order)
                loss = sorted_log_risks[time_index] - torch.log(at_risk_sum + 1e-15)
                losses.append(loss)

            # at_risk_mask[time_index] = False # the i'th sample is no more in the risk-set as the event has already occurred for it

        # if no uncensored samples are in the mini-batch return 0
        if not losses:
            return torch.tensor(0.0, requires_grad=True)

        cox_loss = -torch.mean(torch.stack(losses))
        return cox_loss


def train_nn(opt, mapping_df, device):
    model = MultimodalNetwork(embedding_dim_wsi=opt.embedding_dim_wsi,
                              embedding_dim_omic=opt.embedding_dim_omic,
                              mode=opt.input_mode,
                              fusion_type=opt.fusion_type)
    # model should return None for the absent modality in the unimodal case
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    cox_loss = CoxLoss()
    print(f"Model mode: {model.mode}, fusion_type: {model.fusion_type}")
    # set_trace()
    # if model.wsi_net is not None:
    print("##############  WSINetwork Summary  ##################")
    print_model_summary(model.wsi_net)

    # if model.omic_net is not None:
    print("\n ##############  OmicNetwork Summary  ##############")
    print_model_summary(model.omic_net)

    print("\n ##############  MultimodalNetwork Summary ##############")
    print_model_summary(model)

    # print("--------Model arch------------")
    # print(model)
    # print("--------WSI Model summary -----------")
    # if model.wsi_net is not None:
    #     summary(model.wsi_net, input_size=(3, 1024, 1024))

    if opt.use_gradient_accumulation:
        accumulation_steps = 10
    if opt.use_mixed_precision:
        scaler = GradScaler()

    # the mapping_df below should be split into 'train', 'validation', and 'test', with only the former 2 used for training
    # custom_dataset = CustomDataset(opt, mapping_df, split='train', mode=opt.input_mode)
    custom_dataset = CustomDataset(opt, mapping_df, mode=opt.input_mode)
    # set_trace()
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,)
                                               # collate_fn=mixed_collate)

    # # use a separate train_loader for early fusion that should only handle the embeddings read from the files
    # if opt.fusion_type == 'early':

    for epoch in tqdm(range(1, opt.num_epochs+1)):
        print("epoch: ", epoch, " out of ", opt.num_epochs)
        model.train()
        loss_epoch = 0

        for batch_idx, (tcga_id, days_to_death, days_to_last_followup, event_occurred, x_wsi, x_omic) in enumerate(train_loader):
            # x_wsi is a list of tensors (one tensor for each tile)
            print("batch_index: ", batch_idx, " out of ", np.ceil(len(custom_dataset) / opt.batch_size))
            x_wsi = [x.to(device) for x in x_wsi]
            x_omic = x_omic.to(device)

            days_to_death = days_to_death.to(device)
            days_to_last_followup = days_to_last_followup.to(device)
            event_occurred = event_occurred.to(device)
            print("Days to death: ", days_to_death)
            print("event occurred: ", event_occurred)

            optimizer.zero_grad()
            if opt.use_mixed_precision:
                with autocast():  # should wrap only the forward pass including the loss calculation
                    predictions = model(x_wsi=x_wsi,  # list of tensors (one for each tile)
                                        x_omic=x_omic)
                    loss = cox_loss(predictions.squeeze(),
                                    days_to_death,
                                    event_occurred)
                    print("\n loss: ", loss.data.item())
                    loss_epoch += loss.data.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                print(" Not using mixed precision")
                # model for survival outcome (uses Cox PH partial log likelihood as the loss function)
                # the model output should be considered as beta*X to be used in the Cox loss function

                # for early fusion, the model class should use the inputs from here by the generated embeddings
                predictions = model(opt,
                                    tcga_id,
                                    x_wsi=x_wsi,  # list of tensors (one for each tile)
                                    x_omic=x_omic,
                                    )
                print("Predictions: ", predictions.squeeze())
                print("\n True days to event: ", days_to_death)
                # set_trace()
                # loss = F.nll_loss(predictions, grade)  # cross entropy for cancer grade classification
                loss = cox_loss(predictions.squeeze(),   # predictions are not survival outcomes, rather log-risk scores beta*X
                                days_to_death,
                                event_occurred)  # Cox partial likelihood loss for survival outcome prediction
                # set_trace()
                print("\n loss: ", loss.data.item())
                loss_epoch += loss.data.item()
                loss.backward()
                optimizer.step()

                # Get the primary grade class
                # predictions = predictions.argmax(dim=1, keepdim=True)
                # grade_acc_epoch += predictions.eq(grade.view_as(predictions)).sum().item()

            scheduler.step()
            # break
        print("epoch loss: ", loss_epoch)
    return model, optimizer


def test():
    pass


def validation():
    pass
