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
from models import MultimodalNetwork, OmicNetwork, print_model_summary
from sklearn.model_selection import train_test_split
from utils import mixed_collate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pdb
import pickle
import os
from pdb import set_trace
from captum.attr import IntegratedGradients

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
        exp_sorted_log_risks = torch.exp(sorted_log_risks)

        # initialize all samples to be at-risk (will update it below)
        at_risk_mask = torch.ones_like(sorted_times, dtype=torch.bool)

        losses = []
        for time_index in range(len(sorted_times)):
            # include only the uncensored samples
            if sorted_censor[time_index] == 1:
                at_risk_mask = torch.arange(len(sorted_times)) <= time_index
                at_risk_mask = at_risk_mask.to(device)
                at_risk_sum = torch.sum(exp_sorted_log_risks[
                                            at_risk_mask])  # all are at-risk for the first sample (after arranged in descending order)
                loss = sorted_log_risks[time_index] - torch.log(at_risk_sum + 1e-15)
                losses.append(loss)

            # at_risk_mask[time_index] = False # the i'th sample is no more in the risk-set as the event has already occurred for it

        # if no uncensored samples are in the mini-batch return 0
        if not losses:
            return torch.tensor(0.0, requires_grad=True)

        cox_loss = -torch.mean(torch.stack(losses))
        return cox_loss


def create_data_loaders(opt, mapping_df):
    mapping_df_train, temp_df = train_test_split(mapping_df, test_size=0.3, random_state=40)
    mapping_df_val, mapping_df_test = train_test_split(temp_df, test_size=0.5, random_state=40)

    train_loader = torch.utils.data.DataLoader(
        dataset=CustomDataset(opt, mapping_df_train, mode=opt.input_mode),
        batch_size=opt.batch_size,
        shuffle=True,
        # collate_fn=mixed_collate  # Uncomment if needed
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset=CustomDataset(opt, mapping_df_val, mode=opt.input_mode),
        batch_size=opt.batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=CustomDataset(opt, mapping_df_test, mode=opt.input_mode),
        batch_size=opt.batch_size,
        shuffle=True,
    )

    return train_loader, validation_loader, test_loader


def train_nn(opt, mapping_df, device):
    model = MultimodalNetwork(embedding_dim_wsi=opt.embedding_dim_wsi,
                              embedding_dim_omic=opt.embedding_dim_omic,
                              mode=opt.input_mode,
                              fusion_type=opt.fusion_type)
    # model should return None for the absent modality in the unimodal case

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    cox_loss = CoxLoss()
    # print(f"Model mode: {model.mode}, fusion_type: {model.fusion_type}")
    # the original model is now accessible through the .module attribute of the DataParallel wrapper
    print(f"Model mode: {model.module.mode}, fusion_type: {model.module.fusion_type}")
    # set_trace()
    # if model.wsi_net is not None:
    print("##############  WSINetwork Summary  ##################")
    # print_model_summary(model.wsi_net)
    if model.module.wsi_net is not None:
        print_model_summary(model.module.wsi_net)

    # if model.omic_net is not None:
    print("\n ##############  OmicNetwork Summary  ##############")
    # print_model_summary(model.omic_net)
    if model.module.omic_net is not None:
        print_model_summary(model.module.omic_net)

    print("\n ##############  MultimodalNetwork Summary ##############")
    # print_model_summary(model)
    print_model_summary(model.module)

    # print("--------Model arch------------")
    # print(model)
    # print("--------WSI Model summary -----------")
    # if model.wsi_net is not None:
    #     summary(model.wsi_net, input_size=(3, 1024, 1024))

    if opt.use_gradient_accumulation:
        accumulation_steps = 10
    if opt.use_mixed_precision:
        scaler = GradScaler()

    # # the mapping_df below should be split into 'train', 'validation', and 'test', with only the former 2 used for training
    # # custom_dataset = CustomDataset(opt, mapping_df, split='train', mode=opt.input_mode)
    # # split mapping_df into train/val/test sets
    # mapping_df_train, temp_df = train_test_split(mapping_df, test_size=0.3, random_state=42)
    # mapping_df_val, mapping_df_test = train_test_split(temp_df, test_size=0.5, random_state=42)
    #
    # print(f"Training set size: {mapping_df_train.shape[0]}")
    # print(f"Validation set size: {mapping_df_val.shape[0]}")
    # print(f"Test set size: {mapping_df_test.shape[0]}")
    #
    # custom_dataset_train = CustomDataset(opt, mapping_df_train, mode=opt.input_mode)
    # custom_dataset_validation = CustomDataset(opt, mapping_df_validation, mode=opt.input_mode)
    # custom_dataset_test = CustomDataset(opt, mapping_df_test, mode=opt.input_mode)
    # # set_trace()
    # train_loader = torch.utils.data.DataLoader(dataset=custom_dataset_train,
    #                                            batch_size=opt.batch_size,
    #                                            shuffle=True, )
    # # collate_fn=mixed_collate)
    #
    # validation_loader = torch.utils.data.DataLoader(dataset=custom_dataset_train,
    #                                                 batch_size=opt.batch_size,
    #                                                 shuffle=True, )
    # # collate_fn=mixed_collate)
    #
    # test_loader = torch.utils.data.DataLoader(dataset=custom_dataset_test,
    #                                           batch_size=opt.batch_size,
    #                                           shuffle=True, )
    # # collate_fn=mixed_collate)

    # # use a separate train_loader for early fusion that should only handle the embeddings read from the files
    # if opt.fusion_type == 'early':

    train_loader, validation_loader, test_loader = create_data_loaders(opt, mapping_df)

    for epoch in tqdm(range(1, opt.num_epochs + 1)):
        print(f"Epoch: {epoch} out of {opt.num_epochs}")
        model.train()
        loss_epoch = 0

        # model training in batches using the train dataloader
        for batch_idx, (tcga_id, days_to_event, event_occurred, x_wsi, x_omic) in enumerate(train_loader):
            # x_wsi is a list of tensors (one tensor for each tile)
            print(f"Batch size: {opt.batch_size}")
            print(f"Batch index: {batch_idx + 1} out of {np.ceil(len(train_loader.dataset) / opt.batch_size)}")
            x_wsi = [x.to(device) for x in x_wsi]
            x_omic = x_omic.to(device)
            days_to_event = days_to_event.to(device)
            # days_to_last_followup = days_to_last_followup.to(device)
            event_occurred = event_occurred.to(device)
            print("Days to event: ", days_to_event)
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
                # need to correct the loss calculation to include the time to last followup
                loss = cox_loss(predictions.squeeze(),
                                # predictions are not survival outcomes, rather log-risk scores beta*X
                                days_to_event,
                                event_occurred)  # Cox partial likelihood loss for survival outcome prediction
                # set_trace()
                print("\n loss (train): ", loss.data.item())
                loss_epoch += loss.data.item() * len(
                    x_omic)  # multiplying loss by batch size for accurate epoch averaging
                loss.backward()
                optimizer.step()

        train_loss = loss_epoch / len(train_loader.dataset)  # average training loss per sample for the epoch
        scheduler.step()  # step scheduler after each epoch
        print("train loss: ", train_loss)

        # get predictions on the validation dataset (to keep track of validation loss during training)
        model.eval()
        val_loss_epoch = 0.0
        if epoch % 100:
            with torch.no_grad():
                for batch_idx, (tcga_id, days_to_event, event_occurred, x_wsi, x_omic) in enumerate(validation_loader):
                    # x_wsi is a list of tensors (one tensor for each tile)
                    print(
                        f"Validation Batch index: {batch_idx} out of {np.ceil(len(validation_loader.dataset) / opt.batch_size)}")
                    x_wsi = [x.to(device) for x in x_wsi]
                    x_omic = x_omic.to(device)
                    days_to_event = days_to_event.to(device)
                    event_occurred = event_occurred.to(device)
                    print("Days to event: ", days_to_event)
                    print("event occurred: ", event_occurred)
                    outputs = model(opt,
                                    tcga_id,
                                    x_wsi=x_wsi,  # list of tensors (one for each tile)
                                    x_omic=x_omic,
                                    )
                    loss = cox_loss(outputs.squeeze(),
                                    # predictions are not survival outcomes, rather log-risk scores beta*X
                                    days_to_event,
                                    event_occurred)  # Cox partial likelihood loss for survival outcome prediction
                    # set_trace()
                    print("\n loss (validation): ", loss.data.item())
                    val_loss_epoch += loss.data.item() * len(x_omic)

                    model_path = os.path.join("./saved_models", f"model_epoch_{epoch}.pt")
                    torch.save(model.state_dict(), model_path)
                    print(f"saved model checkpoint at epoch {epoch} to {model_path}")

                val_loss = val_loss_epoch / len(validation_loader.dataset)
                print("Validation loss: ", val_loss)

    return model, optimizer


def test_and_interpret(model, test_loader, cox_loss, device):
    model.eval()
    test_loss = 0.0
    # integrated_gradients = IntegratedGradients(model.omic_net)  # need to check this as the flow of the gradients in backprop should be through the downstream MLP and the omic MLP
    # integrated_gradients = IntegratedGradients(model.fused_mlp)
    integrated_gradients = IntegratedGradients(model.forward_omic_only)
    all_attributions = []

    with torch.no_grad():
        for tcga_id, days_to_event, event_occurred, x_wsi, x_omic in test_loader:
            x_wsi = [x.to(device) for x in x_wsi]
            x_omic = x_omic.to(device)
            days_to_event = days_to_event.to(device)
            event_occurred = event_occurred.to(device)

            predictions = model(opt,
                                tcga_id,
                                x_wsi=x_wsi,
                                x_omic=x_omic)
            loss = cox_loss(predictions.squeeze(),
                            days_to_event,
                            event_occurred)
            test_loss += loss.item() * len(x_omic)

            x_omic.requires_grad = True
            baseline = torch.zeros_like(x_omic)  # is zeros the appropriate baseline?
            # set_trace()
            attrs, delta = integrated_gradients.attribute(inputs=x_omic,
                                                          baselines=baseline,
                                                          return_convergence_delta=True)
            all_attributions.append(attrs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    print("test loss: ", test_loss)

    all_attributions = np.concatenate(all_attributions, axis=0)
    np.save("omic_attributions.npy", all_attributions)

    return test_loss, all_attributions


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mapping_data_path', type=str,
                        default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/joint_fusion/',
                        # on laptop
                        # default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/joint_fusion/', # on Polaris
                        help='Path to input mapping data file')
    parser.add_argument('--input_wsi_path', type=str,
                        default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles/tiles/256px_9.9x/combined_tiles/',
                        # on laptop
                        # default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles/tiles/256px_9.9x/combined_tiles/', # on Polaris
                        help='Path to input WSI tiles')
    parser.add_argument('--input_wsi_embeddings_path', type=str,
                        default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/early_fusion_inputs/',
                        # on laptop
                        # default='/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles/tiles/256px_9.9x/combined_tiles/', # on Polaris
                        help='Path to WSI embeddings generated from pretrained pathology foundation model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--input_size_wsi', type=int, default=256, help="input_size for path images")
    parser.add_argument('--embedding_dim_wsi', type=int, default=384, help="embedding dimension for WSI")
    parser.add_argument('--embedding_dim_omic', type=int, default=256, help="embedding dimension for omic")
    parser.add_argument('--input_mode', type=str, default="wsi_omic", help="wsi, omic, wsi_omic")
    parser.add_argument('--fusion_type', type=str, default="joint_omic",
                        help="early, late, joint, joint_omic, unimodal")
    opt = parser.parse_args()

    mapping_df = pd.read_json(opt.input_mapping_data_path + "mapping_df.json", orient='index')

    # get predictions on test data, and calculate interpretability metrics
    model = MultimodalNetwork(embedding_dim_wsi=opt.embedding_dim_wsi,
                              embedding_dim_omic=opt.embedding_dim_omic,
                              mode=opt.input_mode,
                              fusion_type=opt.fusion_type)
    # model should return None for the absent modality in the unimodal case
    model.to(device)
    cox_loss = CoxLoss()

    train_loader, validation_loader, test_loader = create_data_loaders(opt, mapping_df)

    test_loss, attributions = test_and_interpret(model, test_loader, cox_loss, device)
