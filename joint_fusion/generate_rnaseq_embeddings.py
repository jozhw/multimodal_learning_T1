# Implement a MLP based VAE to create lower dimensional embeddings of the RNASeq data
# conda activate pytorch_py3p10 on laptop
# use 'wandb sweep sweep_config.yaml' followed by 'wandb agent <username>/<project_name>/<sweep_id>' [just follow the instructions after 'wandb sweep'] for HPO runs with wandb
# for running without HPO, simply use "python generate_rnaseq_embeddings.py"

import torch
from torch import nn, optim
import copy
import os
import joblib
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from torch.nn import functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
from pdb import set_trace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print("Running on GPU")
else:
    print("Running on CPU")

# rnaseq_df = pd.read_csv(
#     "/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/batchcorrected_combined_rnaseq_TCGA-LUAD.tsv",
#     delimiter='\t'
# )
#
# checkpoint_dir = "./checkpoints"
# os.makedirs(checkpoint_dir, exist_ok=True)
# # Remove the code below when using shortened TCGA IDs when creating the combined RNASeq file
# # rename the columns by keeping only the minimal part of the TCGA ID, e.g.,  TCGA-44-2655
# column_mapping = {
#     col: '-'.join(col.split('.')[:3]) if 'TCGA' in col else col
#     for col in rnaseq_df.columns
# }
# rnaseq_df = rnaseq_df.rename(columns=column_mapping)
# duplicated_columns = rnaseq_df.columns[rnaseq_df.columns.duplicated()]
# print("duplicated columns: ", duplicated_columns)
#
# # Remove duplicated columns
# rnaseq_df = rnaseq_df.loc[:, ~rnaseq_df.columns.duplicated()]


# set_trace()

class BetaVAE(nn.Module):
    def __init__(self, input_dim=None, latent_dim=None, intermediate_dim=None, beta=None, config=None):
        super(BetaVAE, self).__init__()
        if config: #during training with HPO using wandb (read parameters from the sweep_config.yaml file)
            self.input_dim = config.input_dim
            self.latent_dim = config.latent_dim
            self.intermediate_dim = config.intermediate_dim
            self.lr = config.learning_rate
        else:
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.intermediate_dim = intermediate_dim
            self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(),
            nn.Linear(intermediate_dim, latent_dim * 2)  # *2 for mean and log_var
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(),
            nn.Linear(intermediate_dim, input_dim)
            # nn.Sigmoid()  # assuming data is normalized between 0 and 1
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, log_var = torch.chunk(x, 2, dim=-1)  # log_var = log(std**2)
        return mean, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)  # to ensure the variance is > 0
        eps = torch.randn_like(std).to(device)
        z = mean + eps * std
        # set_trace()
        return self.decode(z), mean, log_var


def loss_function(recon_x, x, mean, log_var):
    reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  # See Geron Eq 17-4
    return reconstruction_loss, kl_div_loss


# function to take a trained VAE and generate embeddings for a new rnaseq dataset
# essentially, just pass the input through the encoder network
# for early fusion, these embeddings should be loaded only once (before training of the downstream MLP starts)
def get_omic_embeddings(x_omic):
    print("In get_omic_embeddings()")
    # work directly on the tensor on the device
    # (CHANGE) actually, need to use the mean and std from the training data
    x_omic_log = torch.log1p(x_omic)
    # mean = x_omic_log.mean(dim=1, keepdim=True)
    # std = x_omic_log.std(dim=1, keepdim=True)
    scaler = joblib.load('scaler.save')
    mean = scaler.mean_
    std = scaler.scale_
    x_omic_scaled = (x_omic_log - mean) / (std + 1e-6)

    # model = BetaVAE(input_dim=input_dim, latent_dim=latent_dim, intermediate_dim=intermediate_dim, beta=beta)
    model = BetaVAE()
    checkpoints = [file for file in os.listdir(checkpoint_dir) if file.endswith('.pth')]
    latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"loaded checkpoint from {checkpoint_path}")
    model.to(device)
    model.eval()
    embeddings = []
    # forward pass through the encoder to obtain the embeddings
    with torch.no_grad():
        for data in DataLoader(TensorDataset(torch.tensor(x_omic_scaled, dtype=torch.float)), batch_size=1):
            input = data[0].to(device)
            mean, _ = model.encode(input)
            embeddings.append(mean.cpu().numpy())
    # Concatenate all batch embeddings into a single numpy array
    embeddings = np.concatenate(embeddings, axis=0)
    # set_trace()
    return embeddings


def main():
    restart_training = False
    wandb.init(project="rnaseq_vae", entity="tnnandi")
    config = wandb.config
    config = None # remove this when using the sweep_config.yaml file

    # For obtaining the trained VAE, use the samples in the TCGA-LUAD database except those for which we are using the WSI samples now (these will be used as the test set)
    X = rnaseq_df.values
    # omit gene_id, gene_name, gene_type
    # X = rnaseq_df.iloc[:, 3:].values

    # log transform to handle zero values
    X_log = np.log1p(X)
    scaler = StandardScaler() # save the scaling parameters to be used during inference
    # transpose so samples are rows
    X_scaled = scaler.fit_transform(X_log.T)

    # save scaler to file
    joblib.dump(scaler, 'scaler.save')

    X_train, X_remaining = train_test_split(X_scaled, test_size=0.2, random_state=42)
    # Split the remaining data into validation (10%) and test (10%)
    X_val, X_test = train_test_split(X_remaining, test_size=0.5, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float).to(device))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float).to(device))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float).to(device))

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)
    # to view in pdb: batch = next(iter(train_loader))

    print("Does X_train have any nan: ", np.isnan(X_train).any())
    print("Does X_val have any nan: ", np.isnan(X_val).any())
    print("Does X_test have any nan: ", np.isnan(X_test).any())
    # set_trace()
    model = BetaVAE(input_dim=input_dim,
                    latent_dim=latent_dim,
                    intermediate_dim=intermediate_dim,
                    beta=beta)
    # model = BetaVAE(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    if restart_training:
        checkpoints = [file for file in os.listdir(checkpoint_dir) if file.endswith('.pth')]
        latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"loaded checkpoint from {checkpoint_path}")

    else:
        if torch.cuda.device_count() > 1:
            print(f"using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for data in train_loader:
                optimizer.zero_grad()
                input = data[0].to(device)
                current_lr = optimizer.param_groups[0]['lr']
                # if epoch == 0:
                #     print("Model summary: ", summary(model, input.shape))
                recon_batch, mean, log_var = model(input)
                reconstruction_loss, kl_divergence_loss = loss_function(recon_batch, input, mean, log_var)
                loss = reconstruction_loss + beta * kl_divergence_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if epoch % 1000 == 0:
                # validation and print losses

                print("epoch: ", epoch, " LR: ", current_lr)
                print("training loss (reconstruction_loss, kl_divergence_loss): ", train_loss, "(", reconstruction_loss,
                      ",",
                      kl_divergence_loss, ")")
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        input = batch[0].to(device)
                        recon_batch, mean, log_var = model(input)
                        reconstruction_loss, kl_divergence_loss = loss_function(recon_batch, input, mean, log_var)
                        loss = reconstruction_loss + beta * kl_divergence_loss
                        val_loss += loss.item()
                # set_trace()
                val_loss /= len(val_loader.dataset)
                train_loss /= len(train_loader.dataset)
                print("validation loss: ", val_loss)

            # save model and other states every 10K epochs
            if epoch % 10000 == 0 and epoch > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
                print(f"checkpoint saved at epoch {epoch} to {checkpoint_path}")

            wandb.log({"epoch": epoch, "LR": current_lr, "train_loss": train_loss,
                       "reconstruction_loss": reconstruction_loss.item(),
                       "kl_div_loss": kl_divergence_loss.item(), "val_loss": val_loss})

            scheduler.step()

            # print(f'epoch {epoch}, train loss: {train_loss}, val loss: {val_loss}')

    wandb.finish()


if __name__ == "__main__":
    mapping_df = pd.read_json(
        '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/joint_fusion/' + "mapping_df.json",
        orient='index')
    # mapping_vae_training_df = pd.read_json(
    #     '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/joint_fusion/' + 'mapping_vae_training_df.json',
    #     orient='index')
    rnaseq_df = pd.read_json(
        '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/joint_fusion/' + 'rnaseq_df.json',
        orient='index')

    # common_indices = mapping_vae_training_df.index.intersection(mapping_df.index)
    common_tcga_ids = set(rnaseq_df.columns).intersection(set(mapping_df.index))
    if len(common_tcga_ids) > 0:
        print("There are common indices between the rnaseq train data and the overall test data")
        print(common_tcga_ids)  # print the common indices
    else:
        print("There are no common indices between the rnaseq train data and the overall test data")
    #
    # # (VAE training data) df containing rnaseq data for samples for which WSI samples are not being used
    # rnaseq_df = pd.DataFrame(mapping_vae_training_df['rnaseq_data'].to_list(), index=mapping_vae_training_df.index).transpose()
    #
    # rnaseq_df.reset_index(inplace=True)
    # rnaseq_df.rename(columns={'index': 'gene_id'}, inplace=True)
    # print("Are all the gene_ids unique: ", rnaseq_df.shape[0] == len(rnaseq_df['gene_id'].unique()))
    # print("Are all the gene_names unique: ", rnaseq_df.shape[0] == len(rnaseq_df['gene_name'].unique()))

    ######################### VAE HYPERPARAMETERS ############################
    input_dim = rnaseq_df.shape[0]  # number of genes (~ 20K)
    intermediate_dim = 512
    lr = 1e-4
    latent_dim = 256  # get from the input opt
    beta = 0.005  # 0.01   # 0: equivalent to standard autoencoder; 1: equivalent to standard VAE

    train_batch_size = 128
    val_batch_size = 8
    test_batch_size = 8
    num_epochs = 200000

    main()

# set_trace()
