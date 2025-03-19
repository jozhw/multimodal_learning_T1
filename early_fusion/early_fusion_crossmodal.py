import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
# from lifelines.plotting import plot_schoenfeld_residuals
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing
import random
import optuna
import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
import torchtuples as tt
from pycox.models import CoxPH
# from pycox.evaluation import concordance_index
import argparse

from pdb import set_trace

seed_value = 142  
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Ensures reproducible (but potentially slower) behavior on GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
use_system = 'cluster' # cluster or laptop

def parse_args():
    parser = argparse.ArgumentParser(description='Model training and evaluation parameters')
    
    # Add arguments with appropriate defaults matching your configuration
    parser.add_argument('--check_PH_assumptions', action='store_true', 
                        help='Check proportional hazards assumptions')
    parser.add_argument('--plot_embs', action='store_true',
                        help='Plot embeddings')
    parser.add_argument('--plot_survival', action='store_true',
                        help='Plot survival curves')
    parser.add_argument('--drop_outliers', action='store_true',
                        help='Drop very high values of survival duration in the data')
    parser.add_argument('--do_bootstrap', action='store_true', default=True,
                        help='Perform bootstrap resampling')
    parser.add_argument('--do_hpo', action='store_true', 
                        help='Perform hyperparameter optimization')
    parser.add_argument('--apply_pca', action='store_true',
                        help='Apply PCA dimensionality reduction')
    parser.add_argument('--use_model', type=str, default='snn', choices=['snn', 'gbst'],
                        help='Model type: gradient boosted survival tree (gbst) or survival neural network (snn)')
    # parser.add_argument('--mode', type=str, default='only_wsi', 
    #                     choices=['rnaseq_wsi', 'only_rnaseq', 'only_wsi'],
    #                     help='Data modality to use')
    
    return parser.parse_args()

args = parse_args()

# on laptop
if use_system == 'laptop':
    input_dir = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/early_fusion_inputs/'
    # id = "2025-02-27-01-12-39_fold_1_epoch_1000"

if use_system == 'cluster':
    # on polaris
    input_dir = '/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/early_fusion_inputs/'
    # id = "2025-02-27-01-12-39_fold_1_epoch_3000" # working
    id = "2025-02-27-01-41-45_fold_1_epoch_3000" # rnaseq and wsi embeddings distributions more similar

# file paths for rnaseq embeddings (generated using generate_rnaseq_embeddings_kfoldCV.py)
train_file = os.path.join(input_dir, f"rnaseq_embeddings_train_checkpoint_{id}.json")
val_file = os.path.join(input_dir, f"rnaseq_embeddings_val_checkpoint_{id}.json")
test_file = os.path.join(input_dir, f"rnaseq_embeddings_test_checkpoint_{id}.json")

train_file = os.path.join(input_dir, f"rnaseq_embeddings_train_checkpoint_hp_subset_{id}.json")
val_file = os.path.join(input_dir, f"rnaseq_embeddings_val_checkpoint_hp_subset_{id}.json")
test_file = os.path.join(input_dir, f"rnaseq_embeddings_test_checkpoint_hp_subset_{id}.json")


# load the rnaseq embeddings
train_rna_embs = pd.read_json(train_file).T
val_rna_embs = pd.read_json(val_file).T
test_rna_embs = pd.read_json(test_file).T

print("Train embeddings shape (RNASeq):", train_rna_embs.shape)

# combine val and test sets into a single held out set (the validation dataset wasn't used in training; kfoldCV was used with the training dataset only)
print("concatenating val and test sets to create a single held out dataset")
test_rna_embs = pd.concat([val_rna_embs, test_rna_embs], axis=0)  # concatenate along rows (samples)
test_validation_tcga_ids = list(val_rna_embs.index) + list(test_rna_embs.index)
print("Test + Validation embeddings shape (RNASeq):", test_rna_embs.shape)

print("Loading WSI data")

# save as parquet file once, and use that for subsequent inference runs
# wsi_embs = pd.read_json(os.path.join(input_dir, 'WSI_embeddings_uni_31Jan_1000tiles.truncated.json'), lines=True).T
# wsi_embs.to_parquet(os.path.join(input_dir, 'WSI_embeddings_uni_31Jan_1000tiles.truncated.json.parquet'))
# set_trace()
# for the subsequent inference runs, use:
wsi_embs = pd.read_parquet(os.path.join(input_dir, 'WSI_embeddings_uni_31Jan_1000tiles.truncated.json.parquet'))

# check shapes
# embedding_shapes = wsi_embs.iloc[:, 0].apply(lambda x: np.array(x).shape)
# unique_shapes = embedding_shapes.unique()
# get the slide level embeddings by averaging the tile level embeddings
print("obtaining slide level embeddings by averaging the tile level embeddings")

wsi_embs["slide_embedding"] = wsi_embs.iloc[:, 0].apply(lambda x: np.mean(x, axis=0))
wsi_embs = wsi_embs.drop(columns=[0])

# remove the problematic WSIs (with penmarks etc)
exclude_ids_wsi = ['TCGA-86-6851', 'TCGA-86-7701', 'TCGA-86-7711', 'TCGA-86-7713', 'TCGA-86-7714', 'TCGA-86-7953', 'TCGA-86-7954', 'TCGA-86-7955', 
                  'TCGA-86-8055', 'TCGA-86-8056', 'TCGA-86-8073', 'TCGA-86-8074', 'TCGA-86-8075', 'TCGA-86-8076', 'TCGA-86-8278', 'TCGA-86-8279', 
                  'TCGA-86-8280', 'TCGA-86-A4P7', 'TCGA-86-A4P8']
wsi_embs = wsi_embs.drop(index=exclude_ids_wsi, errors='ignore')

# find common indices between the rnaseq and WSI embeddings
# common_train_indices = train_rna_embs.index.intersection(wsi_embs.index)
# common_test_indices = test_rna_embs.index.intersection(wsi_embs.index)

common_train_indices = train_rna_embs.index.intersection(wsi_embs.index).to_series().sample(frac=1, random_state=seed_value).index
common_test_indices = test_rna_embs.index.intersection(wsi_embs.index).to_series().sample(frac=1, random_state=seed_value).index


# select only common indices for training and test/validation sets
train_rna_embs = train_rna_embs.loc[common_train_indices]
test_rna_embs = test_rna_embs.loc[common_test_indices]

train_wsi_embs = wsi_embs.loc[common_train_indices]
test_wsi_embs = wsi_embs.loc[common_test_indices]

# prepare survival data
mapping_df = pd.read_json(os.path.join(input_dir, 'mapping_df_31Jan_1000tiles.json')).T
mapping_df['event_occurred'] = mapping_df['event_occurred'].map({'Dead': 1, 'Alive': 0}).astype(bool)

train_survival = mapping_df.loc[common_train_indices]
test_survival = mapping_df.loc[common_test_indices]

train_surv = Surv.from_arrays(train_survival["event_occurred"], train_survival["time"])
test_surv = Surv.from_arrays(test_survival["event_occurred"], test_survival["time"])

print("Train embeddings shape (WSI):", train_wsi_embs.shape)
print("Test + Validation embeddings shape (WSI):", test_wsi_embs.shape)
# print("Test embeddings shape (WSI):", test_embs.shape)

# normalization of embeddings
scaler_rna = StandardScaler()
train_rna_norm = scaler_rna.fit_transform(train_rna_embs)
test_rna_norm = scaler_rna.transform(test_rna_embs)

scaler_wsi = StandardScaler()
train_wsi_embs_array = np.stack(train_wsi_embs["slide_embedding"].values)
test_wsi_embs_array = np.stack(test_wsi_embs["slide_embedding"].values)

train_wsi_norm = scaler_wsi.fit_transform(train_wsi_embs_array)
test_wsi_norm = scaler_wsi.transform(test_wsi_embs_array)


# define the cross modal model that will be trained using a contrastive loss
class CrossModelGatedEncoder(nn.Module):
    def __init__(self, emb_dim=1024, hidden_dim=256, proj_dim=128):
        super().__init__()
        self.rna_attn = nn.Linear(emb_dim, emb_dim)
        self.wsi_attn = nn.Linear(emb_dim, emb_dim)

        self.rna_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

        self.wsi_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    #     # Apply custom weight initialization to all layers
    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     """Weight initialization: Xavier uniform for weights, zeros for biases."""
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight)
    #         nn.init.zeros_(m.bias)

    def forward(self, rna_emb, wsi_emb):
        rna_weighted = rna_emb * torch.sigmoid(self.rna_attn(rna_emb))
        wsi_weighted = wsi_emb * torch.sigmoid(self.wsi_attn(wsi_emb))

        rna_proj = F.normalize(self.rna_encoder(rna_weighted), dim=1)
        wsi_proj = F.normalize(self.wsi_encoder(wsi_weighted), dim=1)

        return rna_proj, wsi_proj # returns normalized attention-weighted embeddings

# contrastive Loss (NT-Xent)
def nt_xent_loss(z1, z2, temperature=0.1): # z1 and z2 are normalized embeddings from two modalities obtained from the encoder (embeddings from the same samples in different modalities)
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    # sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature  # compute the similarity score
    z_norm = F.normalize(z, dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', z_norm, z_norm) / temperature
    # mask out the diagonal entries to not use self-similarities
    mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z.device)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    # labels pointing each example to its positive counterpart
    labels = torch.cat([
        torch.arange(batch_size, device=z1.device) + batch_size,
        torch.arange(batch_size, device=z1.device)
    ])
    # set_trace()
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# normalize embeddings
scaler_rna = StandardScaler()
scaler_wsi = StandardScaler()

train_rna = torch.tensor(scaler_rna.fit_transform(train_rna_embs.values)).float()
train_wsi = torch.tensor(scaler_wsi.fit_transform(train_wsi_embs_array)).float()

test_rna = torch.tensor(scaler_rna.transform(test_rna_embs.values)).float()
test_wsi = torch.tensor(scaler_wsi.transform(test_wsi_embs_array)).float()

# train the cross-modal encoder
batch_size = 32
epochs = 200
encoder = CrossModelGatedEncoder(1024, 512, 256)
optimizer = optim.Adam(encoder.parameters(), lr=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
# scheduler = ExponentialLR(optimizer, gamma=0.99)  

# train the contrastive encoder to generate cross modality informed embeddings
encoder.train()
for epoch in range(epochs):
    permutation = torch.randperm(train_rna.size(0))
    epoch_loss = 0.0
    for i in range(0, train_rna.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        rna_batch, wsi_batch = train_rna[indices], train_wsi[indices]
        optimizer.zero_grad()
        rna_proj, wsi_proj = encoder(rna_batch, wsi_batch)
        # set_trace()
        loss = nt_xent_loss(rna_proj, wsi_proj)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()  # Update the learning rate after each epoch

    current_lr = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}')

# get fused embeddings for GBST
encoder.eval()
# with torch.no_grad():
#     fused_train = torch.cat(encoder(train_rna, train_wsi), dim=1).numpy()
#     fused_test = torch.cat(encoder(test_rna, test_wsi), dim=1).numpy()

with torch.no_grad():
    rna_proj, wsi_proj = encoder(train_rna, train_wsi)
    rna_weight = 0.8 #0.5 #0.8
    wsi_weight = 0.2 #0.5 #0.2
    fused_train = (rna_proj * rna_weight + wsi_proj * wsi_weight).numpy()

    rna_proj_test, wsi_proj_test = encoder(test_rna, test_wsi)
    fused_test = (rna_proj_test * rna_weight + wsi_proj_test * wsi_weight).numpy()

# set_trace()

# wsi_df = pd.DataFrame(wsi_proj.numpy(), columns=[f"{i}" for i in range(wsi_proj.shape[1])])
# rna_df = pd.DataFrame(rna_proj.numpy(), columns=[f"{i}" for i in range(rna_proj.shape[1])])

# wsi_df.index = train_wsi_embs.index
# rna_df.index = train_rna_embs.index

# # Save DataFrames as JSON files
# wsi_df.to_json(f"wsi_embeddings_crossmodal_{wsi_proj.shape[1]}_{timestamp}.json", orient="index")
# rna_df.to_json(f"rna_embeddings_crossmodal_{wsi_proj.shape[1]}_{timestamp}.json", orient="index")

print("WSI and RNA-Seq embeddings saved as JSON files.")

if args.plot_embs:
    def plot_embedding_statistics(rna_embs, wsi_embs, split):
        """
        Plots the statistical distributions (min, max, mean, std) of RNA-Seq and WSI embeddings.

        Parameters:
        rna_embs (pd.DataFrame): Normalized RNA-Seq embeddings (samples x features).
        wsi_embs (pd.DataFrame): Normalized WSI embeddings (samples x features).

        Returns:
        None
        """
        # Compute statistics for RNA-Seq embeddings
        rna_min = rna_embs.min(axis=1)[0] # mean returns both the values as well as the indices
        rna_max = rna_embs.max(axis=1)[0]
        rna_mean = rna_embs.mean(axis=1)
        rna_std = rna_embs.std(axis=1)

        # Compute statistics for WSI embeddings
        wsi_min = wsi_embs.min(axis=1)[0]
        wsi_max = wsi_embs.max(axis=1)[0]
        wsi_mean = wsi_embs.mean(axis=1)
        wsi_std = wsi_embs.std(axis=1)

        plt.figure(figsize=(12, 6))

        # Plot RNA-seq statistics
        plt.plot(rna_min, label='RNA-seq Min', color='blue', linestyle='-')
        plt.plot(rna_max, label='RNA-seq Max', color='blue', linestyle='--')
        plt.plot(rna_mean, label='RNA-seq Mean', color='blue', linestyle='-.')
        plt.plot(rna_std, label='RNA-seq Std', color='blue', linestyle=':')

        # Plot WSI statistics
        plt.plot(wsi_min, label='WSI Min', color='red', linestyle='-')
        plt.plot(wsi_max, label='WSI Max', color='red', linestyle='--')
        plt.plot(wsi_mean, label='WSI Mean', color='red', linestyle='-.')
        plt.plot(wsi_std, label='WSI Std', color='red', linestyle=':')

        plt.xlabel('TCGA Sample Index')
        plt.ylabel('Embedding Value')
        plt.title('Statistics of RNA-seq and WSI Embeddings Across TCGA Samples')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"embs_stats_{split}_{timestamp}.png")
        plt.show()

        set_trace()


    # Function to plot distributions
    def plot_embedding_distributions(rna_embs, wsi_embs, split):
        plt.figure(figsize=(12, 6))
        plt.hist(rna_embs.values.flatten(), bins=100, alpha=0.5, label="RNA-Seq Embeddings", color='blue', density=True)
        plt.hist(wsi_embs.values.flatten(), bins=100, alpha=0.5, label="WSI Embeddings", color='red', density=True)

        plt.xlabel("Embedding Value")
        plt.ylabel("Density")
        plt.title("Distribution of RNA-Seq and WSI Embeddings (Train Dataset)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"embs_dist_{split}_{timestamp}.png")
        plt.show()

    
    # plot_embedding_statistics([rna_proj, rna_proj_test], [wsi_proj, wsi_proj_test], "train + test")
    plot_embedding_statistics(torch.cat([rna_proj, rna_proj_test], dim=0), torch.cat([wsi_proj, wsi_proj_test], dim=0), "train + test")
    # set_trace()
    # plot_embedding_distributions(train_embs, pd.DataFrame(train_wsi_embs_array, index=train_wsi_embs.index), "train")


# set_trace()
# GBST Survival Model
if args.do_hpo:
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        max_depth = trial.suggest_int('max_depth', 1, 3)
        # max_features = trial.suggest_float('max_features', 0.2, 1.0)

        model = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_features=0.5,
            random_state=seed_value
        )

        kf = KFold(n_splits=3, shuffle=True, random_state=seed_value)
        c_indices = []

        for train_idx, val_idx in kf.split(fused_train):
            X_train_fold, X_val_fold = fused_train[train_idx], fused_train[val_idx]
            y_train_fold = train_surv[train_idx]
            y_val_fold = train_surv[val_idx]

            model.fit(X_train_fold, y_train_fold)
            val_preds = model.predict(X_val_fold)

            c_index = concordance_index_censored(
                y_val_fold['event'], y_val_fold['time'], val_preds
            )[0]
            c_indices.append(c_index)

        return np.mean(c_indices)

    num_cpu_cores = multiprocessing.cpu_count()
    study = optuna.create_study(direction='maximize') #, sampler=optuna.samplers.QMCSampler()) # RandomSampler, GPSampler, NSGAIISampler, QMCSampler, CmaEsSampler
    print(f"Sampler is {study.sampler.__class__.__name__}")
    study.optimize(objective, n_trials=400, n_jobs=num_cpu_cores)

    print("Best parameters found: ", study.best_params)
    model = GradientBoostingSurvivalAnalysis(**study.best_params, random_state=seed_value)
else:
    # use previously optimized parameters or defaults
    model = GradientBoostingSurvivalAnalysis(
        n_estimators=87, #88,
        learning_rate=0.085, #0.08975,
        max_depth=3, #3,
        # max_features=0.5,
        random_state=seed_value
    )

    # # for only rnaseq
    # model = GradientBoostingSurvivalAnalysis(
    #     n_estimators=97,
    #     learning_rate=0.07976,
    #     max_depth=3,
    #     max_features=0.5,
    #     random_state=seed_value
    # )

# train the final model on the entire training set
model.fit(fused_train, train_surv)

risk_scores_train = model.predict(fused_train)
risk_scores_test = model.predict(fused_test)

c_index_train = concordance_index_censored(train_survival['event_occurred'], train_survival['time'], risk_scores_train)[0]
c_index_test = concordance_index_censored(test_survival['event_occurred'], test_survival['time'], risk_scores_test)[0]

print(f'Train CI: {c_index_train:.4f}')
print(f'Test CI: {c_index_test:.4f}')

print(f"Train Concordance Index: {c_index_train:.4f}")
print(f"Test Concordance Index: {c_index_test:.4f}")


time_min = test_survival["time"].min()
time_max = test_survival["time"].max()
time_grid = np.linspace(time_min, time_max, 50, endpoint=False)  

times = np.sort(test_survival["time"].astype(float).unique())[:-1]
auc_scores, mean_auc = cumulative_dynamic_auc(train_surv, test_surv, risk_scores_test, times)
# plot the time-dependent AUC curve
plt.figure()
plt.plot(times, auc_scores, 'k-+')
plt.axhline(y=mean_auc, color='black', linestyle='--', label=f"Mean AUC: {mean_auc:.2f}")
plt.xlabel("Time")
plt.ylabel("AUC")
plt.title("Time-Dependent AUC")
plt.grid(True)
plt.savefig(f"constrastive_AUC_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

# set_trace()


# Kaplan-Meier Analysis
median_risk = np.median(risk_scores_test)
high_risk = risk_scores_test >= median_risk
low_risk = risk_scores_test < median_risk

kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()
# set_trace()
kmf_high.fit(test_survival["time"][high_risk.flatten()], test_survival["event_occurred"][high_risk.flatten()], label="High Risk")
kmf_low.fit(test_survival["time"][low_risk.flatten()], test_survival["event_occurred"][low_risk.flatten()], label="Low Risk")

# compute log-rank p-value
logrank_p_value = logrank_test(test_survival["time"][high_risk.flatten()], 
                               test_survival["time"][low_risk.flatten()],
                               test_survival["event_occurred"][high_risk.flatten()], 
                               test_survival["event_occurred"][low_risk.flatten()]).p_value

print(f"log-rank test p-value: {logrank_p_value:.5f}")
# set_trace()
plt.figure()
# kmf_high.plot()
# kmf_low.plot()
kmf_high.plot_survival_function(color='blue', ci_show=False)
kmf_low.plot_survival_function(color='red', ci_show=False)
plt.title(f"CI: {c_index_test:.3f},\nlog-rank test p-value: {logrank_p_value:.5f}")
plt.xlim(left=0)
plt.ylim(top=1, bottom=0)
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.savefig(f"constrastive_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()


if args.do_bootstrap:
    n_bootstraps = 100
    rng = np.random.RandomState(seed=seed_value)
    c_indices_bootstrap = []

    # store vars for KM curves
    time_grid = np.linspace(0, np.max(test_survival["time"]), 10000)
    km_curves_high_risk = []
    km_curves_low_risk = []

    test_survival["time"] = test_survival["time"].astype(float)
    test_survival["event_occurred"] = test_survival["event_occurred"].astype(bool)

    # bootstrapping
    for _ in range(n_bootstraps):
        # resample the test set with replacement
        X_resampled, y_resampled = resample(fused_test, test_survival, random_state=rng)
        
        # predict risk scores for the resampled test set
        risk_scores = model.predict(X_resampled)
        
        # calculate CI
        c_index = concordance_index_censored(y_resampled["event_occurred"], y_resampled["time"], risk_scores)[0]
        c_indices_bootstrap.append(c_index)

        # KM fit for resampled dataset
        median_risk = np.median(risk_scores)
        high_risk = risk_scores >= median_risk
        low_risk = risk_scores < median_risk

        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        kmf_high.fit(y_resampled["time"][high_risk], y_resampled["event_occurred"][high_risk])
        kmf_low.fit(y_resampled["time"][low_risk], y_resampled["event_occurred"][low_risk])

        # store survival probabilities at time grid
        km_curves_high_risk.append(kmf_high.predict(time_grid))
        km_curves_low_risk.append(kmf_low.predict(time_grid))

    c_indices_bootstrap = np.array(c_indices_bootstrap)
    km_curves_high_risk = np.array(km_curves_high_risk)
    km_curves_low_risk = np.array(km_curves_low_risk)

    # compute median and confidence intervals for KM plots
    km_median_high = np.percentile(km_curves_high_risk, 50, axis=0)
    km_lower_high = np.percentile(km_curves_high_risk, 2.5, axis=0)
    km_upper_high = np.percentile(km_curves_high_risk, 97.5, axis=0)

    km_median_low = np.percentile(km_curves_low_risk, 50, axis=0)
    km_lower_low = np.percentile(km_curves_low_risk, 2.5, axis=0)
    km_upper_low = np.percentile(km_curves_low_risk, 97.5, axis=0)

    mean_c_index = np.mean(c_indices_bootstrap)
    std_c_index = np.std(c_indices_bootstrap)
    ci_lower = np.percentile(c_indices_bootstrap, 2.5)
    ci_upper = np.percentile(c_indices_bootstrap, 97.5)

    print(f"Mean C-index: {mean_c_index:.3f}")
    print(f"Standard Deviation of C-index: {std_c_index:.3f}")
    print(f"95% Confidence Interval: ({ci_lower:.3f}, {ci_upper:.3f})")

    # plot c indices across bootstrapped samples
    plt.figure(figsize=(10, 6))
    plt.boxplot(c_indices_bootstrap, vert=True)
    plt.title('Bootstrap Distribution of C-index')
    plt.xlabel('C-index')
    plt.ylim(0.4, 0.8)
    plt.savefig(f"bootstrapped_cindex_contrastive_{timestamp}.png", dpi=300, bbox_inches='tight')
    # plt.show()

    # extract high-risk and low-risk groups from the bootstrapped KM plot
    high_risk_times = test_survival["time"][risk_scores >= np.median(risk_scores)]
    high_risk_events = test_survival["event_occurred"][risk_scores >= np.median(risk_scores)]
    low_risk_times = test_survival["time"][risk_scores < np.median(risk_scores)]
    low_risk_events = test_survival["event_occurred"][risk_scores < np.median(risk_scores)]

    logrank_result = logrank_test(high_risk_times, low_risk_times, high_risk_events, low_risk_events)
    logrank_p_value = logrank_result.p_value

    plt.figure(figsize=(8, 6))
    plt.plot(time_grid, km_median_high, label="High Risk", color="red", linewidth=2)
    plt.fill_between(time_grid, km_lower_high, km_upper_high, color="red", alpha=0.3) #, label="95% CI (High Risk)")

    plt.plot(time_grid, km_median_low, label="Low Risk", color="blue", linewidth=2)
    plt.fill_between(time_grid, km_lower_low, km_upper_low, color="blue", alpha=0.3) #, label="95% CI (Low Risk)")

    # plt.title("Bootstrapped Kaplan-Meier Survival Curves")
    plt.title(f"Bootstrapped Kaplan-Meier Survival Curves\nLog-rank test p-value: {logrank_p_value:.3f}")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"bootstrapped_km_contrastive_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # print(f"Final Log-rank test p-value: {logrank_p_value:.4f}")



set_trace()

