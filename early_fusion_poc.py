# code to carry out early fusion using the embeddings already genereted from the WSI and RNASeq data
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pdb import set_trace
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
from sklearn.model_selection import KFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored
# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer
import optuna

do_hpo = True
# do_hpo = False

input_dir = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/early_fusion_inputs/'
checkpoint_dir = 'checkpoint_2024-04-20-08-43-52'
# load embeddings from WSI
wsi_embeddings = pd.read_json(os.path.join(input_dir, 'WSI_embeddings.json'))
# load embeddings from RNASeq
rnaseq_embeddings = pd.read_json(os.path.join(input_dir, 'rnaseq_embeddings_80K.json'))

# concatenate embeddings
combined_embeddings = {}
for tcga_id in wsi_embeddings.columns:
    combined_embeddings[tcga_id] = wsi_embeddings[tcga_id].iloc[0] + rnaseq_embeddings[tcga_id].iloc[0]
combined_embeddings_df = pd.DataFrame([combined_embeddings])
# combined embeddings size = 384 + 256 = 640

# load survival outcome from clinical data
mapping_df = pd.read_json(os.path.join(input_dir, 'mapping_df.json'))
mapping_df = mapping_df.T
# combine 'days_to_death' and 'days_to_last_followup' into a single column
# assuming that for rows where 'days_to_death' is NaN, 'days_to_last_followup' contains the censoring time
mapping_df['time'] = mapping_df['days_to_death'].fillna(mapping_df['days_to_last_followup'])
# NOTE: TCGA-49-6742 seems to have both 'days_to_death' as well as 'days_to_last_followup' as None. So ignoring this for survival analysis
mapping_df = mapping_df.dropna(subset=['time', 'event_occurred'])

# remove that from the wsi and rnaseq combined embeddings too
combined_embeddings_df = combined_embeddings_df.drop(columns=['TCGA-49-6742'])

# converting to entry types recognized by KMF
mapping_df['event_occurred'] = mapping_df['event_occurred'].map({'Dead': True, 'Alive': False})

# # plot histogram of survival time
# plt.figure(1)
# plt.hist(mapping_df['time'], bins=20)
# plt.show()

# set_trace()

# get the train, validation and test splits (those used during the vae training)
# train_tcga_ids = pd.read_csv(os.path.join(input_dir, checkpoint_dir, 'tcga_ids_train.csv'))
train_tcga_ids = np.load(os.path.join(input_dir, checkpoint_dir, 'tcga_ids_train.npy'), allow_pickle=True).tolist()
validation_tcga_ids = np.load(os.path.join(input_dir, checkpoint_dir, 'tcga_ids_val.npy'), allow_pickle=True).tolist()
test_tcga_ids = np.load(os.path.join(input_dir, checkpoint_dir, 'tcga_ids_test.npy'), allow_pickle=True).tolist()
# Note: len(train_tcga_ids) [413] + len(validation_tcga_ids)[52] +  len(test_tcga_ids)[52] = 517  > number of samples in mapping_df
# samples for which WSI processing couldn't be done (e.g., due to mismatch in magnification factor) are not in mapping_df, but may have been used for training VAE


# split mapping_df into train/validation/test sets
mapping_df_train = mapping_df.loc[mapping_df.index.intersection(train_tcga_ids)]
mapping_df_validation = mapping_df.loc[mapping_df.index.intersection(validation_tcga_ids)]
mapping_df_test = mapping_df.loc[mapping_df.index.intersection(test_tcga_ids)]


# gradient boosted model
# prepare structured array for survival data (see https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html)
# survival_data = np.array([(event, time) for event, time in zip(mapping_df['event_occurred'], mapping_df['time'])],
#                          dtype=[('event', bool), ('time', float)])
# survival_data = Surv.from_arrays(event=mapping_df['event_occurred'], time=mapping_df['time'])

survival_data_train = np.array([(event, time) for event, time in zip(mapping_df_train['event_occurred'], mapping_df_train['time'])],
                         dtype=[('event', bool), ('time', float)])
survival_data_train = Surv.from_arrays(event=mapping_df_train['event_occurred'], time=mapping_df_train['time'])

survival_data_validation = np.array([(event, time) for event, time in zip(mapping_df_validation['event_occurred'], mapping_df_validation['time'])],
                         dtype=[('event', bool), ('time', float)])
survival_data_validation = Surv.from_arrays(event=mapping_df_validation['event_occurred'], time=mapping_df_validation['time'])

survival_data_test = np.array([(event, time) for event, time in zip(mapping_df_test['event_occurred'], mapping_df_test['time'])],
                         dtype=[('event', bool), ('time', float)])
survival_data_test = Surv.from_arrays(event=mapping_df_test['event_occurred'], time=mapping_df_test['time'])

# process the combined embeddings appropriate for model.fit()
embeddings_series = combined_embeddings_df.iloc[0]

# DO THE TRAIN/VAL/TEST SPLIT
X_train_list = [embeddings_series[tcga_id] for tcga_id in train_tcga_ids if tcga_id in embeddings_series]
X_validation_list = [embeddings_series[tcga_id] for tcga_id in validation_tcga_ids if tcga_id in embeddings_series]
X_test_list = [embeddings_series[tcga_id] for tcga_id in test_tcga_ids if tcga_id in embeddings_series]

X_train = np.array(X_train_list)
X_validation = np.array(X_validation_list)
X_test = np.array(X_test_list)


# survival regression
# instantiate a GBM from scikit survival package (https://scikit-survival.readthedocs.io/en/stable/user_guide/boosting.html)
# basically this carries out regression using the embeddings as the input features/covariates
# model = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

if do_hpo:
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        learning_rate = trial.suggest_float('learning_rate', 0.1, 1.0, log=True)
        max_depth = trial.suggest_int('max_depth', 1, 5)

        model = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=0
        )
        model.fit(X_train, survival_data_train)

        risk_scores_validation = model.predict(X_validation)
        c_index_validation = concordance_index_censored(
            survival_data_validation['event'],
            survival_data_validation['time'],
            risk_scores_validation
        )[0]

        # print(f"Trial hyperparameters: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
        # print(f"Validation C-index: {c_index_validation}\n")

        return c_index_validation


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_c_index = study.best_value
    print("Best Hyperparameters:", best_params)
    print("Best Validation Concordance Index:", best_c_index)

    # re-train the model on the full training set using the best hyperparameters
    best_model = GradientBoostingSurvivalAnalysis(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        random_state=0
    )
    best_model.fit(X_train, survival_data_train)

    # evaluate on the test set
    risk_scores_test = best_model.predict(X_test)
    c_index_test = concordance_index_censored(
        survival_data_test['event'],
        survival_data_test['time'],
        risk_scores_test
    )[0]
    print("Test Concordance Index:", c_index_test)
    model = best_model
else:
    model = GradientBoostingSurvivalAnalysis(n_estimators=50, learning_rate=0.01, max_depth=2, random_state=0)
    # model = GradientBoostingSurvivalAnalysis(n_estimators=92, learning_rate=0.15, max_depth=2, random_state=0)

print("fitting model")
model.fit(X_train, survival_data_train)
print("model fitting over")

# set_trace()

# get predictions (edit this to get only test set predictions)
risk_scores_train = model.predict(X_train)
risk_scores_validation = model.predict(X_validation)
risk_scores_test = model.predict(X_test)

# model prediction evaluation: concordance index
c_index_train = concordance_index_censored(mapping_df_train['event_occurred'], mapping_df_train['time'], risk_scores_train)
c_index_validation = concordance_index_censored(mapping_df_validation['event_occurred'], mapping_df_validation['time'], risk_scores_validation)
c_index_test = concordance_index_censored(mapping_df_test['event_occurred'], mapping_df_test['time'], risk_scores_test)

# (c-index, number of concordant pairs, number of discordant pairs, number of tied pairs, number of uncomparable pairs)
# For a dataset with N individuals, there are N(N-1)/2 unique pairs.
# The concordance index considers each of these pairs and evaluates whether the observed outcomes are in agreement (concordant) or disagreement (discordant),
# with the outcomes predicted by the model, taking into account censoring
print(f"Train concordance index: {c_index_train}")
print(f"Validation concordance index: {c_index_validation}")
print(f"Test concordance index: {c_index_test}")

# set_trace()

# risk_scores_test = risk_scores_train
# mapping_df_test = mapping_df_train

# risk_scores_test = risk_scores_validation
# mapping_df_test = mapping_df_validation

median_risk_test = np.median(risk_scores_test)
high_risk_test = risk_scores_test >= median_risk_test
low_risk_test = risk_scores_test < median_risk_test

kmf_high_test = KaplanMeierFitter()
kmf_low_test = KaplanMeierFitter()
kmf_test = KaplanMeierFitter()
# kmf_true = KaplanMeierFitter()

kmf_high_test.fit(durations=mapping_df_test['time'][high_risk_test], event_observed=mapping_df_test['event_occurred'][high_risk_test], label='High Risk')
kmf_low_test.fit(durations=mapping_df_test['time'][low_risk_test], event_observed=mapping_df_test['event_occurred'][low_risk_test], label='Low Risk')
# kmf_test.fit(durations=mapping_df_test['time'], event_observed=mapping_df_test['event_occurred'], label='all')
# kmf_true.fit()




# survival plots for the GBM predictions
kmf_high_test.plot_survival_function(color='blue')
kmf_low_test.plot_survival_function(color='red')
# kmf_test.plot_survival_function(color='black')
plt.title('Survival Analysis: High Risk vs Low Risk based on Predicted Risk Scores')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# stratified survival plots for the GBM predictions
# kmf_high.plot_survival_function()
# kmf_low.plot_survival_function()
# plt.title('Survival Analysis: High Risk vs Low Risk based on Predicted Risk Scores')
# plt.xlabel('Time')
# plt.ylabel('Survival Probability')
# plt.show()


# evaluation: survival plots
# kmf = KaplanMeierFitter()
# kmf.fit(mapping_df['time'], event_observed=mapping_df['event_occurred'])
#
#
# kmf.plot_survival_function()
# plt.title('Kaplan-Meier Survival Estimate')
# plt.xlabel('Time (days)')
# plt.ylabel('Survival Probability')
# plt.show()

# evaluation: concordance index + log-rank p-values

set_trace()
