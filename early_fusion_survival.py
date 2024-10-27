import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import os
import optuna
from sklearn.model_selection import KFold
import h5py
from pdb import set_trace

# do_hpo = True
do_hpo = True
num_folds = 10
fusion = True
only_wsi = False
only_omic = False


# checkpoint_dir = 'checkpoint_2024-10-18-06-49-51'
checkpoint_dir = 'checkpoint_2024-10-18-21-57-09'
fold_id = 0
# input_dir = os.path.join('/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/')
input_dir = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/joint_fusion/'
rnaseq_epoch = 2800 #1900 #1500  # epoch that will be used from the saved model

# load embeddings from WSI and rnaseq
wsi_embeddings_all = pd.read_json(os.path.join(input_dir, 'WSI_embeddings.json'))

# load bulk rnaseq embeddings from trained VAE
rnaseq_embeddings_train = pd.read_json(os.path.join(input_dir, checkpoint_dir, f'rnaseq_embeddings_train_{checkpoint_dir}_fold_{fold_id}_epoch_{rnaseq_epoch}.json'))
rnaseq_embeddings_val = pd.read_json(os.path.join(input_dir, checkpoint_dir, f'rnaseq_embeddings_val_{checkpoint_dir}_fold_{fold_id}_epoch_{rnaseq_epoch}.json'))
rnaseq_embeddings_test = pd.read_json(os.path.join(input_dir, checkpoint_dir, f'rnaseq_embeddings_test_{checkpoint_dir}_fold_{fold_id}_epoch_{rnaseq_epoch}.json'))

# combine rnaseq embeddings into a single dataframe
rnaseq_embeddings_all = pd.concat([rnaseq_embeddings_train, rnaseq_embeddings_val, rnaseq_embeddings_test], axis=1)

# to keep only the matched samples, find common TCGA IDs in both RNASeq and WSI embeddings
rnaseq_columns = rnaseq_embeddings_all.columns
wsi_columns = wsi_embeddings_all.columns
common_tcga_ids = rnaseq_columns.intersection(wsi_columns)
print(f"Number of common IDs between RNASeq and WSI datasets: {len(common_tcga_ids)}")

# Concatenate embeddings
combined_embeddings = {}
excluded_columns = rnaseq_columns.symmetric_difference(wsi_columns)
# set_trace()
# excluded_columns = ['TCGA-05-4395', 'TCGA-86-8281']
# excluded_columns = []
# for tcga_id in rnaseq_embeddings_all.columns:  # Iterate over all RNASeq embeddings

# combine embeddings based on whether the task specific model is multimodal or unimodal
tcga_index = 0
for tcga_id in common_tcga_ids:
    tcga_index += 1
    print(f"TCGA ID: {tcga_id}, {tcga_index} of {len(common_tcga_ids)}")
    if tcga_id in excluded_columns:
        continue
    if fusion:
        combined_embeddings[tcga_id] = wsi_embeddings_all[tcga_id].iloc[0] + rnaseq_embeddings_all[tcga_id].iloc[0]
    elif only_wsi:
        combined_embeddings[tcga_id] = wsi_embeddings_all[tcga_id].iloc[0]
    elif only_omic:
        combined_embeddings[tcga_id] = rnaseq_embeddings_all[tcga_id].iloc[0]

combined_embeddings_df = pd.DataFrame([combined_embeddings])

# load file containing mapping information for the TCGA LUAD bulk rnaseq, histology and clinical data (even status and time to event)
h5_file = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/joint_fusion/mapping_data.h5'

with h5py.File(h5_file, 'r') as hdf:
    train_group = hdf['train']
    test_group = hdf['test']
    val_group = hdf['val']

    def extract_data(group):
        ids = []
        days_to_event = []
        event_occurred = []
        for patient_id in group.keys():
            ids.append(patient_id)
            days_to_event.append(group[patient_id]['days_to_event'][()])
            event_occurred.append(group[patient_id]['event_occurred'][()])
        df = pd.DataFrame({
            'tcga_id': ids,
            'time': days_to_event,
            'event_occurred': event_occurred
        }).set_index('tcga_id')

        df = df.drop(excluded_columns, errors='ignore')

        return df

    mapping_df_train = extract_data(train_group)
    mapping_df_val = extract_data(val_group)
    mapping_df_test = extract_data(test_group)

# combine validation and test sets (this is because the validation set was not used for HPO for the rnaseq embedding generation)
mapping_df_test = pd.concat([mapping_df_test, mapping_df_val])

mapping_df_train = mapping_df_train.dropna(subset=['time', 'event_occurred'])
mapping_df_test = mapping_df_test.dropna(subset=['time', 'event_occurred'])


# create the Surv objects for sksurv (package for survival analysis)
survival_data_train = Surv.from_arrays(event=mapping_df_train['event_occurred'], time=mapping_df_train['time'])
survival_data_test = Surv.from_arrays(event=mapping_df_test['event_occurred'], time=mapping_df_test['time'])

embeddings_series = combined_embeddings_df.iloc[0]
X_train = np.array([embeddings_series[tcga_id] for tcga_id in mapping_df_train.index if tcga_id in embeddings_series])
X_test = np.array([embeddings_series[tcga_id] for tcga_id in mapping_df_test.index if tcga_id in embeddings_series])

# Train gradient boosted survival model using the Cox PH loss function
# hyperparameter optimization
if do_hpo:
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 1.0)
        max_depth = trial.suggest_int('max_depth', 1, 5)
        model = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators,
                                                 learning_rate=learning_rate,
                                                 max_depth=max_depth,
                                                 random_state=0)

        kf = KFold(n_splits=num_folds) # use kfold CV for the GBM training
        c_indexes = []
        for train_index, test_index in kf.split(X_train):
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            survival_data_train_fold = survival_data_train[train_index]
            model.fit(X_train_fold, survival_data_train_fold)
            risk_scores_fold = model.predict(X_test_fold)
            c_index_fold = concordance_index_censored(survival_data_train[test_index]['event'],
                                                      survival_data_train[test_index]['time'],
                                                      risk_scores_fold)
            c_indexes.append(c_index_fold[0])

        return np.mean(c_indexes)


    study = optuna.create_study(direction='maximize') # be careful to use "maximize" here are as we are trying to get the maximum CI
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # retrain the best model (obtained from kfold CV) on the full training set
    best_model = GradientBoostingSurvivalAnalysis(n_estimators=best_params['n_estimators'],
                                                  learning_rate=best_params['learning_rate'],
                                                  max_depth=best_params['max_depth'],
                                                  random_state=0)
    best_model.fit(X_train, survival_data_train)
    model = best_model
else:
    # model = GradientBoostingSurvivalAnalysis(n_estimators=31, learning_rate=0.85, max_depth=1, random_state=0)
    # model = GradientBoostingSurvivalAnalysis(n_estimators=72, learning_rate=0.1, max_depth=2, random_state=0)
    model = GradientBoostingSurvivalAnalysis(n_estimators=10, learning_rate=1, max_depth=1, random_state=0)
    model.fit(X_train, survival_data_train)

# obtain risk scores for the train and test splits
risk_scores_train = model.predict(X_train)
risk_scores_test = model.predict(X_test)

# convert 'event_occured' to bool for compatibility with the function used for calculating CI
mapping_df_train['event_occurred'] = mapping_df_train['event_occurred'].astype(bool)
mapping_df_test['event_occurred'] = mapping_df_test['event_occurred'].astype(bool)

c_index_train = concordance_index_censored(mapping_df_train['event_occurred'],
                                           mapping_df_train['time'],
                                           risk_scores_train)
c_index_test = concordance_index_censored(mapping_df_test['event_occurred'],
                                          mapping_df_test['time'],
                                          risk_scores_test)

print(f"Train concordance index: {c_index_train}") # format (CI, num of concordant pairs, num of discordant pairs, tied pairs, incomparable pairs)
print(f"Test concordance index: {c_index_test}")

# get survival plots and carry out log-rank test

median_risk_test = np.median(risk_scores_test)
# stratify patients into two groups based on the predicted median risk scores
high_risk_test = risk_scores_test >= median_risk_test
low_risk_test = risk_scores_test < median_risk_test

# create kaplan meier plots for the two patient groups
kmf_high_test = KaplanMeierFitter()
kmf_low_test = KaplanMeierFitter()
kmf_high_test.fit(durations=mapping_df_test['time'][high_risk_test],
                  event_observed=mapping_df_test['event_occurred'][high_risk_test],
                  label='High Risk')
kmf_low_test.fit(durations=mapping_df_test['time'][low_risk_test],
                 event_observed=mapping_df_test['event_occurred'][low_risk_test],
                 label='Low Risk')

log_rank_test = logrank_test(
    mapping_df_test['time'][high_risk_test],
    mapping_df_test['time'][low_risk_test],
    event_observed_A=mapping_df_test['event_occurred'][high_risk_test],
    event_observed_B=mapping_df_test['event_occurred'][low_risk_test],
)
p_value = log_rank_test.p_value
# set_trace()
kmf_high_test.plot_survival_function(color='blue')
kmf_low_test.plot_survival_function(color='red')
plt.title('Patient stratification based on predicted risk scores\nLog-rank test p-value: {:.4f}'.format(p_value))
# plt.title('Log-rank test p-value: {:.4f}'.format(p_value))
plt.xlabel('Time (days)')
plt.ylabel('Survival probability')
plt.xlim([0, None])
plt.ylim([0, 1])
plt.grid(True)
if fusion:
    plt.savefig(f'KM_{rnaseq_epoch}_early_fusion.kfoldCV.png')
elif only_wsi:
    plt.savefig(f'KM_{rnaseq_epoch}_only_wsi.kfoldCV.png')
elif only_omic:
    plt.savefig(f'KM_{rnaseq_epoch}_only_omic.kfoldCV.png')

# print(f"Test concordance index: {c_index_test}")
