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
from sksurv.metrics import concordance_index_censored
import os
import random
import optuna
import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
from pdb import set_trace

seed_value = 42  
random.seed(seed_value)
np.random.seed(seed_value)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
use_system = 'cluster' # cluster or laptop
do_hpo = False
# do_hpo = False
check_PH_assumptions = False
plot_embs = False
plot_survival = False
drop_outliers = False # drop the very high values of survival duration in the data
do_bootstrap = True
kronecker_product_fusion = False
hadamard_product_fusion = False
mode = 'only_rnaseq' # 'rnaseq_wsi', 'only_rnaseq', 'only_wsi'

# on laptop
if use_system == 'laptop':
    input_dir = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/early_fusion_inputs/'
    # id = "2025-02-09-22-42-55_fold_3_epoch_1700"
    # id = "2025-02-13-17-57-25_fold_3_epoch_3900" # probably the HPO run names were messed up due to overwriting (ran on 8 GPUs)
    # id = "2025-02-14-07-29-39_fold_2_epoch_2500"
    id = "2025-02-14-07-29-39_fold_2_epoch_1500"

if use_system == 'cluster':
    # on polaris
    input_dir = '/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/early_fusion_inputs/'
    # id = "2025-02-09-22-42-55_fold_3_epoch_1700"
    # id = "2025-02-13-17-57-25_fold_3_epoch_3900"
    id = "2025-02-14-07-29-39_fold_2_epoch_1500"

# file paths for rnaseq embeddings (generated using generate_rnaseq_embeddings_kfoldCV.py)
train_file = os.path.join(input_dir, f"rnaseq_embeddings_train_checkpoint_{id}.json")
val_file = os.path.join(input_dir, f"rnaseq_embeddings_val_checkpoint_{id}.json")
test_file = os.path.join(input_dir, f"rnaseq_embeddings_test_checkpoint_{id}.json")

# load the rnaseq embeddings
train_embs = pd.read_json(train_file).T
val_embs = pd.read_json(val_file).T
test_embs = pd.read_json(test_file).T

print("Train embeddings shape (RNASeq):", train_embs.shape)

# print("Test embeddings shape (RNASeq):", test_embs.shape)

# combine val and test sets into a single held out set (the validation dataset wasn't used in training; kfoldCV was used with the training dataset only)
print("concatenating val and test sets to create a single held out dataset")
test_validation_embs = pd.concat([val_embs, test_embs], axis=0)  # concatenate along rows (samples)
test_validation_tcga_ids = list(val_embs.index) + list(test_embs.index)
print("Test + Validation embeddings shape (RNASeq):", test_validation_embs.shape)

# load embeddings from WSI and rnaseq
print("loading WSI tile level embeddings")
# wsi_embs = pd.read_json(os.path.join(input_dir, 'WSI_embeddings_average_uni_29Jan.rounded.json')).T
wsi_embs = pd.read_json(os.path.join(input_dir, 'WSI_embeddings_average_uni_30Jan_1000tiles.rounded.json')).T
# get the slide level embeddings by averaging the tile level embeddings
print("obtaining slide level embeddings by averaging the tile level embeddings")
wsi_embs["slide_level_embedding"] = wsi_embs.apply(lambda row: np.mean(np.stack(row.values), axis=0), axis=1)
wsi_embs = wsi_embs[["slide_level_embedding"]].rename(columns={"slide_level_embedding": "slide_embedding"})

# edited to handle the scenario where the TCGA samples for the WSI embeddings are a subset of the RNASeq embeddings  
# find common indices between the rnaseq and WSI embeddings
common_train_indices = train_embs.index.intersection(wsi_embs.index)
common_test_validation_indices = test_validation_embs.index.intersection(wsi_embs.index)

# Select only common indices for training and test/validation sets
train_embs = train_embs.loc[common_train_indices]
test_validation_embs = test_validation_embs.loc[common_test_validation_indices]

# using wsi_embs for both as there was no training done
train_wsi_embs = wsi_embs.loc[common_train_indices]
test_validation_wsi_embs = wsi_embs.loc[common_test_validation_indices]

print("Train embeddings shape (WSI):", train_wsi_embs.shape)
print("Test + Validation embeddings shape (WSI):", test_validation_wsi_embs.shape)
# print("Test embeddings shape (WSI):", test_embs.shape)


# embeddings from the wsi and the rnaseq modality may have different min/max values
# need to normalize them to bring them to the same scale

# a scaler that does nothing
class IdentityScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        return X

###### normalize rnaseq embeddings based on training set
# use min/max across all the embeddings
# for fit_transform() input needs to be of shape n_samples x n_features
print("scaling rnaseq embeddings")
# scaler_rna = MinMaxScaler(feature_range=(-1, 1))
# scaler = RobustScaler()
# scaler_rna = StandardScaler()
scaler_rna = IdentityScaler()
train_embs_normalized = pd.DataFrame(scaler_rna.fit_transform(train_embs), index=train_embs.index) # fit only using the training data
test_validation_embs_normalized = pd.DataFrame(scaler_rna.transform(test_validation_embs),
                                               index=test_validation_embs.index)

# Normalize WSI embeddings using training data
print("scaling WSI embeddings")
# scaler_wsi = MinMaxScaler(feature_range=(-1, 1))
# scaler = RobustScaler()
# scaler_wsi = StandardScaler()
scaler_wsi = IdentityScaler()
# train_wsi_embs = wsi_embs.loc[train_embs.index]  # Select only training samples
# # train_wsi_embs_normalized = pd.DataFrame(scaler_wsi.fit_transform(list(train_wsi_embs["slide_embedding"])), index=train_wsi_embs.index)
# # test_validation_wsi_embs_normalized = pd.DataFrame(scaler_wsi.transform(list(wsi_embs.loc[test_validation_embs.index]["slide_embedding"])),
# #                                                    index=test_validation_embs.index)

# train_wsi_embs_array = np.stack(train_wsi_embs["slide_embedding"].values)
# test_validation_wsi_embs_array = np.stack(wsi_embs.loc[test_validation_embs.index]["slide_embedding"].values)
# train_wsi_embs_normalized = pd.DataFrame(scaler_wsi.fit_transform(train_wsi_embs_array), index=train_wsi_embs.index)
# test_validation_wsi_embs_normalized = pd.DataFrame(scaler_wsi.transform(test_validation_wsi_embs_array), index=test_validation_embs.index)


# ensure non-empty arrays before stacking
if not train_wsi_embs.empty:
    train_wsi_embs_array = np.stack(train_wsi_embs["slide_embedding"].values)
    train_wsi_embs_normalized = pd.DataFrame(scaler_wsi.fit_transform(train_wsi_embs_array), index=train_wsi_embs.index)
else:
    train_wsi_embs_normalized = pd.DataFrame()

if not test_validation_wsi_embs.empty:
    test_validation_wsi_embs_array = np.stack(test_validation_wsi_embs["slide_embedding"].values)
    test_validation_wsi_embs_normalized = pd.DataFrame(scaler_wsi.transform(test_validation_wsi_embs_array), index=test_validation_wsi_embs.index)
else:
    test_validation_wsi_embs_normalized = pd.DataFrame()

# concatenate normalized embeddings to obtain multimodal embeddings
print("concatenating embeddings")
if mode == 'rnaseq_wsi':
    if kronecker_product_fusion:
        from sklearn.decomposition import PCA

        # Define reduced dimension size
        reduced_dim = 32 #256  
        # PCA for dimensionality reduction
        pca_rnaseq = PCA(n_components=reduced_dim)
        pca_wsi = PCA(n_components=reduced_dim)

        # Fit PCA on training embeddings and transform both train and test sets
        train_rnaseq_reduced = pca_rnaseq.fit_transform(train_embs_normalized)
        train_wsi_reduced = pca_wsi.fit_transform(train_wsi_embs_normalized)

        test_validation_rnaseq_reduced = pca_rnaseq.transform(test_validation_embs_normalized)
        test_validation_wsi_reduced = pca_wsi.transform(test_validation_wsi_embs_normalized)

        # Apply Kronecker Product
        # Kronecker fusion should be applied row-wise, preserving the number of samples
        X_train = np.array([np.kron(rna_row, wsi_row) for rna_row, wsi_row in zip(train_rnaseq_reduced, train_wsi_reduced)])
        X_test_validation = np.array([np.kron(rna_row, wsi_row) for rna_row, wsi_row in zip(test_validation_rnaseq_reduced, test_validation_wsi_reduced)])


        print(f"Kronecker-fused training set shape: {X_train.shape}")
        print(f"Kronecker-fused test/validation set shape: {X_test_validation.shape}")

    elif hadamard_product_fusion:
        # Element-wise multiplication (Hadamard product)
        X_train = train_embs_normalized * train_wsi_embs_normalized  
        X_test_validation = test_validation_embs_normalized * test_validation_wsi_embs_normalized  

    else:
        X_train = pd.concat([train_embs_normalized, train_wsi_embs_normalized], axis=1)
        X_test_validation = pd.concat([test_validation_embs_normalized, test_validation_wsi_embs_normalized], axis=1)
elif mode == 'only_wsi':
    X_train = pd.concat([train_wsi_embs_normalized], axis=1)
    X_test_validation = pd.concat([test_validation_wsi_embs_normalized], axis=1)
if mode == 'only_rnaseq':
    X_train = pd.concat([train_embs_normalized], axis=1)
    X_test_validation = pd.concat([test_validation_embs_normalized], axis=1)

if plot_embs:
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
        rna_min = rna_embs.min(axis=1)
        rna_max = rna_embs.max(axis=1)
        rna_mean = rna_embs.mean(axis=1)
        rna_std = rna_embs.std(axis=1)

        # Compute statistics for WSI embeddings
        wsi_min = wsi_embs.min(axis=1)
        wsi_max = wsi_embs.max(axis=1)
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

    
    # plot_embedding_statistics(test_validation_embs_normalized, test_validation_wsi_embs_normalized)
    plot_embedding_statistics(train_embs, pd.DataFrame(train_wsi_embs_array, index=train_wsi_embs.index), "train")
    # set_trace()
    plot_embedding_distributions(train_embs, pd.DataFrame(train_wsi_embs_array, index=train_wsi_embs.index), "train")
    # plot_embedding_distributions(test_validation_embs, pd.DataFrame(test_validation_wsi_embs_array, index=test_validation_embs.index))
    # plot_embedding_distributions(train_embs_normalized, train_wsi_embs_normalized)

    set_trace()
# load survival data
print("loading survival data")
# mapping_df = pd.read_json(os.path.join(input_dir, 'mapping_df_29Jan.json')).T
mapping_df = pd.read_json(os.path.join(input_dir, 'mapping_df_30Jan_1000tiles.json')).T
# convert the 'event_occurred' values to 0 and 1 for compatibility with Sksurv
# mapping_df['event_occurred'] = mapping_df['event_occurred'].map({'Dead': 1, 'Alive': 0}).astype(int)
mapping_df['event_occurred'] = mapping_df['event_occurred'].map({'Dead': 1, 'Alive': 0}).astype(bool)

# prepare train and test survival data
train_survival = mapping_df.loc[common_train_indices]
test_validation_survival = mapping_df.loc[common_test_validation_indices]

if drop_outliers:
    # Inspect the top survival times
    print(mapping_df["time"].describe())
    filtered_mapping_df = mapping_df[mapping_df["time"] <= 4000]
    print(f"Removed {len(mapping_df) - len(filtered_mapping_df)} samples with survival time > 4000 days")
    # Ensure train and test indices only include available samples
    train_survival = filtered_mapping_df.reindex(train_embs.index)
    test_validation_survival = filtered_mapping_df.reindex(test_validation_embs.index)

    # Drop only rows where 'time' is NaN (caused by outlier removal)
    train_survival = train_survival.dropna(subset=["time"])
    test_validation_survival = test_validation_survival.dropna(subset=["time"])

    # Print new dataset sizes
    print(f"New Train Survival Size: {len(train_survival)}")
    print(f"New Test Survival Size: {len(test_validation_survival)}") 

    # drop the same samples from X_train and X_test_validation
    X_train = X_train.reindex(train_survival.index).dropna()
    X_test_validation = X_test_validation.reindex(test_validation_survival.index).dropna()

train_survival["event_occurred"] = train_survival["event_occurred"].astype(bool)
test_validation_survival["event_occurred"] = test_validation_survival["event_occurred"].astype(bool)

train_surv = Surv.from_arrays(train_survival["event_occurred"].values, train_survival["time"].values)
test_validation_surv = Surv.from_arrays(test_validation_survival["event_occurred"].values, test_validation_survival["time"].values)

if plot_survival:
    plt.figure(figsize=(8, 6))
    
    # Kaplan-Meier Fitter for train data
    kmf_train = KaplanMeierFitter()
    kmf_train.fit(train_survival["time"], train_survival["event_occurred"], label="Train Survival")
    kmf_train.plot_survival_function(color='blue')

    # Kaplan-Meier Fitter for test data
    kmf_test = KaplanMeierFitter()
    kmf_test.fit(test_validation_survival["time"], test_validation_survival["event_occurred"], label="Test Survival")
    kmf_test.plot_survival_function(color='red')

    # Add plot labels and styling
    plt.title("Survival Analysis: Train vs. Test")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig("survival_analysis.png", dpi=300, bbox_inches="tight")
    
    # Show the plot
    plt.show()

if check_PH_assumptions:
    def check_ph_assumption(data, dataset_name):
        """Check proportional hazards assumption using median survival time for given dataset."""
        print(f"\nChecking Proportional Hazards Assumption for {dataset_name}...\n")
        # checking KM plots

        median_time = np.median(data['time'])
        high_risk = data['time'] <= median_time
        low_risk = data['time'] > median_time

        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        kmf_high.fit(data["time"][high_risk], data["event_occurred"][high_risk], label="High risk")
        kmf_low.fit(data["time"][low_risk], data["event_occurred"][low_risk], label="Low risk")

        plt.figure()
        # kmf_high.plot()
        # kmf_low.plot()
        kmf_high.plot_survival_function(color='blue')
        kmf_low.plot_survival_function(color='red')
        plt.title(f"{dataset_name} stratified by risk estimates")
        plt.xlim(left=0)
        plt.ylim(top=1)
        plt.xlabel("Time (days)")
        plt.ylabel("Survival Probability")
        plt.grid(True)
        # plt.savefig(f"{mode}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    check_ph_assumption(train_survival, "Training Data")
    check_ph_assumption(test_validation_survival, "Test Data")
    

# set_trace()
# HPO with optuna
print(f" ********  Training for mode: {mode}  **********")
if do_hpo:
    def objective(trial):
        # # n_estimators = trial.suggest_int('n_estimators', 10, 100)
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        max_depth = trial.suggest_int('max_depth', 1, 3)    # 1,5

        # n_estimators = trial.suggest_int('n_estimators', 10, 500)
        # learning_rate = trial.suggest_float('learning_rate', 0.005, 0.5, log=True)
        # max_depth = trial.suggest_int('max_depth', 1, 10) 

        model = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators,
                                                 learning_rate=learning_rate,
                                                 max_depth=max_depth,
                                                 max_features='sqrt', # remove it?
                                                 random_state=seed_value)  
        print(f"X_train shape: {X_train.shape}, train_surv shape: {train_surv.shape}")
        kf = KFold(n_splits=5, shuffle=True, random_state=seed_value)  # 5-fold CV : 45
        c_indices = []

        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            survival_train_fold = train_surv[train_index]
            survival_val_fold = train_surv[val_index]

            model.fit(X_train_fold, survival_train_fold)
            risk_scores_val = model.predict(X_val_fold)

            # Compute concordance index on validation fold
            c_index_val = concordance_index_censored(
                survival_val_fold["event"], # name changed internally by sksurv
                survival_val_fold["time"],
                risk_scores_val
            )[0]
            c_indices.append(c_index_val)

            # # Optuna pruning (stops unpromising trials early based on the performance of the first few validation sets)
            # trial.report(np.mean(c_indices), fold)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()
            
            # # Aggressive Pruning
            # if fold >= 2 and np.mean(c_indices) < 0.55:
            #     raise optuna.exceptions.TrialPruned()
            
        return np.mean(c_indices)  # average CI across folds

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    model = GradientBoostingSurvivalAnalysis(**best_params)
else:
    print("training using model configuration determined from HPO")
    # model = GradientBoostingSurvivalAnalysis(n_estimators=31, learning_rate=0.85, max_depth=1, random_state=0)
    # model = GradientBoostingSurvivalAnalysis(n_estimators=20, learning_rate=0.05, max_depth=3, random_state=0)
    if mode == 'rnaseq_wsi':
        model = GradientBoostingSurvivalAnalysis(n_estimators=70, 
                                                learning_rate=0.01, 
                                                max_depth=2, 
                                                max_features='sqrt',
                                                random_state=seed_value)
    elif mode == 'only_rnaseq':
        model = GradientBoostingSurvivalAnalysis(n_estimators=39, 
                                        learning_rate=0.0542, 
                                        max_depth=2, 
                                        max_features='sqrt',
                                        random_state=seed_value)
    elif mode == 'only_wsi':
        model = GradientBoostingSurvivalAnalysis(n_estimators=10, 
                                        learning_rate=0.025, 
                                        max_depth=3, 
                                        max_features='sqrt',
                                        random_state=seed_value)        




# train the model
model.fit(X_train, train_surv)

# predict risk scores
risk_scores_train = model.predict(X_train)
risk_scores_test = model.predict(X_test_validation)

# compute concordance indices
c_index_train = concordance_index_censored(train_survival["event_occurred"], train_survival["time"], risk_scores_train)[0]
c_index_test = concordance_index_censored(test_validation_survival["event_occurred"], test_validation_survival["time"], risk_scores_test)[0]

print(f"Train Concordance Index: {c_index_train:.4f}")
print(f"Test Concordance Index: {c_index_test:.4f}")

# Kaplan-Meier Analysis
median_risk = np.median(risk_scores_test)
high_risk = risk_scores_test >= median_risk
low_risk = risk_scores_test < median_risk

kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()

kmf_high.fit(test_validation_survival["time"][high_risk], test_validation_survival["event_occurred"][high_risk], label="High Risk")
kmf_low.fit(test_validation_survival["time"][low_risk], test_validation_survival["event_occurred"][low_risk], label="Low Risk")

# compute log-rank p-value
logrank_p_value = logrank_test(test_validation_survival["time"][high_risk], 
                               test_validation_survival["time"][low_risk],
                               test_validation_survival["event_occurred"][high_risk], 
                               test_validation_survival["event_occurred"][low_risk]).p_value

print(f"log-rank test p-value: {logrank_p_value:.4f}")

plt.figure()
# kmf_high.plot()
# kmf_low.plot()
kmf_high.plot_survival_function(color='blue')
kmf_low.plot_survival_function(color='red')
plt.title(f"CI: {c_index_test:.3f},\nlog-rank test p-value: {logrank_p_value:.3f}")
plt.xlim(left=0)
plt.ylim(top=1, bottom=0)
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.savefig(f"{mode}_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()


if do_bootstrap:
    n_bootstraps = 100
    rng = np.random.RandomState(seed=seed_value)
    c_indices = []

    # store vars for KM curves
    time_grid = np.linspace(0, np.max(test_validation_survival["time"]), 10000)
    km_curves_high_risk = []
    km_curves_low_risk = []

    test_validation_survival["time"] = test_validation_survival["time"].astype(float)
    test_validation_survival["event_occurred"] = test_validation_survival["event_occurred"].astype(bool)

    # bootstrapping
    for _ in range(n_bootstraps):
        # Resample the test set with replacement
        X_resampled, y_resampled = resample(X_test_validation, test_validation_survival, random_state=rng)
        
        # Predict risk scores for the resampled test set
        risk_scores = model.predict(X_resampled)
        
        # Calculate the concordance index
        c_index = concordance_index_censored(y_resampled["event_occurred"], y_resampled["time"], risk_scores)[0]
        c_indices.append(c_index)

        # Kaplan-Meier fit for resampled dataset
        median_risk = np.median(risk_scores)
        high_risk = risk_scores >= median_risk
        low_risk = risk_scores < median_risk

        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        kmf_high.fit(y_resampled["time"][high_risk], y_resampled["event_occurred"][high_risk])
        kmf_low.fit(y_resampled["time"][low_risk], y_resampled["event_occurred"][low_risk])

        # Store survival probabilities at time grid
        km_curves_high_risk.append(kmf_high.predict(time_grid))
        km_curves_low_risk.append(kmf_low.predict(time_grid))

    # Convert to numpy array for convenience
    c_indices = np.array(c_indices)
    km_curves_high_risk = np.array(km_curves_high_risk)
    km_curves_low_risk = np.array(km_curves_low_risk)

    # Compute median and confidence intervals for KM plots
    km_median_high = np.percentile(km_curves_high_risk, 50, axis=0)
    km_lower_high = np.percentile(km_curves_high_risk, 2.5, axis=0)
    km_upper_high = np.percentile(km_curves_high_risk, 97.5, axis=0)

    km_median_low = np.percentile(km_curves_low_risk, 50, axis=0)
    km_lower_low = np.percentile(km_curves_low_risk, 2.5, axis=0)
    km_upper_low = np.percentile(km_curves_low_risk, 97.5, axis=0)

    # Calculate statistics
    mean_c_index = np.mean(c_indices)
    std_c_index = np.std(c_indices)
    ci_lower = np.percentile(c_indices, 2.5)
    ci_upper = np.percentile(c_indices, 97.5)

    print(f"Mean C-index: {mean_c_index:.3f}")
    print(f"Standard Deviation of C-index: {std_c_index:.3f}")
    print(f"95% Confidence Interval: ({ci_lower:.3f}, {ci_upper:.3f})")

    # plot c indices across bootstrapped samples
    plt.figure(figsize=(10, 6))
    plt.boxplot(c_indices, vert=True)
    plt.title('Bootstrap Distribution of C-index')
    plt.xlabel('C-index')
    plt.ylim(0.4, 0.8)
    plt.savefig(f"bootstrapped_cindex_{mode}_{timestamp}.png", dpi=300, bbox_inches='tight')
    # plt.show()

    # Extract high-risk and low-risk groups from the bootstrapped KM plot
    high_risk_times = test_validation_survival["time"][risk_scores >= np.median(risk_scores)]
    high_risk_events = test_validation_survival["event_occurred"][risk_scores >= np.median(risk_scores)]
    low_risk_times = test_validation_survival["time"][risk_scores < np.median(risk_scores)]
    low_risk_events = test_validation_survival["event_occurred"][risk_scores < np.median(risk_scores)]

    # Compute the log-rank test p-value
    logrank_result = logrank_test(high_risk_times, low_risk_times, high_risk_events, low_risk_events)
    logrank_p_value = logrank_result.p_value

    # Plot bootstrapped KM curves with the final log-rank test p-value

    # Plot bootstrapped KM curves with uncertainty regions
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
    plt.savefig(f"bootstrapped_km_{mode}_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print the final log-rank test result
    print(f"Final Log-rank test p-value: {logrank_p_value:.3f}")


set_trace()

