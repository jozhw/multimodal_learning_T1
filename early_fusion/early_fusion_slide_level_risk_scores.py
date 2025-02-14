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
import optuna
import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pdb import set_trace


timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
do_hpo = False
# do_hpo = False
check_PH_assumptions = False
plot_embs = False
plot_survival = False
mode = 'rnaseq_wsi' # 'rnaseq_wsi', 'only_rnaseq', 'only_wsi'

# on laptop
input_dir = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/early_fusion_inputs/'
# id = "2025-02-09-22-42-55_fold_3_epoch_1700"
id = "2025-02-13-17-57-25_fold_3_epoch_3900"

# # on polaris
# input_dir = '/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/early_fusion_inputs/'
# # id = "2025-02-09-22-42-55_fold_3_epoch_1700"
# id = "2025-02-13-17-57-25_fold_3_epoch_3900"

# file paths for rnaseq embeddings (generated using generate_rnaseq_embeddings_kfoldCV.py)
train_file = os.path.join(input_dir, f"rnaseq_embeddings_train_checkpoint_{id}.json")
val_file = os.path.join(input_dir, f"rnaseq_embeddings_val_checkpoint_{id}.json")
test_file = os.path.join(input_dir, f"rnaseq_embeddings_test_checkpoint_{id}.json")

# load the rnaseq embeddings
train_embs = pd.read_json(train_file).T
val_embs = pd.read_json(val_file).T
test_embs = pd.read_json(test_file).T

print("Train embeddings shape:", train_embs.shape)
print("Validation embeddings shape:", val_embs.shape)
print("Test embeddings shape:", test_embs.shape)

# combine val and test sets into a single held out set
print("concatenating val and test sets to create a single held out dataset")
test_validation_embs = pd.concat([val_embs, test_embs], axis=0)  # concatenate along rows (samples)
test_validation_tcga_ids = list(val_embs.index) + list(test_embs.index)

# load embeddings from WSI and rnaseq
print("loading WSI tile level embeddings")
wsi_embs = pd.read_json(os.path.join(input_dir, 'WSI_embeddings_average_uni_29Jan.rounded.json')).T
# get the slide level embeddings by averaging the tile level embeddings
print("obtaining slide level embeddings by averaging the tile level embeddings")
wsi_embs["slide_level_embedding"] = wsi_embs.apply(lambda row: np.mean(np.stack(row.values), axis=0), axis=1)
wsi_embs = wsi_embs[["slide_level_embedding"]].rename(columns={"slide_level_embedding": "slide_embedding"})

# embeddings from the wsi and the rnaseq modality may have different min/max values
# need to normalize them to bring them to the same scale

###### normalize rnaseq embeddings based on training set
# use min/max across all the embeddings
# for fit_transform() input needs to be of shape n_samples x n_features
print("scaling rnaseq embeddings")
scaler_rna = MinMaxScaler(feature_range=(-1, 1))
train_embs_normalized = pd.DataFrame(scaler_rna.fit_transform(train_embs), index=train_embs.index)
test_validation_embs_normalized = pd.DataFrame(scaler_rna.transform(test_validation_embs),
                                               index=test_validation_embs.index)

# Normalize WSI embeddings using training data
print("scaling WSI embeddings")
scaler_wsi = MinMaxScaler(feature_range=(-1, 1))
train_wsi_embs = wsi_embs.loc[train_embs.index]  # Select only training samples
train_wsi_embs_normalized = pd.DataFrame(scaler_wsi.fit_transform(list(train_wsi_embs["slide_embedding"])), index=train_wsi_embs.index)
test_validation_wsi_embs_normalized = pd.DataFrame(scaler_wsi.transform(list(wsi_embs.loc[test_validation_embs.index]["slide_embedding"])),
                                                   index=test_validation_embs.index)

# concatenate normalized embeddings to obtain multimodal embeddings
print("concatenating embeddings")
if mode == 'rnaseq_wsi':
    X_train = pd.concat([train_embs_normalized, train_wsi_embs_normalized], axis=1)
    X_test_validation = pd.concat([test_validation_embs_normalized, test_validation_wsi_embs_normalized], axis=1)
elif mode == 'only_wsi':
    X_train = pd.concat([train_wsi_embs_normalized], axis=1)
    X_test_validation = pd.concat([test_validation_wsi_embs_normalized], axis=1)
if mode == 'only_rnaseq':
    X_train = pd.concat([train_embs_normalized], axis=1)
    X_test_validation = pd.concat([test_validation_embs_normalized], axis=1)

if plot_embs:
    rna_min = test_validation_embs_normalized.min(axis=1)
    rna_max = test_validation_embs_normalized.max(axis=1)
    rna_mean = test_validation_embs_normalized.mean(axis=1)
    rna_std = test_validation_embs_normalized.std(axis=1)

    # Calculate statistics for WSI embeddings
    wsi_min = test_validation_wsi_embs_normalized.min(axis=1)
    wsi_max = test_validation_wsi_embs_normalized.max(axis=1)
    wsi_mean = test_validation_wsi_embs_normalized.mean(axis=1)
    wsi_std = test_validation_wsi_embs_normalized.std(axis=1)

    # Plotting
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

    # Adding labels and legend
    plt.xlabel('TCGA Sample Index')
    plt.ylabel('Embedding Value')
    plt.title('Statistics of RNA-seq and WSI Embeddings Across TCGA Samples')
    plt.legend()
    plt.grid(True)
    plt.show()


# load survival data
print("loading survival data")
mapping_df = pd.read_json(os.path.join(input_dir, 'mapping_df_29Jan.json')).T
# convert the 'event_occurred' values to 0 and 1 for compatibility with Sksurv
# mapping_df['event_occurred'] = mapping_df['event_occurred'].map({'Dead': 1, 'Alive': 0}).astype(int)
mapping_df['event_occurred'] = mapping_df['event_occurred'].map({'Dead': 1, 'Alive': 0}).astype(bool)

# prepare train and test survival data
train_survival = mapping_df.loc[train_embs.index]
test_validation_survival = mapping_df.loc[test_validation_embs.index]

# # Inspect the top survival times
# print(mapping_df["time"].describe())
# filtered_mapping_df = mapping_df[mapping_df["time"] <= 4000]
# print(f"Removed {len(mapping_df) - len(filtered_mapping_df)} samples with survival time > 4000 days")
# # Ensure train and test indices only include available samples
# train_survival = filtered_mapping_df.reindex(train_embs.index)
# test_validation_survival = filtered_mapping_df.reindex(test_validation_embs.index)

# # Drop only rows where 'time' is NaN (caused by outlier removal)
# train_survival = train_survival.dropna(subset=["time"])
# test_validation_survival = test_validation_survival.dropna(subset=["time"])

# # Print new dataset sizes
# print(f"New Train Survival Size: {len(train_survival)}")
# print(f"New Test Survival Size: {len(test_validation_survival)}") 

# # drop the same samples from X_train and X_test_validation
# X_train = X_train.reindex(train_survival.index).dropna()
# X_test_validation = X_test_validation.reindex(test_validation_survival.index).dropna()

# train_survival["event_occurred"] = train_survival["event_occurred"].astype(bool)
# test_validation_survival["event_occurred"] = test_validation_survival["event_occurred"].astype(bool)

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
if do_hpo:
    def objective(trial):
        # n_estimators = trial.suggest_int('n_estimators', 10, 100)
        n_estimators = trial.suggest_int('n_estimators', 1, 100)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 1.0)
        max_depth = trial.suggest_int('max_depth', 1, 5)    # 1,5

        model = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators,
                                                 learning_rate=learning_rate,
                                                 max_depth=max_depth,
                                                 random_state=0)
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 5-fold CV
        c_indices = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            survival_train_fold = train_surv[train_index]

            model.fit(X_train_fold, survival_train_fold)
            risk_scores_val = model.predict(X_val_fold)

            # Compute concordance index on validation fold
            c_index_val = concordance_index_censored(
                train_survival.iloc[val_index]["event_occurred"],
                train_survival.iloc[val_index]["time"],
                risk_scores_val
            )[0]
            c_indices.append(c_index_val)

        return np.mean(c_indices)  # Average CI across folds

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    model = GradientBoostingSurvivalAnalysis(**best_params)
else:
    print("training using model configuration determined from HPO")
    # model = GradientBoostingSurvivalAnalysis(n_estimators=31, learning_rate=0.85, max_depth=1, random_state=0)
    model = GradientBoostingSurvivalAnalysis(n_estimators=84, learning_rate=0.99, max_depth=4, random_state=0)

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
logrank_p_value = logrank_test(test_validation_survival["time"][high_risk], test_validation_survival["time"][low_risk],
                               test_validation_survival["event_occurred"][high_risk], test_validation_survival["event_occurred"][low_risk]).p_value

print(f"log-rank test p-value: {logrank_p_value:.4f}")

plt.figure()
# kmf_high.plot()
# kmf_low.plot()
kmf_high.plot_survival_function(color='blue')
kmf_low.plot_survival_function(color='red')
plt.title(f"CI: {c_index_test:.3f},\nlog-rank test p-value: {logrank_p_value:.3f}")
plt.xlim(left=0)
plt.ylim(top=1)
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.savefig(f"{mode}_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

set_trace()

