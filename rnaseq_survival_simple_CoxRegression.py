# Code to carry out simple Cox regression analysis using the RNASeq data as the covariates and the survival time as the target
import pandas as pd
import numpy as np
from collections import Counter
import ast
from lifelines import CoxPHFitter
from sklearn.feature_selection import VarianceThreshold
from pdb import set_trace

base_dir = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/'
data_rnaseq_all_genes_df = pd.read_csv(base_dir + 'combined_rnaseq_TCGA-LUAD.tsv', delimiter='\t')
data_clinical_df = pd.read_csv(base_dir + 'combined_clinical_TCGA-LUAD.tsv', delimiter='\t')


############################   CLEAN THE RNASEQ DATASET #####################################
# keep only the protein coding genes
data_rnaseq_df = data_rnaseq_all_genes_df[data_rnaseq_all_genes_df['gene_type'] == 'protein_coding']
# for some of the samples, there may be more than one rnaseq dataset (those with -01A- or -01B-are from the primary tumor while those with -11A- are from tissue adjacent to the tumor (so, normal?)
# https://docs.gdc.cancer.gov/Encyclopedia/pages/images/TCGA-TCGAbarcode-080518-1750-4378.pdf
# in such cases, we will keep only one tumor sample (-01A-)
# Remove columns that have -11A- within their names
filtered_columns = ['gene_id', 'gene_name', 'gene_type'] + [col for col in data_rnaseq_df.columns if '-01A-' in col]
data_rnaseq_df = data_rnaseq_df[filtered_columns]

# rename the columns by keeping only the minimal part of the TCGA ID, e.g.,  TCGA-44-2655
column_mapping = {
    col: '-'.join(col.split('-')[:3]) if 'TCGA' in col else col
    for col in data_rnaseq_df.columns
}
data_rnaseq_df = data_rnaseq_df.rename(columns=column_mapping)
data_rnaseq_df = data_rnaseq_df.loc[:, ~data_rnaseq_df.columns.duplicated()]
# set_trace()
column_counts = Counter(data_rnaseq_df.columns)
non_unique_columns = [col for col, count in column_counts.items() if count > 1]
print("Non-unique column names:", non_unique_columns)


###################################### EXTRACT RELEVANT DATA FROM THE CLINICAL DF ##########################

# find the common TCGA sample names
# common_columns = data_clinical_df.columns.intersection(data_rnaseq_df.columns)
# Keep only the common columns in both DataFrames
# The below also ensures both dfs have the same sample ordering
# data_clinical_df = data_clinical_df[common_columns]
# data_rnaseq_df = data_rnaseq_df[common_columns]
# set_trace()
# extract 'days_to_death' and 'vital_status' into lists for each column
days_to_death_list = []
vital_status_list = []
tcga_sample_list = []
for col in data_clinical_df.columns:
    print(data_clinical_df[col])
    # print(col)
    val = data_clinical_df[col].values[0]
    val = val.replace('nan', 'None')
    try:
        val_list = ast.literal_eval(val)
    except ValueError as e:
        print(f"Error processing column {col}: {e}")
        continue
    print(val_list)
    # set_trace()
    days_to_death = val_list[0]
    vital_status = val_list[1]
    days_to_death_list.append(days_to_death)
    vital_status_list.append(vital_status)
    tcga_sample_list.append(col)

####################################### PROCESS DATSETS FOR COX REGRESSION ANALYSIS ####################
data_rnaseq_df = data_rnaseq_df.set_index('gene_name').T
# drop non-gene expression columns
data_rnaseq_df = data_rnaseq_df.drop(['gene_id', 'gene_type'], errors='ignore')

data_clinical_df = pd.DataFrame({
    'sample_id': tcga_sample_list,
    'time_to_event': days_to_death_list,
    'event_occurred': vital_status_list
}).set_index('sample_id')


# find the common TCGA sample names
common_rows = data_clinical_df.index.intersection(data_rnaseq_df.index)
# Keep only the common rows in both DataFrames
# The below also ensures both dfs have the same sample ordering
data_clinical_df = data_clinical_df.loc[common_rows]
data_rnaseq_df = data_rnaseq_df.loc[common_rows]

# set_trace()
############################################# SET UP COX REGRESSION #################
combined_df = pd.concat([data_rnaseq_df, data_clinical_df], axis=1)
combined_df['event_occurred'] = combined_df['event_occurred'].apply(lambda x: 1 if x == 'Dead' else 0)

# remove low variance columns
X = combined_df.iloc[:, :-2]
selector = VarianceThreshold(threshold=0.01)
X_high_variance = selector.fit_transform(X)
X_high_variance_df = pd.DataFrame(X_high_variance, columns=X.columns[selector.get_support()])
combined_df = pd.concat([X_high_variance_df, combined_df.iloc[:, -2:]], axis=1)

print(combined_df)

model = CoxPHFitter()
model.fit(combined_df, duration_col='time_to_event', event_col='event_occurred')
print(model.summary)

set_trace()


