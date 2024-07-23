import pandas as pd
from pdb import set_trace

clinical_data = pd.read_csv('/mnt/c/Users/tnandi/Downloads/spectra-tcga-luad/spectra-tcga-luad/TCGA-LUAD_RNAseq_top12_20240702_151006_sample_data.csv')
rnaseq_data = pd.read_csv('/mnt/c/Users/tnandi/Downloads/spectra-tcga-luad/spectra-tcga-luad/TCGA-LUAD_RNAseq_top12_20240702_151006_gene_counts.csv')

clinical_out_file = 'combined_clinical_TCGA-LUAD.csv'

clinical_data_subset_df = clinical_data[['patient', 'days_to_death', 'days_to_last_follow_up', 'vital_status']]
clinical_data_subset_without_duplicates_df = clinical_data_subset_df.drop_duplicates()
print(clinical_data_subset_without_duplicates_df)
# clinical_data_subset_without_duplicates_df = clinical_data_subset_df.drop_duplicates(subset=['patient'])
# get rows where all entries except patinet ID are NaN
missing_data_all_rows = clinical_data_subset_without_duplicates_df[clinical_data_subset_without_duplicates_df.drop(columns=['patient']).isna().all(axis=1)]
print("rows where all entries except patinet ID are NaN: ", missing_data_all_rows)
# get rows with both days_to_death and 'days_to_last_follow_up' as NaN
missing_days = clinical_data_subset_without_duplicates_df[
    clinical_data_subset_without_duplicates_df['days_to_death'].isna() & clinical_data_subset_without_duplicates_df['days_to_last_follow_up'].isna()
]
print("rows where both days_to_death and 'days_to_last_follow_up' are NaN: ", missing_days)

# Remove those rows from the DataFrame
filtered_df = clinical_data_subset_without_duplicates_df.drop(missing_days.index)

# reformat it to be compatible with the previous version and create_image_molecular_mapping.py
filtered_df = filtered_df.set_index('patient').apply(lambda row: [row['days_to_death'], row['days_to_last_follow_up'], row['vital_status']], axis=1).to_frame().T


# Write the resulting DataFrame to a CSV file
filtered_df.to_csv(clinical_out_file, index=False, na_rep='NaN')

print(f"Filtered DataFrame written to {clinical_out_file}")

set_trace()
# in the rnaseq df, keep only columns 'gene_id', 'gene_name', 'gene_type' and those with 'tpm' in them
columns_to_keep = ['gene_id', 'gene_name', 'gene_type']
columns_to_keep.extend([col for col in rnaseq_data.columns if 'tpm' in col])
filtered_rnaseq_data = rnaseq_data[columns_to_keep]

filtered_rnaseq_data_trimmed = filtered_rnaseq_data.iloc[:-4]
new_column_names = {col: col.replace('tpm_unstranded_', '') for col in filtered_rnaseq_data_trimmed.columns if 'tpm_unstranded_' in col}
filtered_rnaseq_data_trimmed.rename(columns=new_column_names, inplace=True)
filtered_rnaseq_data_trimmed.to_json('rnaseq_from_heidi.json', orient='records', lines=True)

# keep only the protein coding genes
filtered_rnaseq_data = filtered_rnaseq_data[filtered_rnaseq_data['gene_type'] == 'protein_coding']


set_trace()