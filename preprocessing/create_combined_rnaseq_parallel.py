# creates a combined file that contains rnaseq data from all samples

import pandas as pd
import xml.etree.ElementTree as ET
from pdb import set_trace
import os
from concurrent.futures import ProcessPoolExecutor

# read the metadata json file that contains the paths to the clinical and the rnaseq tsv files
base_dir = '/mnt/c/Users/tnandi/Downloads/TCGA-LUAD-RNASeq_clinical_all/gdc_download_20240224_072723.742270.tar/'
meta_file = '/mnt/c/Users/tnandi/Downloads/TCGA-LUAD-RNASeq_clinical_all/metadata.cart.2024-02-24.json'
metadata = pd.read_json(meta_file)

# Get only rows corresponding to TSV files with rnaseq data
metadata_rnaseq = metadata[metadata['data_format'] == 'TSV']

# create a new column that contains the full path for the files
metadata_rnaseq['full_path'] = base_dir + '/' + metadata_rnaseq['file_id'] + '/' + metadata_rnaseq['file_name']

# extract 'entity_submitter_id' from the 'associated_entities' column
metadata_rnaseq['entity_submitter_id'] = metadata_rnaseq['associated_entities'].apply(
    lambda x: x[0]['entity_submitter_id'] if isinstance(x, list) and len(x) > 0 else None)

# metadata_rnaseq['entity_submitter_id'] = metadata_rnaseq['entity_submitter_id'].str.split('-').apply(lambda x: '-'.join(x[:5]))
repeats_exist = metadata_rnaseq['entity_submitter_id'].duplicated().any()
print("Do repeats exist? ", repeats_exist)


def read_process_file(full_path, entry_submitter_id):
    df_file = pd.read_csv(full_path, sep='\t', header=1, skiprows=[2, 3, 4, 5])
    # print(df_file)
    # print(df_file['gene_name'][1234])
    # return df_file[['tpm_unstranded']] if 'tpm_unstranded' in df_file else pd.DataFrame()
    return df_file[['tpm_unstranded']].rename(columns={'tpm_unstranded': entry_submitter_id})


file_paths = metadata_rnaseq['full_path'].tolist()
entry_submitter_ids = metadata_rnaseq['entity_submitter_id'].tolist()
# file_paths = metadata_rnaseq['full_path'].iloc[:10].tolist()  # to debug using 10 samples
# set_trace()

# Get the gene IDs and gene names from a single rnaseq file
df_single = pd.read_csv(file_paths[0], sep='\t', header=1, skiprows=[2, 3, 4, 5])
gene_df = {
    'gene_id': df_single['gene_id'],
    'gene_name': df_single['gene_name'],
    'gene_type': df_single['gene_type']
}
gene_columns_df = pd.DataFrame(gene_df)

print("Creating combined df")
with ProcessPoolExecutor() as executor:
    df_list = list(executor.map(read_process_file, file_paths, entry_submitter_ids))

concatenated_rnaseq_df = pd.concat(df_list, axis=1)
concatenated_rnaseq_with_gene_ids_df = pd.concat([gene_columns_df, concatenated_rnaseq_df], axis=1)
output_file = 'combined_rnaseq_TCGA-LUAD.csv'
# set_trace()
# concatenated_rnaseq_with_gene_ids_df.to_csv(output_file, sep='\t', index=False)
# set_trace()

# keep only the protein coding genes
data_rnaseq_df = concatenated_rnaseq_with_gene_ids_df[concatenated_rnaseq_with_gene_ids_df['gene_type'] == 'protein_coding']
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
data_rnaseq_df.to_csv(output_file, sep='\t', index=False)
set_trace()

# tree = ET.parse(
#     '/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_clinical.tar/00e01b05-d939-49e6-b808-e1bab0d6773b/nationwidechildrens.org_clinical.TCGA-J2-8192.xml')
# root = tree.getroot()
#
# data = []
#
# for child in root:
#     row = {}
#     for subchild in child:
#         row[subchild.tag] = subchild.text
#     data.append(row)
#
# df = pd.DataFrame(data)
#
# # df.set_index('id', inplace=True)
#
# print(df)
# set_trace()
