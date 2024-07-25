import csv
import os
import pandas as pd
import ast
import random
import numpy as np
from pdb import set_trace

remove_housekeeping_genes = False  # whether to remove the housekeeping genes from features'
remove_lowly_expressed_genes = False  # whether to remove the lowly expressed genes from features'
num_tiles_per_wsi = 200

# code to create a csv file containing the TCGA ID, the corresponding TCGA WSI image file names, and clinical data (dead/alive, time to death/time to last followup)

# base_dir = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/' # for laptop
base_dir = '/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/'  # for Polaris

# tiles_dir = base_dir + 'TCGA_WSI/batch_corrected/processed_svs/tiles/256px_9.9x/' # laptop
# tiles_dir = base_dir + 'TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_otsu_B/tiles/256px_9.9x/' # tiles on Polaris (with otsu's threshold;not stain corrected for 449 samples) [mistakenly deleted this dir]
tiles_dir = base_dir + 'TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/256px_9.9x/'
# added tiles from 'TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/256px_9.9x_clean_from_penmarks/' to above # added the separately extracted tiles from the WSIs with penmark with the clean ones

# path to the csv file containing the clinical data
# input_csv_path = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/combined_clinical_TCGA-LUAD.csv'
# data_clinical_df = pd.read_csv(base_dir + 'combined_clinical_TCGA-LUAD.csv',
#                                delimiter='\t')  # contains [time_to_death, time_to_last_followup, dead/alive status] for all LUAD samples

data_clinical_df = pd.read_csv(base_dir + 'combined_clinical_TCGA-LUAD.csv') # from red_heidi_data.py
# data_rnaseq_df = pd.read_csv(base_dir + 'combined_rnaseq_TCGA-LUAD.csv', delimiter='\t') # contains TPM for all protein coding genes for all LUAD samples

data_rnaseq_df = pd.read_csv(base_dir + 'batchcorrected_combined_rnaseq_TCGA-LUAD.tsv',
                             delimiter='\t')  # ue the batch corrected rnaseq samples

# csv file with the mapped data
output_csv_path = base_dir + './mapped_data_23july.csv'
# json file with the mapped data
output_json_path = base_dir + './mapped_data_23july.json'

png_files_dict = {}
# set_trace()
# Get the list of the extracted tiles (in png/jpg format) for each sample
count_tcga_wsi = 0

# for tile_dir in os.listdir(tiles_dir):
#     if "TCGA" in tile_dir:  # skipping the "combined_tiles" directory
#         count_tcga_wsi += 1
#         for filename in os.listdir(tiles_dir + tile_dir):
#             if filename.endswith('.png') or filename.endswith('.jpg'):
#                 tcga_id = filename.split('-')[0] + '-' + filename.split('-')[1] + '-' + filename.split('-')[
#                     2]  # + '-' + filename.split('-')[3]
#                 png_file_name = filename  # .rsplit('.', 1)[0]
#                 # Note: there may be multiple WSIs for a single patient, so that needs to be accounted for
#                 if tcga_id in png_files_dict:
#                     png_files_dict[tcga_id].append(png_file_name)
#                 else:
#                     png_files_dict[tcga_id] = [png_file_name]

# use only num_tiles_per_wsi (currently set to 200) randomly chosen tiles from each WSI
for tile_dir in os.listdir(tiles_dir):
    if "TCGA" in tile_dir:  # skipping the "combined_tiles" directory
        count_tcga_wsi += 1
        count_tiles = 0
        for filename in os.listdir(os.path.join(tiles_dir, tile_dir)):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                count_tiles += 1
                tcga_id = '-'.join(filename.split('-')[:3])
                if tcga_id in png_files_dict:
                    png_files_dict[tcga_id].append(filename)
                else:
                    png_files_dict[tcga_id] = [filename]
        print("ID: ", tile_dir, " count_tiles: ", count_tiles)

# limit each key to have only 200 files in val
for tcga_id in png_files_dict:
    if len(png_files_dict[tcga_id]) > 200:
        png_files_dict[tcga_id] = random.sample(png_files_dict[tcga_id], 200)

# find keys with values that are not equal to 200
keys_with_different_length = {key: len(value) for key, value in png_files_dict.items() if len(value) != 200}

if keys_with_different_length:
    for key, length in keys_with_different_length.items():
        print(f"Key: {key}, Length of values: {length}")
else:
    print("All keys have 200 entries")

# check where 1 sample is missing (count_tcga_wsi = 449 whereas total number of WSIs = 450)
# set_trace()
# clinical_data_dict = {}

# extract 'days_to_death' , 'days_to_last_followup', and 'vital_status' into lists for each column
days_to_death_list = []
days_to_last_followup_list = []
vital_status_list = []
tcga_id_list = []
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
    days_to_last_followup = val_list[1]
    vital_status = val_list[2]
    days_to_death_list.append(days_to_death)
    days_to_last_followup_list.append(days_to_last_followup)
    vital_status_list.append(vital_status)
    tcga_id_list.append(col)

data_clinical_df = pd.DataFrame({
    'sample_id': tcga_id_list,
    'days_to_death': days_to_death_list,
    'days_to_last_followup': days_to_last_followup_list,
    'event_occurred': vital_status_list
}).set_index('sample_id')

combined_df = data_clinical_df.copy()

combined_df['tiles'] = [[] for _ in range(len(combined_df))]

for idx in combined_df.index:
    print(idx)
    combined_df.at[idx, 'tiles'] = png_files_dict.get(idx, [])

# add another columns (that is essentially a dictionary) for the rnaseq data
print("Do gene_name have repeats: ", data_rnaseq_df['gene_name'].duplicated().any())
print("Do gene_id have repeats: ", data_rnaseq_df['gene_id'].duplicated().any())

data_rnaseq_df = data_rnaseq_df.set_index('gene_id')
gene_id_name_dict = data_rnaseq_df['gene_name'].to_dict()
# set_trace()
# remove housekeeping genes
if remove_housekeeping_genes:
    # list of housekeeping genes obtained from https://www.tau.ac.il/~elieis/HKG/
    housekeeping_genes_df = pd.read_csv('../preprocessing/housekeeping_genes.txt', sep='\t', header=None,
                                        names=['gene_name', 'accession'])
    housekeeping_genes = housekeeping_genes_df['gene_name'].str.strip().tolist()  # 3804 genes
    housekeeping_genes_rows = data_rnaseq_df[data_rnaseq_df['gene_name'].isin(
        housekeeping_genes)]  # of the 3804 housekeeping genes, 3526 are protein-coding genes
    data_rnaseq_df = data_rnaseq_df[~data_rnaseq_df['gene_name'].isin(housekeeping_genes)]

rnaseq_transposed_df = data_rnaseq_df.drop(columns=['gene_name', 'gene_type']).T

# remove lowly expressed genes
if remove_lowly_expressed_genes:
    # define the TPM threshold
    tpm_threshold = 1
    lowly_expressed_genes = rnaseq_transposed_df.columns[(rnaseq_transposed_df <= tpm_threshold).all(axis=0)]
    lowly_expressed_genes_list = lowly_expressed_genes.tolist()
    lowly_expressed_gene_names = [gene_id_name_dict[gene_id] for gene_id in lowly_expressed_genes_list if
                                  gene_id in gene_id_name_dict]
    print("lowly expressed genes: ", lowly_expressed_gene_names)
    rnaseq_transposed_df = rnaseq_transposed_df.loc[:, (rnaseq_transposed_df > tpm_threshold).any(axis=0)]
    # 14800 genes remain for tpm_threshold = 1,

# set_trace()

# change the index of rnaseq_transposed_df to be consistent with those of combined_df (and remove repetitions; one TCGA sample may have > 1 rnaseq sample)
new_indices = ['TCGA-' + index.split('.')[1] + '-' + index.split('.')[2] for index in rnaseq_transposed_df.index]
rnaseq_transposed_df.index = new_indices
rnaseq_transposed_df = rnaseq_transposed_df[~rnaseq_transposed_df.index.duplicated(keep='first')]

combined_df['rnaseq_data'] = [{} for _ in range(len(combined_df))]
for tcga_id in combined_df.index:
    if tcga_id in rnaseq_transposed_df.index:
        # create a dictionary for each row with gene_id: expression_value
        combined_df.at[tcga_id, 'rnaseq_data'] = rnaseq_transposed_df.loc[tcga_id].to_dict()

# combined_df['rnaseq_data'] = None
# for tcga_id in combined_df.index:
#     if tcga_id in rnaseq_transposed.index:
#         combined_df.at[tcga_id, 'rnaseq_data'] = rnaseq_transposed.loc[tcga_id].to_dict()
#     else:
#         combined_df.at[tcga_id, 'rnaseq_data'] = {}

print("Does the rnaseq data have any nan: ",
      combined_df['rnaseq_data'].apply(lambda d: any(np.isnan(v) for v in d.values())).any())

# find indices without rnaseq data
indices_without_rnaseq_data = combined_df[combined_df['rnaseq_data'].apply(lambda d: not d)].index.tolist()
# remove the above indices from combined_df
combined_df = combined_df[combined_df['rnaseq_data'].apply(lambda d: bool(d))]

# print("Number of rows where WSI list is empty: ", combined_df['tiles'].apply(len).eq(0).sum())
print("Number of rows where WSI list is not empty: ", combined_df['tiles'].apply(len).ne(0).sum())
# # check for duplicated rows based on the 'tiles' column
# duplicated_tiles = combined_df.duplicated(subset=['tiles'])
#
# if duplicated_tiles.any():
#     print("There are repeating rows in the 'tiles' column.")
#     duplicated_rows = combined_df[combined_df.duplicated(subset=['tiles'], keep=False)]
#     print("Duplicated rows:")
#     print(duplicated_rows)
# else:
#     print("No repeating rows found in the 'tiles' column.")
# set_trace()

combined_df.to_csv(output_csv_path)
combined_df.to_json(output_json_path, orient='index')
print(f"filtered data has been written to {output_csv_path} and {output_json_path}.")

set_trace()

# with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         tcga_id = row['TCGA ID']
#         if tcga_id not in data:
#             data[tcga_id] = {
#                 'png_files': ' & '.join(png_files_dict.get(tcga_id, [])), # multiple png files will be ampersand separated
#                 # 'censored': row['censored'],
#                 'survival_months': row['Survival months']
#             }

# set_trace()

# with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
#     fieldnames = ['TCGA_ID', 'png_files', 'censored', 'survival_months']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for tcga_id, item in data.items():
#         writer.writerow({
#             'TCGA_ID': tcga_id,
#             'png_files': item['png_files'],
#             'censored': item['censored'],
#             'survival_months': item['survival_months']
#         })
#
# print(f"filtered data has been written to {output_csv_path}.")
