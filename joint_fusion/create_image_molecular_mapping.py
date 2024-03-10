import csv
import os
import pandas as pd
import ast
from pdb import set_trace

# code to create a csv file containing the TCGA ID, the corresponding TCGA WSI image file names, and clinical data (dead/alive, time to death/time to last followup)

base_dir = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/'
# path to the directory containing the tiles
tiles_dir = base_dir + 'data_3/tiles/256px_9.9x/'

# path to the csv file containing the clinical data
# input_csv_path = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/combined_clinical_TCGA-LUAD.csv'
data_clinical_df = pd.read_csv(base_dir + 'combined_clinical_TCGA-LUAD.csv', delimiter='\t')
data_rnaseq_df = pd.read_csv(base_dir + 'combined_rnaseq_TCGA-LUAD.csv', delimiter='\t')

# csv file with the mapped data
output_csv_path = base_dir + './mapped_data.csv'

png_files_dict = {}

for tile_dir in os.listdir(tiles_dir):
    for filename in os.listdir(tiles_dir + tile_dir):
        if filename.endswith('.png'):
            tcga_id = filename.split('-')[0] + '-' + filename.split('-')[1] + '-' + filename.split('-')[2] #+ '-' + filename.split('-')[3]
            png_file_name = filename #.rsplit('.', 1)[0]
            # Note: there may be multiple WSIs for a single patient, so that needs to be accounted for
            if tcga_id in png_files_dict:
                png_files_dict[tcga_id].append(png_file_name)
            else:
                png_files_dict[tcga_id] = [png_file_name]

# set_trace()
clinical_data_dict = {}

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


# Add a column from png_files_dict to the above df
# Creating a DataFrame with a single column to hold the lists from png_files_dict
combined_df = data_clinical_df.copy()

# Initialize a new column in combined_df for the file lists
combined_df['tiles'] = [[] for _ in range(len(combined_df))]

# Populate the 'files' column with data from png_files_dict where available
for idx in combined_df.index:
    combined_df.at[idx, 'tiles'] = png_files_dict.get(idx, [])


set_trace()

# add another columns (that is essentially a dictionary) for the rnaseq data
print("Do gene_name have repeats: ", data_rnaseq_df['gene_name'].duplicated().any())
print("Do gene_id have repeats: ", data_rnaseq_df['gene_id'].duplicated().any())

data_rnaseq_df = data_rnaseq_df.set_index('gene_id')

rnaseq_transposed = data_rnaseq_df.drop(columns=['gene_name', 'gene_type']).T
combined_df['rnaseq_data'] = [{} for _ in range(len(combined_df))]
for tcga_id in combined_df.index:
    if tcga_id in rnaseq_transposed.index:
        # create a dictionary for each row with gene_id: expression_value
        combined_df.at[tcga_id, 'rnaseq_data'] = rnaseq_transposed.loc[tcga_id].to_dict()

combined_df.to_csv(output_csv_path)
print(f"filtered data has been written to {output_csv_path}.")

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
