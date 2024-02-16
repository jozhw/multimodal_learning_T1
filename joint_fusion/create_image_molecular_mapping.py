import csv
import os
from pdb import set_trace

# code to create a csv file containing the TCGA WSI image file names, the corresponding TCGA IDs, censored and survival time data

# path to the directory containing the png files
png_files_directory = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/joint_fusion/data/TCGA_GBMLGG/all_st'

# path to the csv file containing the the patient TCGA ID, censored, survival time, and other molecular feature data
input_csv_path = '/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/joint_fusion/data/TCGA_GBMLGG/all_dataset.csv'

# csv file with the mapped data
output_csv_path = './mapped_data.csv'

png_files = {}

for filename in os.listdir(png_files_directory):
    if filename.endswith('.png'):
        tcga_id = filename.split('-')[0] + '-' + filename.split('-')[1] + '-' + filename.split('-')[2] #+ '-' + filename.split('-')[3]
        png_file_name = filename #.rsplit('.', 1)[0]
        # Note: there may be multiple WSIs for a single patient, so that needs to be accounted for
        if tcga_id in png_files:
            png_files[tcga_id].append(png_file_name)
        else:
            png_files[tcga_id] = [png_file_name]

# set_trace()
data = {}



with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        tcga_id = row['TCGA ID']
        if tcga_id not in data:
            data[tcga_id] = {
                'png_files': ' & '.join(png_files.get(tcga_id, [])), # multiple png files will be ampersand separated
                'censored': row['censored'],
                'survival_months': row['Survival months']
            }

# set_trace()

with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['TCGA_ID', 'png_files', 'censored', 'survival_months']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for tcga_id, item in data.items():
        writer.writerow({
            'TCGA_ID': tcga_id,
            'png_files': item['png_files'],
            'censored': item['censored'],
            'survival_months': item['survival_months']
        })

print(f"filtered data has been written to {output_csv_path}.")
