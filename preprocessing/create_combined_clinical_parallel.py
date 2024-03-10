# creates a combined file that contains clinical data from all samples

import pandas as pd
import xml.etree.ElementTree as ET
from pdb import set_trace
import os
from concurrent.futures import ProcessPoolExecutor

# read the metadata json file that contains the paths to the clinical and the rnaseq tsv files
base_dir = '/mnt/c/Users/tnandi/Downloads/TCGA-LUAD-RNASeq_clinical_all/gdc_download_20240224_072723.742270.tar/'
meta_file = '/mnt/c/Users/tnandi/Downloads/TCGA-LUAD-RNASeq_clinical_all/metadata.cart.2024-02-24.json'
metadata = pd.read_json(meta_file)

# Get only rows corresponding to BCR XML files with clinical data
metadata_clinical = metadata[metadata['data_format'] == 'BCR XML']
# create a new column that contains the full path for the files
metadata_clinical['full_path'] = base_dir + '/' + metadata_clinical['file_id'] + '/' + metadata_clinical['file_name']
# set_trace()
# extract 'entity_submitter_id' from the 'associated_entities' column
metadata_clinical['entity_submitter_id'] = metadata_clinical['associated_entities'].apply(
    lambda x: x[0]['entity_submitter_id'] if isinstance(x, list) and len(x) > 0 else None)

# metadata_clinical['entity_submitter_id'] = metadata_clinical['entity_submitter_id'].str.split('-').apply(lambda x: '-'.join(x[:5]))
repeats_exist = metadata_clinical['entity_submitter_id'].duplicated().any()
print("Do repeats exist? ", repeats_exist)

namespaces = {
    'luad': 'http://tcga.nci/bcr/xml/clinical/luad/2.7',
}
def read_process_file(full_path, entry_submitter_id):
    df_file = pd.read_xml(full_path, xpath='//luad:patient | //luad:admin', namespaces=namespaces)
    print(full_path)
    print(df_file['days_to_death'])
    print(df_file['days_to_last_followup'])
    print(df_file['vital_status'])
    # print(df_file['patient_withdrawal'])
    combined = df_file.apply(lambda row: [row['days_to_death'], row['days_to_last_followup'], row['vital_status']], axis=1)
    # print(df_file.columns)
    # return df_file[['days_to_death']].rename(columns={'days_to_death': entry_submitter_id})
    return pd.DataFrame({entry_submitter_id: combined})


file_paths = metadata_clinical['full_path'].tolist()
entry_submitter_ids = metadata_clinical['entity_submitter_id'].tolist()
# file_paths = metadata_clinical['full_path'].iloc[:10].tolist()  # to debug using 10 samples
# set_trace()

print("Creating combined df")
with ProcessPoolExecutor() as executor:
    df_list = list(executor.map(read_process_file, file_paths, entry_submitter_ids))

# set_trace()
concatenated_clinical_df = pd.concat(df_list, axis=1)
output_file = 'combined_clinical_TCGA-LUAD.tsv'
concatenated_clinical_df.to_csv(output_file, sep='\t', index=False)

# for data in metadata['

set_trace()


