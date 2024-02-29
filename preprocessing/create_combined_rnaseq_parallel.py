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

#metadata_rnaseq['full_path'] = os.path.join(base_dir, metadata_rnaseq['file_id'], metadata_rnaseq['file_name'])
metadata_rnaseq['full_path'] = base_dir + '/' + metadata_rnaseq['file_id'] + '/' + metadata_rnaseq['file_name']

# extract 'entity_submitter_id' from the 'associated_entities' column
metadata_rnaseq['entity_submitter_id'] = metadata_rnaseq['associated_entities'].apply(lambda x: x[0]['entity_submitter_id'] if isinstance(x, list) and len(x) > 0 else None)
metadata_rnaseq['entity_submitter_id'] = metadata_rnaseq['entity_submitter_id'].str.split('-').apply(lambda x: '-'.join(x[:5]))

set_trace()
print("completed metadata_rnaseq['entity_submitter_id']")

def read_process_file(full_path):
    df = pd.read_csv(full_path, sep='\t', header=1, skiprows=[2,3,4,5])
    return df[['tpm_unstranded']] if 'tpm_unstranded' in df else pd.DataFrame()

file_paths = metadata_rnaseq['full_path'].tolist()
set_trace()
# file_paths = metadata_rnaseq['full_path'].iloc[:10].tolist()

with ProcessPoolExecutor() as executor:
    df_list = list(executor.map(read_process_file, file_paths))

concatenated_rnaseq_df = pd.concat(df_list, axis=1)
output_file = 'combined_rnaseq_TCGA-LUAD.tsv'
concatenated_rnaseq_df.to_csv(output_file, sep='\t', index=False)

#for data in metadata['

set_trace()

tree = ET.parse('/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_clinical.tar/00e01b05-d939-49e6-b808-e1bab0d6773b/nationwidechildrens.org_clinical.TCGA-J2-8192.xml')
root = tree.getroot()

data = []

for child in root:
    row = {}
    for subchild in child:
        row[subchild.tag] = subchild.text
    data.append(row)

df = pd.DataFrame(data)

# df.set_index('id', inplace=True)

print(df)
set_trace()

