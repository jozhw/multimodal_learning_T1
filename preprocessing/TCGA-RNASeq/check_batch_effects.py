import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from pdb import set_trace

directory = '/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_WSI_RNASeq_clinical/RNASeq/'
col_names = ['gene_id', 'gene_name', 'gene_type', 'unstranded', 'stranded_first', 'stranded_second', 'tpm_unstranded',
             'fpkm_unstranded', 'fpkm_uq_unstranded']
# metadata_file = directory + 'metadata.cart.2024-02-21.json'
metadata_file = '/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_WSI_RNASeq_clinical/RNASeq/metadata.cart.2024-02-21.json'

print("metadata file: ", metadata_file)

# create mapping between RNASeq file names and the TCGA IDs
with open(metadata_file, 'r') as file:
    json_data = file.read()

data = json.loads(json_data)

rnaseq_tcgaid_map = {}

for item in data:
    file_name = item['file_name']
    print("file_name: ", file_name)
    entity_submitter_id = item['annotations'][0]['entity_submitter_id']
    rnaseq_tcgaid_map[file_name] = entity_submitter_id


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, sep='\t', names=col_names, skiprows=6)
    data = data[data['gene_type'] == 'protein_coding']  # keep only the protein coding genes
    data.set_index('gene_name', inplace=True)  # rename the index with the gene names
    return data['tpm_unstranded']


patient_files = [file for file in os.listdir(directory) if file.endswith('.tsv')]
patient_ids = [value for key, value in rnaseq_tcgaid_map.items() for key in patient_files]


# load and preprocess data for each patient
all_data = []
for file in patient_files:
    # file = directory + file
    patient_data = load_and_preprocess_data(directory + file)
    all_data.append(patient_data)
    # print(all_data)
    # set_trace()

# combine data into a single df
combined_data = pd.concat(all_data, axis=1)
new_column_names = [f"{col}_{i+1}" for i, col in enumerate(combined_data.columns)]
combined_data.columns = new_column_names
# combined_data.columns = patient_ids

# transpose the df so that genes are features and rows are samples (patients)
X = combined_data.T

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', marker='o', edgecolor='k', s=50)
for i, pseudo_tcga_id in enumerate(new_column_names):
    plt.text(X_pca[i, 0], X_pca[i, 1], pseudo_tcga_id)
plt.title('PCA of Patients')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.grid(True)
plt.show()


set_trace()
# patient_ids =
