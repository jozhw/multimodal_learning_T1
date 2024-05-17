import pandas as pd
import matplotlib.pyplot as plt
from pdb import set_trace
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_csv('./combined_rnaseq_TCGA-LUAD.tsv', delimiter='\t')
data_bc = pd.read_csv('./batchcorrected_combined_rnaseq_TCGA-LUAD.tsv', delimiter='\t')
print("are all columns unique: ", data.columns.is_unique)
# set_trace()
# keep only the protein coding genes
data = data[data['gene_type'] == 'protein_coding']
# data_bc only contains the protein coding genes

# extract the tissue source site codes
tissue_sites = data.columns.str.extract(r'TCGA-(\d+)-')[0].unique()
tissue_sites_bc = data_bc.columns.str.extract(r'TCGA.(\d+).')[0].unique()

# drop non-numeric columns for PCA, t-SNE, UMAP
numeric_data = data.drop(['gene_id', 'gene_name', 'gene_type'], axis=1).T
numeric_data_bc = data_bc.drop(['gene_id', 'gene_name', 'gene_type'], axis=1).T
# extract labels for coloring (extracting from the sample IDs)
labels = numeric_data.index.str.split('-').map(lambda x: x[1] + '-' + x[2])
labels_bc = numeric_data_bc.index.str.split('.').map(lambda x: x[1] + '.' + x[2])
print("Are all labels unique (i.e., all are from different patients): ", labels.is_unique)

# # PCA
# pca = PCA(n_components=2)
# pca_results = pca.fit_transform(numeric_data)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(numeric_data)
# tsne_results_bc = tsne.fit_transform(numeric_data_bc)
# UMAP
# umap_results = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(numeric_data)

def plot_results(components, title):
    plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(components[:, 0], components[:, 1], c=pd.factorize(labels)[0], cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter, ticks=range(len(set(labels))))
    scatter = plt.scatter(components[:, 0], components[:, 1])
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

# plot_results(pca_results, 'PCA of TCGA-LUAD Data')


# plot_results(tsne_results, 't-SNE of TCGA-LUAD Data')
# plot_results(tsne_results_bc, 't-SNE of batch-corrected TCGA-LUAD Data')

# plot_results(umap_results, 'UMAP of TCGA-LUAD Data')

# DSC calculation (https://bioinformatics.mdanderson.org/public-software/tcga-batch-effects/)
tissue_labels = numeric_data.index.str.split('-').str[1]
tissue_sites = np.unique(tissue_labels)
# standardizing the features
# numeric_data = StandardScaler().fit_transform(numeric_data)

# calculate the global (over tissue sites) mean for each gene
global_mean = numeric_data.mean(axis=0)

# total number of samples
total_n = numeric_data.shape[0]

# initialize dictionary to store DSC values
dsc_values = {}
# set_trace()
# loop over each unique tissue site
batch_count = 0
for site in tissue_sites:
    batch_count += 1
    print(f"{batch_count} of {len(tissue_sites)} batches")
    site_mask = tissue_labels == site
    # set_trace()
    site_data = numeric_data[site_mask, :]
    site_mean = np.mean(site_data, axis=0)

    # number of samples in site
    n_site = site_data.shape[0]

    # proportion of samples in this site
    pi_i = n_site / total_n

    # within-site scatter matrix (Sw)
    deviations_within = site_data - site_mean
    Sw = np.dot(deviations_within.T, deviations_within)

    # between-site scatter matrix (Sb)
    mean_deviation_between = site_mean - global_mean
    Sb = n_site * pi_i * np.dot(mean_deviation_between[:, None], mean_deviation_between[None, :])

    # calculate DSC
    Db = np.sqrt(np.trace(Sb))
    Dw = np.sqrt(np.trace(Sw))
    DSC = Db / Dw if Dw != 0 else 0  # Avoid division by zero
    dsc_values[site] = DSC
    print(f"Tissue site: {site}, DSC: {DSC}")
    set_trace()

# Output DSC values
dsc_df = pd.DataFrame(list(dsc_values.items()), columns=['Tissue_Site', 'DSC'])
dsc_df.to_csv('dsc_values.csv', index=False)
print("DSC values saved to 'dsc_values.csv'.")

set_trace()