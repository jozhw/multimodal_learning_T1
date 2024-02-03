# load data from individual modalities
# pre-process/transform them individually
# create dataloaders

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

class MultimodalDataset(Dataset):
    def __init__(self, image_paths, gene_data, labels, transform=None):
        """
        image_paths: List of paths to image files
        gene_data: A numpy array or tensor of gene expression data (samples x genes)
        labels: A list or array of labels
        transform: torchvision transforms for preprocessing images
        """
        self.image_paths = image_paths
        self.gene_data = torch.tensor(gene_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def _preprocess_hist_images(self):
        self.data = preprocess_images(self.image_paths)

    def _preprocess_omic(self):
        self.data = preprocess_omic(self.omic_paths)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        gene_expression = self.gene_data[idx]
        label = self.labels[idx]
        return image, gene_expression, label

def preprocess_hist_images():

    # remove background

    # extract patches


    pass




def preprocess_omic():

    # normalization/transformation

    # batch correction

    pass

