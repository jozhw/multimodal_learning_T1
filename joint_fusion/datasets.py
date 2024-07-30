import os

import numpy as np
import pandas as pd
from pdb import set_trace
from PIL import Image
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets, transforms


class CustomDataset(Dataset):
    def __init__(self, opt, mapping_df, split=None, mode='wsi', train_val_test = "train"):
        print("------------------- mapping_df.columns ----------------", mapping_df.columns)
        self.opt = opt
        self.train_val_test = train_val_test
        self.mapping_df = mapping_df
        # transformations/augmentations for WSI data
        if self.train_val_test == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                # transforms.RandomCrop(opt.input_size_wsi),
                # transforms.RandomCrop(256), # cropping or resizing not required as the tiles are already 256 x 256
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631], std=[0.21716536, 0.26081574, 0.20723464]) # from lunit (https://github.com/lunit-io/benchmark-ssl-pathology/releases)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631], std=[0.21716536, 0.26081574, 0.20723464]),
            ])

        # log transform for rnaseq data
        self.mapping_df['rnaseq_data'] = self.mapping_df['rnaseq_data'].apply(
            lambda x: np.log1p(np.array(list(x.values()))))

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = self.mapping_df.iloc[index]
        tcga_id = self.mapping_df.index[index]
        days_to_death = sample['days_to_death']
        days_to_last_followup = sample['days_to_last_followup']
        days_to_event = sample['time']
        event_occurred = 1 if sample['event_occurred'] == 'Dead' else 0
        tiles = sample['tiles']
        # convert these tile paths to images
        # x_wsi = [Image.open(self.opt.input_wsi_path + tile).convert('RGB') for tile in tiles] #.convert('RGB')
        x_wsi = [self.transforms(Image.open(self.opt.input_wsi_path + tile)) for tile in tiles]
        rnaseq_data = sample['rnaseq_data']
        # x_omic = torch.tensor(list(rnaseq_data.values()))
        x_omic = torch.tensor(rnaseq_data, dtype=torch.float32)
        if self.transforms:
            pass

        # return tcga_id, days_to_death, days_to_last_followup, event_occurred, x_wsi, x_omic

        return tcga_id, days_to_event, event_occurred, x_wsi, x_omic

    def __len__(self):
        return len(self.mapping_df)
