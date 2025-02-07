import os

import numpy as np
import pandas as pd
from pdb import set_trace
import time
from PIL import Image
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets, transforms
import cv2
from joblib import Parallel, delayed
import h5py
from multiprocessing import Manager
import ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDatasetOld(Dataset):
    def __init__(self, opt, mapping_df, split=None, mode='wsi', train_val_test="train"):
        # print("------------------- mapping_df.columns ----------------", mapping_df.columns)
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
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464])
                # from lunit (https://github.com/lunit-io/benchmark-ssl-pathology/releases)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464]),
            ])

        # log transform for rnaseq data
        self.mapping_df['rnaseq_data'] = self.mapping_df['rnaseq_data'].apply(
            lambda x: np.log1p(np.array(list(x.values()))))

    def __getitem__(self, index):
        start_time = time.time()
        if torch.is_tensor(index):
            index = index.tolist()
        sample = self.mapping_df.iloc[index]
        tcga_id = self.mapping_df.index[index]
        days_to_death = sample['days_to_death']
        days_to_last_followup = sample['days_to_last_followup']
        days_to_event = sample['time']
        event_occurred = 1 if sample['event_occurred'] == 'Dead' else 0
        tiles = sample['tiles']
        step1_time = time.time()
        # convert these tile paths to images
        # x_wsi = [Image.open(self.opt.input_wsi_path + tile).convert('RGB') for tile in tiles] #.convert('RGB')
        x_wsi = [self.transforms(Image.open(self.opt.input_wsi_path + tile)) for tile in tiles]
        step2_time = time.time()
        rnaseq_data = sample['rnaseq_data']
        # x_omic = torch.tensor(list(rnaseq_data.values()))
        x_omic = torch.tensor(rnaseq_data, dtype=torch.float32)
        # if self.transforms:
        #     pass
        step3_time = time.time()

        print(
            f"Index: {index}, Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s")

        # return tcga_id, days_to_death, days_to_last_followup, event_occurred, x_wsi, x_omic

        return tcga_id, days_to_event, event_occurred, x_wsi, x_omic

    def __len__(self):
        return len(self.mapping_df)


# using opencv
class CustomDatasetCV(Dataset):
    def __init__(self, opt, mapping_df, split=None, mode='wsi', train_val_test="train"):
        self.opt = opt
        self.train_val_test = train_val_test
        self.mapping_df = mapping_df

        # transformations/augmentations for WSI data
        if self.train_val_test == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464])  # from lunit
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464]),
            ])

        # log transform for rnaseq data
        self.mapping_df['rnaseq_data'] = self.mapping_df['rnaseq_data'].apply(
            lambda x: np.log1p(np.array(list(x.values()))))

    def __getitem__(self, index):
        start_time = time.time()

        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.mapping_df.iloc[index]
        tcga_id = self.mapping_df.index[index]
        days_to_death = sample['days_to_death']
        days_to_last_followup = sample['days_to_last_followup']
        days_to_event = sample['time']
        event_occurred = 1 if sample['event_occurred'] == 'Dead' else 0
        tiles = sample['tiles']

        step1_time = time.time()

        # Load images using OpenCV and apply transformations
        images = []
        for tile in tiles:
            image_path = os.path.join(self.opt.input_wsi_path, tile)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.transforms(image)
            images.append(image)

        step2_time = time.time()

        rnaseq_data = sample['rnaseq_data']
        x_omic = torch.tensor(rnaseq_data, dtype=torch.float32)

        step3_time = time.time()

        # Timing print statements
        print(
            f"Index: {index}, Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s")

        return tcga_id, days_to_event, event_occurred, images, x_omic

    def __len__(self):
        return len(self.mapping_df)


class CustomDatasetCached(Dataset):
    def __init__(self, opt, mapping_df, split=None, mode='wsi', train_val_test="train"):
        self.opt = opt
        self.train_val_test = train_val_test
        self.mapping_df = mapping_df
        self.cache_dir = "./image_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # transformations/augmentations for WSI data
        if self.train_val_test == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464])  # from lunit
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464]),
            ])

        # log transform for rnaseq data
        self.mapping_df['rnaseq_data'] = self.mapping_df['rnaseq_data'].apply(
            lambda x: np.log1p(np.array(list(x.values()))))

    def __getitem__(self, index):
        start_time = time.time()

        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.mapping_df.iloc[index]
        tcga_id = self.mapping_df.index[index]
        days_to_death = sample['days_to_death']
        days_to_last_followup = sample['days_to_last_followup']
        days_to_event = sample['time']
        event_occurred = 1 if sample['event_occurred'] == 'Dead' else 0
        tiles = sample['tiles']

        step1_time = time.time()

        # Load preprocessed images if they exist
        cached_images = []
        for tile in tiles:
            cached_image_path = os.path.join(self.cache_dir, f"{tile}.pt")
            if os.path.exists(cached_image_path):
                cached_image = torch.load(cached_image_path)
            else:
                image = Image.open(self.opt.input_wsi_path + tile).convert('RGB')
                cached_image = self.transforms(image)
                torch.save(cached_image, cached_image_path)
            cached_images.append(cached_image)

        step2_time = time.time()

        rnaseq_data = sample['rnaseq_data']
        x_omic = torch.tensor(rnaseq_data, dtype=torch.float32)

        step3_time = time.time()

        # Timing print statements
        print(
            f"Index: {index}, Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s")

        return tcga_id, days_to_event, event_occurred, cached_images, x_omic

    def __len__(self):
        return len(self.mapping_df)


# using opencv with cached images
# caching significantly accelerates data loading after the first epoch
# class CustomDatasetCachedCV(Dataset):
class CustomDataset(Dataset):
    def __init__(self, opt, mapping_df, split=None, mode='wsi', train_val_test="train"):
        self.opt = opt
        self.train_val_test = train_val_test
        self.mapping_df = mapping_df
        self.cache_dir = "./image_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # transformations/augmentations for WSI data
        # not required: already available in the functions for loading the models in generate_wsi_embeddings.py
        # if self.train_val_test == "train":
        #     # the flips and jitter transformations may not be required as we are using the pretrained lunit dino model
        #     self.transforms = transforms.Compose([
        #         # transforms.RandomHorizontalFlip(0.5),
        #         # transforms.RandomVerticalFlip(0.5),
        #         # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
        #                              std=[0.21716536, 0.26081574, 0.20723464])  # from lunit
        #     ])
        # else:
        #     self.transforms = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
        #                              std=[0.21716536, 0.26081574, 0.20723464]),
        #     ])
        # self.transforms = transforms.Compose([transforms.ToTensor()])
        self.transforms = transforms.ToTensor()
        # log transform for rnaseq data
        # not required for SPECTRA processed data
        # self.mapping_df['rnaseq_data'] = self.mapping_df['rnaseq_data'].apply(
        #     lambda x: np.log1p(np.array(list(x.values()))))

    def __getitem__(self, index):
        start_time = time.time()

        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.mapping_df.iloc[index]
        tcga_id = self.mapping_df.index[index]
        # days_to_death = sample['days_to_death']
        # days_to_last_followup = sample['days_to_last_followup']
        days_to_event = sample['time']
        event_occurred = 1 if sample['event_occurred'] == 'Dead' else 0
        tiles = sample['tiles']

        step1_time = time.time()

        # Load preprocessed images if they exist, else preprocess and cache
        cached_images = []
        for tile in tiles:
            cached_image_path = os.path.join(self.cache_dir, f"{tile}.pt")
            try:
                if os.path.exists(cached_image_path):
                    cached_image = torch.load(cached_image_path)
                    # set_trace()
                    # cached_image = self.transforms(cached_image)
                    if not isinstance(cached_image, torch.Tensor):
                        # print("Skipping ToTensor(), already a tensor [for uni]")
                        cached_image = self.transforms(cached_image)
                else:
                    raise FileNotFoundError
            except (FileNotFoundError, RuntimeError):
                image_path = os.path.join(self.opt.input_wsi_path, tile)
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"Image {tile} not found at {image_path}")

                # check if the tiles are in RGB and BGR format and convert to RGB if required


                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                cached_image = self.transforms(image)
                # cached_image = transforms.ToTensor()(image).to(device).requires_grad_()
                torch.save(cached_image, cached_image_path)
            cached_images.append(cached_image)

        step2_time = time.time()
        cached_images = torch.stack(cached_images)

        # rnaseq_data = sample['rnaseq_data']
        # set_trace()
        rnaseq_data = ast.literal_eval(sample['rnaseq_data'])
        # set_trace()
        rnaseq_values = np.array(list(rnaseq_data.values()), dtype=np.float32)
        # convert to PyTorch tensor and enable gradient flow
        x_omic = torch.from_numpy(rnaseq_values).requires_grad_()
        # x_omic = torch.tensor(rnaseq_data, dtype=torch.float32)

        step3_time = time.time()

        # Timing print statements
        print(f"Index: {index}, Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s")

        return tcga_id, days_to_event, event_occurred, cached_images, x_omic

    def __len__(self):
        return len(self.mapping_df)


class CustomDatasetDelayed(Dataset):
    def __init__(self, opt, mapping_df, split=None, mode='wsi', train_val_test="train"):
        self.opt = opt
        self.train_val_test = train_val_test
        self.mapping_df = mapping_df
        self.cache_dir = "./image_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # transformations/augmentations for WSI data
        if self.train_val_test == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464]),
            ])

        # log transform for rnaseq data
        self.mapping_df['rnaseq_data'] = self.mapping_df['rnaseq_data'].apply(
            lambda x: np.log1p(np.array(list(x.values()))))

        # Preprocess and cache images in parallel
        Parallel(n_jobs=-1)(delayed(self._preprocess_image)(tiles) for tiles in self.mapping_df['tiles'])

    def _preprocess_image(self, tiles):
        for tile in tiles:
            image_path = os.path.join(self.opt.input_wsi_path, tile)
            image = Image.open(image_path)
            image = self.transforms(image)
            cached_image_path = os.path.join(self.cache_dir, f"{tile}.h5")
            with h5py.File(cached_image_path, 'w') as f:
                f.create_dataset('image', data=image.numpy())

    def __getitem__(self, index):
        start_time = time.time()

        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.mapping_df.iloc[index]
        tcga_id = self.mapping_df.index[index]
        days_to_death = sample['days_to_death']
        days_to_last_followup = sample['days_to_last_followup']
        days_to_event = sample['time']
        event_occurred = 1 if sample['event_occurred'] == 'Dead' else 0
        tiles = sample['tiles']

        step1_time = time.time()

        # Load preprocessed images from cache
        cached_images = []
        for tile in tiles:
            cached_image_path = os.path.join(self.cache_dir, f"{tile}.h5")
            with h5py.File(cached_image_path, 'r') as f:
                cached_image = torch.tensor(f['image'][:])
            cached_images.append(cached_image)

        step2_time = time.time()

        rnaseq_data = sample['rnaseq_data']
        x_omic = torch.tensor(rnaseq_data, dtype=torch.float32)

        step3_time = time.time()

        # Timing print statements
        print(
            f"Index: {index}, Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s")

        return tcga_id, days_to_event, event_occurred, cached_images, x_omic

    def __len__(self):
        return len(self.mapping_df)


# caching the whole dataset as it doesn't change between epochs
class CustomDatasetCacheWhole(Dataset):
    def __init__(self, opt, mapping_df, split=None, mode='wsi', train_val_test="train"):
        self.opt = opt
        self.train_val_test = train_val_test
        self.mapping_df = mapping_df
        self.cache_file = 'image_cache.h5'

        # Initialize transformations
        if self.train_val_test == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464]),
            ])

        # Check if the cache file exists
        if not os.path.exists(self.cache_file):
            self.cache_data()

    def cache_data(self):
        # Create HDF5 file for caching
        with h5py.File(self.cache_file, 'w') as h5f:
            for idx, sample in self.mapping_df.iterrows():
                tiles = sample['tiles']
                for tile in tiles:
                    tile_path = self.opt.input_wsi_path + tile
                    image = Image.open(tile_path).convert('RGB')
                    transformed_image = self.transforms(image)
                    h5f.create_dataset(tile, data=transformed_image.numpy(), compression='gzip')

    def __getitem__(self, index):
        start_time = time.time()

        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.mapping_df.iloc[index]
        tcga_id = self.mapping_df.index[index]
        days_to_death = sample['days_to_death']
        days_to_last_followup = sample['days_to_last_followup']
        days_to_event = sample['time']
        event_occurred = 1 if sample['event_occurred'] == 'Dead' else 0
        tiles = sample['tiles']

        step1_time = time.time()

        # Load cached images from HDF5 file
        cached_images = []
        with h5py.File(self.cache_file, 'r') as h5f:
            for tile in tiles:
                cached_image = torch.tensor(h5f[tile][()])
                cached_images.append(cached_image)

        step2_time = time.time()

        rnaseq_data = sample['rnaseq_data']
        x_omic = torch.tensor(rnaseq_data, dtype=torch.float32)

        step3_time = time.time()

        # Timing print statements
        print(
            f"Index: {index}, Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s")

        return tcga_id, days_to_event, event_occurred, cached_images, x_omic

    def __len__(self):
        return len(self.mapping_df)


class HDF5Dataset(Dataset):
    def __init__(self, opt, h5_file, split, mode='wsi', train_val_test="train"):
        self.opt = opt
        self.train_val_test = train_val_test
        self.h5_file = h5py.File(h5_file, 'r')
        if split == 'all':
            self.dataset = self.h5_file
        else:
            self.dataset = self.h5_file[split]

        self.getitem_count = 0

        # # shared memory manager for caching
        # manager = Manager()
        # self.cache = manager.dict()

        # local cache dict
        self.cache = {}

        # Transformations/augmentations for WSI data
        if self.train_val_test == "train":
            self.transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomVerticalFlip(0.5),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464])  # from lunit
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],
                                     std=[0.21716536, 0.26081574, 0.20723464]),
            ])

    def __len__(self):
        return len(self.dataset)
        # return min(len(self.dataset), 8) # for debugging using smaller number of samples

    def __getitem__(self, index):
        self.getitem_count += 1  # Increment the counter

        # for debugging using smaller number of samples
        # if self.getitem_count > 8:
        #     raise StopIteration("stopping after 8 samples")

        start_time = time.time()
        if torch.is_tensor(index):
            index = index.tolist()

        patient_id = list(self.dataset.keys())[index]
        patient_data = self.dataset[patient_id]
        # set_trace()
        # days_to_death = patient_data['days_to_death'][()]
        # days_to_last_followup = patient_data['days_to_last_followup'][()]
        days_to_event = patient_data['days_to_event'][()]
        event_occurred = patient_data['event_occurred'][()]
        step1_time = time.time()

        rnaseq_data = patient_data['rnaseq_data'][()]
        rnaseq_data = np.log1p(rnaseq_data)  # log transformation
        x_omic = torch.tensor(rnaseq_data, dtype=torch.float32)
        step2_time = time.time()

        if patient_id not in self.cache:
            images_group = patient_data['images']
            images = []
            for key in images_group.keys():
                image_data = images_group[key][()]
                image = Image.fromarray(image_data)
                image = self.transforms(image)  # this is taking significant time
                if self.train_val_test == 'test':
                    image.requires_grad_()  # to calculate the gradient of the output w.r.t. this tensor for getting the saliency maps
                    # set_trace()
                images.append(image)
            self.cache[patient_id] = images
        else:
            images = self.cache[patient_id]
        step3_time = time.time()

        # images_group = patient_data['images']
        # images = []
        # for key in images_group.keys():
        #     image_data = images_group[key][()]
        #     image = Image.fromarray(image_data)
        #     image = self.transforms(image)
        #     images.append(image)
        # step3_time = time.time()

        # print(
        #     f"Index: {index}, loaded {self.getitem_count} of {len(self.dataset) / torch.cuda.device_count()} samples [total samples: {len(self.dataset)} ], Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s")

        return patient_id, days_to_event, event_occurred, images, x_omic
