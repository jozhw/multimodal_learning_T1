# code to train models, or use pretrained models to generate WSI embeddings
import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from pdb import set_trace

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256


def extract_tcga_id(filename):
    parts = filename.split(
        "-")  # sample png file name: TCGA-02-0011-01Z-00-DX1.7CF44982-AE5A-47A5-8D07-E526522DC094_1.png
    tcga_id = "-".join(parts[:3])
    return tcga_id


# create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)
        # self.images = [os.listdir(os.path.join(self.image_dir, dir)) for dir in os.listdir(self.image_dir)]
        # self.images = [item for sublist in self.images for item in sublist]  # list of all tiles from all samples
        # set_trace()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        tcga_id = extract_tcga_id(self.images[idx])

        if self.transform:
            image = self.transform(image)

        return image, tcga_id, self.images[idx]


transform = Compose([
    Resize((224, 224)),  # resize image to 224x224 for the model
    ToTensor(),
    # normalization parameters for lunit DINO from https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
    # mean: [ 0.70322989, 0.53606487, 0.66096631 ]
    # std: [ 0.21716536, 0.26081574, 0.20723464 ]
    Normalize(mean=[0.70322989, 0.53606487, 0.66096631], std=[0.21716536, 0.26081574, 0.20723464]),
])

dataset = CustomDataset(
    image_dir='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/preprocessing/TCGA_WSI/batch_corrected/processed_svs/tiles/256px_9.9x/combined_tiles/',
    transform=transform
)

# create the dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # can set shuffle to False for inference


# Use the pretrained Lunit-DINO model (Kang 2023) for WSI feature extraction (trained on histopathology images)
# Lunit-DINO uses the ViT-S architecture with DINO for SSL

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model


# initialize ViT-S/16 trunk using DINO pre-trained weight
model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters in the model: {total_params / 1e6}M")

total_images_processed = 0
total_images = len(dataset)
# set_trace()
# Generate embeddings using the above model
model.eval()
embeddings_list = []
tcga_ids_list = []
png_filenames_list = []
with torch.no_grad():
    for images, tcga_ids, png_filenames in dataloader:
        batch_size = images.shape[0]
        total_images_processed += batch_size
        print(
            f"Processing {batch_size} images in this batch. Total processed: {total_images_processed}/{total_images}.")
        print("TCGA IDs in this batch: ", tcga_ids)
        print("PNG files in this batch: ", png_filenames)
        images = images.to(device)
        outputs = model(images)
        embeddings_list.extend(outputs.cpu().numpy())
        tcga_ids_list.extend(tcga_ids)
        png_filenames_list.extend(png_filenames)

        # if total_images_processed > 16:
        #     break

# save the embeddings alowngiwth the tcga ids in a file
embeddings_file = "./lunit-DINO-embeddings-LUAD.csv"
with open(embeddings_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['tilename', 'TCGA_ID', 'embedding'])
    for fname, tcga_id, embedding in zip(png_filenames_list, tcga_ids_list, embeddings_list):
        writer.writerow([fname, tcga_id, embedding.tolist()])
