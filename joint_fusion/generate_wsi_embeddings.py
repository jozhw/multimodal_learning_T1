# code to train models, or use pretrained models to generate WSI embeddings
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from pdb import set_trace

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        
        return image

transform = Compose([
    Resize((224, 224)),  # resize image to 224x224 for the model
    ToTensor(),
    # normalization parameters for lunit DINO from https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
    # mean: [ 0.70322989, 0.53606487, 0.66096631 ]
    # std: [ 0.21716536, 0.26081574, 0.20723464 ]
    Normalize(mean=[ 0.70322989, 0.53606487, 0.66096631 ], std=[ 0.21716536, 0.26081574, 0.20723464 ]),
])

dataset = CustomDataset(
    image_dir='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/data_from_pathomic_fusion/data/TCGA_GBMLGG/TCGA_GBMLGG/TCGA_GBMLGG/all_st',
    transform=transform
)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# Use the pretrained Lunit-DINO model for WSI feature extraction (trained on histopathology images)
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
print(f"Total number of trainable parameters in the model: {total_params/1e6}M")

total_images_processed = 0
total_images = len(dataset)
set_trace()
# Generate embeddings using the above model
model.eval()
with torch.no_grad():
    for images in dataloader:
        batch_size = images.shape[0]
        total_images_processed += batch_size
        print(f"Processing {batch_size} images in this batch. Total processed: {total_images_processed}/{total_images}.")
        images = images.to(device)
        outputs = model(images)
        set_trace()

