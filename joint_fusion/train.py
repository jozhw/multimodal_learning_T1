import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import datasets
import models

image_paths = ["path/to/image1", "path/to/image2"]  
gene_data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  
labels = [0, 1]  

def train(model, dataloader, optimizer):
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

dataset = MultimodalDataset(image_paths, gene_data, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = MultimodalNetwork(gene_input_dim=3, num_classes=2)  

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for images, gene_expressions, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images, gene_expressions)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")



