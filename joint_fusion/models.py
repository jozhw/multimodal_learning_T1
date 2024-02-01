import torch.nn as nn
import torchvision.models as models

class MultimodalNetwork(nn.Module):
    def __init__(self, gene_input_dim, num_classes):
        super(MultimodalNetwork, self).__init__()
        # Image modality sub-model: A pre-trained CNN without the final layer
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final fully connected layer

        # Gene expression modality sub-model: A simple feedforward network
        self.gene_ffn = nn.Sequential(
            nn.Linear(gene_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Fusion and classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256 + 512, 256),  # Assuming resnet18's penultimate features are 512-dimensional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, gene_expression):
        image_features = self.cnn(image)
        gene_features = self.gene_ffn(gene_expression)
        combined_features = torch.cat((image_features, gene_features), dim=1)
        output = self.classifier(combined_features)
        return output



