import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pdb import set_trace


# make the output dimension a hyperparameter
class WSINetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(WSINetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 256 * 256, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class OmicNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(OmicNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(320, 512), # the molecular data has 320 features
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class MultimodalNetwork(nn.Module):
    def __init__(self):
        super(MultimodalNetwork, self).__init__()
        self.wsi_net = WSINetwork(embedding_dim=128)
        self.omic_net = OmicNetwork(embedding_dim=128)

        # downstream MLP for fused data
        self.downstream_mlp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x_wsi, x_omic):
        wsi_embedding = self.wsi_net(x_wsi)
        omic_embedding = self.omic_net(x_omic)

        # concatenate embeddings
        combined_embedding = torch.cat((wsi_embedding, omic_embedding), dim=1)

        # process combined embedding with downstream MLP
        output = self.downstream_mlp(combined_embedding)

        return output
