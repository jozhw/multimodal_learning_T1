import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pdb import set_trace


# make the output(embedding) dimension a hyperparameter
class WSINetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(WSINetwork, self).__init__()
        self.embedding_dim = embedding_dim

        # self.net = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(32 * 256 * 256, embedding_dim),
        #     nn.ReLU()
        # )

        # self.net = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(32, embedding_dim),
        #     nn.ReLU()
        # )

        # 18 layer resnet
        # self.resnet = models.resnet18(pretrained=False)

        # remove the fully connected layer (classifier) and the final pooling layer
        # extract the final set of features
        # https://stackoverflow.com/questions/55083642/extract-features-from-last-hidden-layer-pytorch-resnet18
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]) # is of shape [batch_size, 512, 32, 32]
        # go from above to the embedding layer
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        resnet18 = models.resnet18(pretrained=False)
        # remove the fully connected layer (classifier) and the final pooling layer
        # extract the final set of features
        # https://stackoverflow.com/questions/55083642/extract-features-from-last-hidden-layer-pytorch-resnet18
        layers = list(resnet18.children())[:-2]
        num_features_extracted = 512  # fixed for resnet18

        self.net = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features_extracted, embedding_dim),
            nn.ReLU()
        )
        # set_trace()

    def forward(self, x):
        print("+++++++++++++ Input shape within WSINetwork: ", x.shape)
        return self.net(x)

class OmicNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(OmicNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(320, 512),  # the molecular/genomic data has 320 features
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        print("+++++++++++++ Input shape within omic network: ", x.shape)
        return self.net(x)


class MultimodalNetwork(nn.Module):
    def __init__(self, embedding_dim_wsi, embedding_dim_omic, mode):
        super(MultimodalNetwork, self).__init__()
        self.wsi_net = WSINetwork(embedding_dim_wsi)
        self.omic_net = OmicNetwork(embedding_dim_omic)
        self.mode = mode
        embedding_dim = self.wsi_net.embedding_dim + self.omic_net.embedding_dim
        # downstream MLP for fused data
        self.fused_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x_wsi, x_omic):
        wsi_embedding = self.wsi_net(x_wsi)
        omic_embedding = self.omic_net(x_omic)

        print("input mode: ", self.mode)
        print("wsi_embedding.shape: ", wsi_embedding.shape)
        print("omic_embedding.shape: ", omic_embedding.shape)
        # concatenate embeddings
        if self.mode == 'wsi':
            combined_embedding = wsi_embedding
        elif self.mode == 'omic':
            combined_embedding = omic_embedding
        else:
            combined_embedding = torch.cat((wsi_embedding, omic_embedding), dim=1)

        # print("combined_embedding.shape: ", combined_embedding.shape)

        # use combined embedding with downstream MLP
        output = self.fused_mlp(combined_embedding)

        return output


def print_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    print("NOTE: these do not account for the memory required for storing the optimizer states and the activations")
    print(f"Total Parameters (million): {total_params / 1e6}")
    memory_bytes = total_params * 4  # 4 bytes for a torch.float32 model parameter
    # memory_mb = memory_bytes / (1024 ** 2)
    memory_gb = memory_bytes / 1e9
    print(f"Estimated Memory (GB): {memory_gb}")
