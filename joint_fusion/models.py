import torch
import torch.nn as nn
import torch.nn.functional as F
from generate_rnaseq_embeddings import get_omic_embeddings
from generate_wsi_embeddings import WSIEncoder
import torchvision.models as models
from pdb import set_trace
from torchvision.models.vision_transformer import vit_b_32
from torchvision.models import ViT_B_32_Weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # can train the whole model or freeze certain layers
        resnet18 = models.resnet18(pretrained=True)
        # remove the fully connected layer (classifier) and the final pooling layer
        # extract the final set of features
        # https://stackoverflow.com/questions/55083642/extract-features-from-last-hidden-layer-pytorch-resnet18
        layers = list(resnet18.children())[:-2]
        num_features_extracted = 512  # fixed for resnet18

        # self.net = nn.Sequential(
        #     *layers,
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(num_features_extracted, embedding_dim),
        #     nn.ReLU()
        # )
        # # set_trace()

        # vision transformer vit_b_32 arch ('base' version with 32 x 32 patches)
        vit = vit_b_32(weights=ViT_B_32_Weights)
        layers = list(vit.children())  # get all the layers
        vit_top = nn.Sequential(*layers[:-2])  # remove the normalization and the pooling layer
        self.feature_extractor = nn.Sequential(*layers)
        num_features = 768
        # self.embedding_layer = nn.Linear(num_features, embedding_dim)
        self.net = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(num_features, embedding_dim)
        )

    def forward(self, x):
        print("+++++++++++++ Input shape within WSINetwork: ", x.shape)
        return self.net(x)


class OmicNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(OmicNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(320, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        print("+++++++++++++ Input shape within omic network: ", x.shape)
        return self.net(x)


class MultimodalNetwork(nn.Module):
    def __init__(self, embedding_dim_wsi, embedding_dim_omic, mode, fusion_type):
        super(MultimodalNetwork, self).__init__()

        self.mode = mode  # wsi_omic, wsi or omic
        self.fusion_type = fusion_type
        self.wsi_encoder = WSIEncoder() # use for early fusion using the pretrained foundation model for generating tile embeddings
        if self.mode != 'wsi_omic':
            self.fusion_type = None
        print(f"Initializing with mode={mode}, fusion_type={fusion_type}")
        # if self.fusion_type is not None: # not unimodal
        if self.mode == 'wsi_omic':
            self.wsi_net = WSINetwork(embedding_dim_wsi)
            self.omic_net = OmicNetwork(embedding_dim_omic)
            embedding_dim = self.wsi_net.embedding_dim + self.omic_net.embedding_dim
        elif self.mode == 'wsi':
            self.wsi_net = WSINetwork(embedding_dim_wsi)
            self.omic_net = None
            embedding_dim = self.wsi_net.embedding_dim
        elif self.mode == 'omic':
            self.wsi_net = None
            self.omic_net = OmicNetwork(embedding_dim_omic)
            embedding_dim = self.omic_net.embedding_dim

        # embedding_dim = embedding_dim_omic
        print("Embedding dimension based on which the dimension of the input of the downstream MLP is set: ", embedding_dim)
        # downstream MLP for fused data (directly tied to the Cox loss)
        self.fused_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.stored_omic_embedding = None

    # def get_omic_embedding(self):
    #     # get the embeddings from a stored file (obtained from the latent space of a trained VAE)
    #     omic_embeddings =

    def forward(self, x_wsi, x_omic):
        if self.fusion_type == 'joint':
            wsi_embedding = self.wsi_net(x_wsi)
            omic_embedding = self.omic_net(x_omic)
            print("wsi_embedding.shape: ", wsi_embedding.shape)
            print("omic_embedding.shape: ", omic_embedding.shape)

        elif self.fusion_type == 'early':
            wsi_embedding = self.wsi_encoder.get_wsi_embeddings(x_wsi)  # get from pretrained foundation models; x_wsi contain data from all tiles
            omic_embedding = get_omic_embeddings(x_omic)  # get from a simple VAE based encoder
            print("wsi_embedding.shape: ", wsi_embedding.shape)
            print("omic_embedding.shape: ", omic_embedding.shape)

        elif self.fusion_type is None:  # unimodal case
            print("This is a unimodal case")
            if self.mode == 'wsi':
                wsi_embedding = self.wsi_encoder.get_wsi_embeddings(x_wsi) # x_wsi contain data from all tiles
                print("wsi_embedding.shape (should be [batch_size, embedding_dim]): ", wsi_embedding.shape)
            elif self.mode == 'omic':
                if self.stored_omic_embedding is None and x_omic is not None: # to avoid calling this function for every forward pass
                    self.stored_omic_embedding = get_omic_embeddings(x_omic)
                    self.stored_omic_embedding = torch.tensor(self.stored_omic_embedding, dtype=torch.float32).to(x_wsi[0].device)
                    print("omic_embedding.shape (should be [batch_size, embedding_dim]): ", self.stored_omic_embedding.shape)
                omic_embedding = self.stored_omic_embedding # reuse the stored embeddings for early fusion

        print("input mode: ", self.mode)

        # concatenate embeddings
        if self.mode == 'wsi':
            combined_embedding = wsi_embedding
        elif self.mode == 'omic':
            combined_embedding = omic_embedding
        elif self.mode == 'wsi_omic':
            wsi_embedding_tensor = torch.tensor(wsi_embedding)
            omic_embedding_tensor = torch.tensor(omic_embedding)
            combined_embedding = torch.cat((wsi_embedding_tensor, omic_embedding_tensor), dim=1)

        combined_embedding = torch.tensor(combined_embedding).to(device)
        print("combined_embedding.shape: ", combined_embedding.shape)
        # use combined embedding with downstream MLP for getting the output that enters the loss function
        output = self.fused_mlp(combined_embedding)

        return output


def print_model_summary(model):
    if model is None:
        print("model is NoneType")
        return
    total_params = sum(p.numel() for p in model.parameters())
    print("NOTE: these do not account for the memory required for storing the optimizer states and the activations")
    print(f"Total Parameters (million): {total_params / 1e6}")
    memory_bytes = total_params * 4  # 4 bytes for a torch.float32 model parameter
    # memory_mb = memory_bytes / (1024 ** 2)
    memory_gb = memory_bytes / 1e9
    print(f"Estimated Memory (GB): {memory_gb}")
