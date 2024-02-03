import torch.nn as nn
import torchvision.models as models
from pdb import set_trace

class MultimodalNetwork(nn.Module):
    def __init__(self, opt, cv_id):
        super(MultimodalNetwork, self).__init__()
        self.linear1 = nn.Linear(in_features=16,
                                 out_features=16)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=16,
                                 out_features=16)

        # self.omic_net = omicNetwork()
        # self.wsi_net = wsiNetwork()

    def forward(self, **kwargs):
        x = kwargs['x_path']
        # Get the input dimension dynamically
        input_dim = x.size(-1)
        hidden_size = 16
        output_dim = 16

        self.linear1.in_features = input_dim
        self.linear1.out_features = hidden_size
        self.linear2.in_features = hidden_size
        self.linear2.out_features = output_dim

        # Pass input through linear and ReLU layers
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        # set_trace()
        return x
        # path_vec = kwargs['x_path']
        # features = self.fusion(path_vec, omic_vec)
        # return features


class wsiNetwork(nn.Module):
    def __init__():
        pass


class omicNetwork(nn.Module):
    def __init__():
        pass

