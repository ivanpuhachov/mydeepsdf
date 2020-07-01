import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn


class SingleShapeSDF(nn.Module):
    # TODO: initialization
    # TODO: weight normalization
    # TODO: dropouts
    def __init__(self, layer_dims):
        super(SingleShapeSDF, self).__init__()
        input_dim = 3
        output_dim = 1
        layer_dims.append(output_dim)
        layer_dims.insert(0, input_dim)
        self.num_layers = len(layer_dims)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=0.2)
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
    #     self.init_weights()
    #
    # def init_weights(self):
    #     for layer in range(self.num_layers-1):
    #         nn.init.normal_(self.layers[layer].weight, 0, 0.1)
    #         nn.init.normal_(self.layers[layer].bias, 0, 1)

    def forward(self, x):
        if x.is_cuda:
            device = x.get_device()
        else:
            device = torch.device("cpu")
        x.to(device)
        for layer in range(self.num_layers - 2):
            if layer > 0:
                x = self.dropout(x)
            x = self.layers[layer](x)
            x = nn.functional.relu(x)
        x = self.layers[-1](x)
        x = torch.tanh(x)
        return x


def sdfblock(in_dim, out_dim, height_dim):
    return nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.utils.weight_norm(nn.Linear(in_dim, height_dim)),
                    nn.LeakyReLU(inplace=True),

                    nn.Dropout(p=0.2),
                    nn.utils.weight_norm(nn.Linear(height_dim, height_dim)),
                    nn.LeakyReLU(inplace=True),

                    nn.Dropout(p=0.2),
                    nn.Linear(height_dim, height_dim),
                    nn.ReLU(inplace=True),

                    nn.Dropout(p=0.2),
                    nn.utils.weight_norm(nn.Linear(height_dim, out_dim)),
                    nn.LeakyReLU(inplace=True),
                )

class FamilyShapeDecoderSDF(nn.Module):
    def __init__(self, family_size=20, latent_size=256, n_blocks=2, h_blocks=512):
        """

        :param family_size: number of shapes in the family
        :param latent_size: dimensionality of latent vector space
        :param n_blocks: number of residual blocks (TODO: implement)
        :param h_blocks: height of redisual blocks (number of neurons)
        """
        super(FamilyShapeDecoderSDF, self).__init__()
        self.latent_size = latent_size
        self.latent_vector = torch.nn.Parameter(torch.randn([family_size, latent_size], dtype=torch.float32))
        assert h_blocks > latent_size + 3
        # self.n_blocks = n_blocks
        self.h_blocks = h_blocks
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.block0 = sdfblock(in_dim=3+self.latent_size, out_dim=self.h_blocks-3-self.latent_size, height_dim=self.h_blocks)
        self.block1 = sdfblock(in_dim=self.h_blocks, out_dim=1, height_dim=self.h_blocks)
        self.blocks = []
        # TODO: weights intitialization

    def forward(self, xyz, family_id):
        return self.forward_customlatent(xyz, self.latent_vector[family_id])

    def forward_customlatent(self, xyz, latent):
        batchsize = xyz.shape[0]
        inputs = torch.cat([latent.repeat(batchsize, 1), xyz], dim=1)
        x = inputs
        # TODO: weight norm
        x = self.block0(x)
        x = torch.cat([inputs, x], dim=1)
        x = self.block1(x)
        x = torch.tanh(x)
        return x


def deepsdfloss(outputs, targets):
    # TODO: move that delta somewhere to clean up the code
    delta = 0.1
    # TODO: investigate nn.MSELoss()(torch.clamp(...) -...)
    return torch.mean(torch.abs(
        torch.clamp(outputs, min=-delta, max=delta) - torch.clamp(targets, min=-delta, max=delta)
    ))
