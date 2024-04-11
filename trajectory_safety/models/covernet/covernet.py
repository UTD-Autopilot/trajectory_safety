# nuScenes dev-kit.
# Code written by Freddy Boulton, Tung Phan 2020.
from typing import List, Tuple, Callable, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as f

from .backbone import calculate_backbone_feature_dim


class CoverNet(nn.Module):
    """ Implementation of CoverNet https://arxiv.org/pdf/1911.10298.pdf """

    def __init__(self, backbone: nn.Module, num_modes: int,
                 n_hidden_layers: List[int] = None,
                 input_shape: Tuple[int, int, int] = (3, 500, 500),
                 state_vector_shape: int = 3):
        """
        Inits Covernet.
        :param backbone: Backbone model. Typically ResNetBackBone or MobileNetBackbone
        :param num_modes: Number of modes in the lattice
        :param n_hidden_layers: List of dimensions in the fully connected layers after the backbones.
            If None, set to [4096]
        :param input_shape: Shape of image input. Used to determine the dimensionality of the feature
            vector after the CNN backbone.
        """

        if n_hidden_layers and not isinstance(n_hidden_layers, list):
            raise ValueError(f"Param n_hidden_layers must be a list. Received {type(n_hidden_layers)}")

        super().__init__()

        if not n_hidden_layers:
            n_hidden_layers = [4096]

        self.backbone = backbone

        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)
        n_hidden_layers = [backbone_feature_dim + state_vector_shape] + n_hidden_layers + [num_modes]

        linear_layers = [nn.Linear(in_dim, out_dim)
                         for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:])]

        self.head = nn.ModuleList(linear_layers)
        self.relu = nn.ReLU()

    def forward(self, image_tensor: torch.Tensor,
                agent_state_vector: torch.Tensor) -> torch.Tensor:
        """
        :param image_tensor: Tensor of images in the batch.
        :param agent_state_vector: Tensor of agent state vectors in the batch
        :return: Logits for the batch.
        """

        backbone_features = self.backbone(image_tensor)

        logits = torch.cat([backbone_features, agent_state_vector], dim=1)

        for linear in self.head:
            logits = self.relu(linear(logits))

        return logits
