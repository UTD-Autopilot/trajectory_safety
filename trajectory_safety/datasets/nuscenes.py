import os
import random
import warnings
import torchvision
import torch
import numpy as np
from torchvision import transforms

from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from shapely.errors import ShapelyDeprecationWarning
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

from .utils import rotate_points

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


class NuScenesTrajectoryPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split='train'):
        # split must be one of ['mini_train', 'mini_val', 'train', 'train_val', 'val']

        self.dataset_path = dataset_path
        if split.startswith('mini_'):
            self.nusc = NuScenes('v1.0-mini', dataroot=self.dataset_path, verbose=False)
        else:
            self.nusc = NuScenes('v1.0-trainval', dataroot=self.dataset_path, verbose=False)
        self.helper = PredictHelper(self.nusc)
        self.sample_ids = get_prediction_challenge_split("mini_train", dataroot=self.dataset_path)

        static_layer_rasterizer = StaticLayerRasterizer(self.helper)
        agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=1)
        self.mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        instance_token, sample_token = self.sample_ids[index].split("_")
        future_xy_local = self.helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
        img = self.mtp_input_representation.make_input_representation(instance_token, sample_token)
        agent_state_vector = torch.Tensor([[self.helper.get_velocity_for_agent(instance_token, sample_token),
                                    self.helper.get_acceleration_for_agent(instance_token, sample_token),
                                    self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)]]).reshape(-1)
        agent_state_vector = torch.nan_to_num(agent_state_vector, 0)
        if self.transform is not None:
            img = self.transform(img)

        future_xy_local = rotate_points(future_xy_local, -90)
        return img, agent_state_vector, future_xy_local
