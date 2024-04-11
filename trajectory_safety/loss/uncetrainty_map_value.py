import cv2
import os
import torch
import numpy as np
from utils.geometry import get_point_bev_pixel_coordinate

def get_point_bev_pixel_coordinate(points, image_size_x, image_size_y, bev_range):
    pixel_y = image_size_y - (points[:, 0] * (image_size_y/(bev_range*2)) + (image_size_y/2))
    pixel_x = points[:, 1] * (image_size_x/(bev_range*2)) + (image_size_x/2)
    return pixel_y, pixel_x

class UncertaintyValueLoss:

    def __init__(self, uncertainty_map_path, image_size_x=224, image_size_y=224, bev_range=50.0):
        self.uncertainty_map_path = '../FIERY_ENN1/enn_A'
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.bev_range = bev_range

    def __call__(self, batch_trajectory: torch.Tensor, batch_sample_token) -> torch.Tensor:

        batch_values = []
        for trajectory, sample_token in zip(batch_trajectory, batch_sample_token):
            agent_image = cv2.imread(os.path.join(self.uncertainty_map_path, sample_token))
            values = []
            ys, xs = get_point_bev_pixel_coordinate(trajectory, self.image_size_x, self.image_size_y, self.bev_range)
            for y, x in zip(ys, xs):
                value = agent_image[y,x]
                values.append(value)
            batch_values.append(values)

        return torch.mean(batch_values)
