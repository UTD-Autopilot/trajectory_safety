import cv2
import os
import torch
import numpy as np
import cv2

import matplotlib.pyplot as plt

def get_point_bev_pixel_coordinate(points, image_size_x, image_size_y, bev_range):
    pixel_x = points[:, 1] * (image_size_x/(bev_range*2)) + (image_size_x/2)
    pixel_y = (-points[:, 0]) * (image_size_y/(bev_range*2)) + (image_size_y/2)
    pixel_x = pixel_x.astype(int)
    pixel_y = pixel_y.astype(int)
    return pixel_y, pixel_x

class UncertaintyValueLoss:
    def __init__(self, uncertainty_map_path, trajectories, bev_size_x=224, bev_size_y=224, bev_range=50.0):
        self.uncertainty_map_path = uncertainty_map_path    # '../FIERY_ENN1/enn_A'
        self.trajectories = trajectories
        self.bev_size_x = bev_size_x
        self.bev_size_y = bev_size_y
        self.bev_range = bev_range

    def get_lattice_uncertainty_values(self, sample_token):
        batch_values = []
        for trajectory in self.trajectories:
            uncertainty_map = np.load(os.path.join(self.uncertainty_map_path, sample_token+'.npy'))
            uncertainty_map = cv2.resize(uncertainty_map, (self.bev_size_x, self.bev_size_y), interpolation=cv2.INTER_LINEAR)
            uncertainty_map = np.flip(uncertainty_map, [0,1])
            values = []
            ys, xs = get_point_bev_pixel_coordinate(trajectory, self.bev_size_x, self.bev_size_y, self.bev_range)
            # print(trajectory)
            # print(xs)
            # print(ys)
            for y, x in zip(ys, xs):
                value = uncertainty_map[y][x]
                values.append(value)
            batch_values.append(values)

            # plt.clf()
            # plt.imshow(uncertainty_map)
            # plt.colorbar()
            # plt.scatter(xs, ys, c='red')
            # plt.savefig('tmp/uncertainty_value_loss_traj.png')
        return np.array(batch_values)

    def __call__(self, batch_logits: torch.Tensor, batch_sample_token) -> torch.Tensor:
        losses = []
        for logits, sample_token in zip(batch_logits, batch_sample_token):
            uncertainty_values = torch.tensor(self.get_lattice_uncertainty_values(sample_token), device=logits.device)
            # shape of uncertainty_values is (len(trajectory), timestep), we average over timestep dim
            loss = logits * uncertainty_values.mean(dim=-1)
            losses.append(loss)
        return torch.mean(torch.stack(losses))
