import numpy as np
import matplotlib.pyplot as plt

from trajectory_safety.datasets.carla import CarlaTrajectoryPredictionDataset
from trajectory_safety.datasets.utils import rotate_points

if __name__ == '__main__':
    dataset = CarlaTrajectoryPredictionDataset([
        '/home/txw190000/data/Workspace/data/carla/train/Town06_100npcs_1'
    ])

    for i in range(50):
        image, state_vector, gt_trajectory = dataset[i]

        figure, ax = plt.subplots(figsize=(10, 8))
        plt.title("Trajectory Prediction")
        plt.imshow(np.moveaxis(image.cpu().numpy(), 0, 2))

        bev_range = 50
        image_size_x = image.shape[1]
        image_size_y = image.shape[2]

        trajectory_x = gt_trajectory[:, 1] * (image_size_x/(bev_range*2)) + (image_size_x/2)
        trajectory_y = (-gt_trajectory[:, 0]) * (image_size_y/(bev_range*2)) + (image_size_y/2)
        ax.scatter(trajectory_x, trajectory_y, color='blue')

        plt.savefig(f'tmp/dataset_debug_{i}.png')
        plt.close()
