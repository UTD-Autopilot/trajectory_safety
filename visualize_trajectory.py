import os
import torch
import pickle
import argparse

import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from trajectory_safety.datasets import get_dataloader_nuscenes, get_dataloader_carla
from trajectory_safety.datasets.utils import rotate_points
from trajectory_safety.models.covernet.backbone import ResNetBackbone
from trajectory_safety.models.covernet.covernet import CoverNet
from trajectory_safety.loss.constant_lattice import ConstantLatticeLoss


def visualize_trajectory(dataset='nuscenes_mini', lattice_set='epsilon_4'):
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    image_size_x = 224
    image_size_y = 224
    bev_range = 50.0

    if dataset == 'nuscenes':
        train_loader, test_loader = get_dataloader_nuscenes(batch_size=1)
    elif dataset == 'nuscenes_mini':
        train_loader, test_loader = get_dataloader_nuscenes(split='mini', batch_size=1)
    elif dataset == 'carla':
        train_loader, test_loader = get_dataloader_carla(batch_size=1)
    else:
        raise NotImplementedError(f'Dataset {dataset} not supported.')
    
    with open(f'trajectory_safety/models/covernet/lattice/{lattice_set}.pkl', 'rb') as f:
        lattice = pickle.load(f)
        # In our data the vehicle is heading to the x axis, but the lattice is assuming the vehicle
        # is heading toward y axis, so we need to rotate these -90 degrees.
        lattice = rotate_points(np.array(lattice), -90)
    
    backbone = ResNetBackbone('resnet50')

    lattice = np.array(lattice)
    model = CoverNet(backbone, num_modes=lattice.shape[0], input_shape=(3,224,224), state_vector_shape=3)
    model = torch.nn.DataParallel(model)
    model.to(device)

    save_dir = f'saves/{dataset}/resnet50_{lattice_set}/'
    latest_model_timestamp = None
    latest_model_path = None
    for filename in os.listdir(save_dir):
        if not filename.startswith('model'):
            continue
        timestamp = os.path.getmtime(os.path.join(save_dir, filename))
        if latest_model_timestamp is None or timestamp > latest_model_timestamp:
            latest_model_timestamp = timestamp
            latest_model_path = os.path.join(save_dir, filename)

    try:
        model.load_state_dict(torch.load(latest_model_path))
    except Exception as e:
        print(e)

    
    for i, data in enumerate(test_loader):
    
        figure, ax = plt.subplots(figsize=(10, 8))
        plt.title("Trajectory Prediction")

        image, state_vector, gt_trajectory = data

        with torch.no_grad():
            logits = model(image, state_vector)
            _, predictions = torch.topk(logits, 1, 1)
            predictions = predictions.detach().cpu().numpy()[0]
        pred_trajectories = lattice[predictions]

        plt.imshow(np.moveaxis(image[0].cpu().numpy(), 0, 2))

        # print('pred_trajectories', pred_trajectories.shape)
        # print('gt_trajectory', gt_trajectory.shape)
        for trajectory in gt_trajectory[0]:
            # Transform the trejectory to image space for plotting
            trajectory_x = trajectory[1] * (image_size_x/(bev_range*2)) + (image_size_x/2)
            trajectory_y = (-trajectory[0]) * (image_size_y/(bev_range*2)) + (image_size_y/2)
            if dataset in ['nuscenes', 'nuscenes_mini']:
                trajectory_y += 62.5 # nuscenes agent vehicle is not in the center
            ax.scatter(trajectory_x, trajectory_y, color='red')

        for trajectory in pred_trajectories[0]:
            # Transform the trejectory to image space for plotting
            trajectory_x = trajectory[1] * (image_size_x/(bev_range*2)) + (image_size_x/2)
            trajectory_y = (-trajectory[0]) * (image_size_y/(bev_range*2)) + (image_size_y/2)
            if dataset in ['nuscenes', 'nuscenes_mini']:
                trajectory_y += 62.5
            ax.scatter(trajectory_x, trajectory_y, color='blue') 
        
        os.makedirs(f'plots/{dataset}', exist_ok=True)
        plt.savefig(f'plots/{dataset}/{i}.png')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Covernet trajectory visualization'
    )

    parser.add_argument('-d', '--dataset', type=str, choices=['nuscenes', 'nuscenes_mini', 'carla'])
    parser.add_argument('-g', '--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in args.gpus])

    visualize_trajectory(dataset=args.dataset)