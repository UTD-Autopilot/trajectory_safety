import os
import matplotlib.pyplot as plt

from trajectory_safety.datasets.carla import CarlaTrajectoryPredictionDataset
from trajectory_safety.datasets.nuscenes import NuScenesTrajectoryPredictionDataset
from torch.utils.data import Dataset, DataLoader, random_split

def get_dataloader_carla(records_folder='/home/txw190000/data/Workspace/data/carla', batch_size=64):
    train_records_folder = os.path.join(records_folder, "train")
    test_records_folder = os.path.join(records_folder, "test")

    train_dataset_paths = []
    
    for name in os.listdir(train_records_folder):
        dir = os.path.join(train_records_folder, name)
        if os.path.isdir(dir):
            train_dataset_paths.append(dir)

    test_dataset_paths = []
    
    for name in os.listdir(test_records_folder):
        dir = os.path.join(test_records_folder, name)
        if os.path.isdir(dir):
            test_dataset_paths.append(dir)

    print("Training:", train_dataset_paths)
    print("Testing:", test_dataset_paths)

    train_dataset = CarlaTrajectoryPredictionDataset(train_dataset_paths)
    test_dataset = CarlaTrajectoryPredictionDataset(test_dataset_paths)

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    print("Training&testing dataset size:", train_size, test_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


def get_dataloader_nuscenes(dataset_path='../data/nuscenes/trainval', split='trainval', batch_size=64):

    if split == 'trainval':
        train_dataset = NuScenesTrajectoryPredictionDataset(dataset_path, 'train')
        test_dataset = NuScenesTrajectoryPredictionDataset(dataset_path, 'val')
    elif split == 'mini':
        train_dataset = NuScenesTrajectoryPredictionDataset(dataset_path, 'mini_train')
        test_dataset = NuScenesTrajectoryPredictionDataset(dataset_path, 'mini_val')
    else:
        raise ValueError('Split must be mini or trainval.')

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    print("Training&testing dataset size:", train_size, test_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader
