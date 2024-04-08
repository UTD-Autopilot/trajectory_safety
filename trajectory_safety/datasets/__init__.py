import os
import matplotlib.pyplot as plt

from trajectory_safety.datasets.carla import CarlaTrajectoryPredictionDataset
from trajectory_safety.datasets.nuscenes import NuScenesTrajectoryPredictionDataset
from torch.utils.data import Dataset, DataLoader, random_split

def get_dataloader_carla(records_folder='/home/txw190000/data/Workspace/data/carla'):
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

    print(len(train_dataset))
    img, state_vector, y = train_dataset[1840]
    #plt.imshow(np.array(img).transpose(1, 2, 0))
    print(y)

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    print("Training&testing dataset size:", train_size, test_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


def get_dataloader_nuscenes():
    nuscenes_path = '../data/nuscenes/mini'
    train_dataset = NuScenesTrajectoryPredictionDataset(nuscenes_path, 'mini_train')
    test_dataset = NuScenesTrajectoryPredictionDataset(nuscenes_path, 'mini_val')

    print(len(train_dataset))
    img, state_vector, y = train_dataset[0]
    plt.imshow(np.array(img).transpose(1, 2, 0))
    plt.savefig('debug.png')
    print(y)

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    print("Training&testing dataset size:", train_size, test_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader
