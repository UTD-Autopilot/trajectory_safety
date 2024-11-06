import pickle
import numpy as np
import matplotlib.pyplot as plt

from trajectory_safety.loss.uncetrainty_map_value import UncertaintyValueLoss
from trajectory_safety.datasets import CachedDataset
from trajectory_safety.datasets.nuscenes import NuScenesTrajectoryPredictionDataset
from trajectory_safety.datasets.utils import rotate_points

if __name__ == '__main__':
    
    dataset_path = 'data/nuscenes/mini'
    dataset = NuScenesTrajectoryPredictionDataset(dataset_path, 'mini_train')

    lattice_set='epsilon_4'
    with open(f'trajectory_safety/models/covernet/lattice/{lattice_set}.pkl', 'rb') as f:
        lattice = pickle.load(f)
        # In our data the vehicle is heading to the x axis, but the lattice is assuming the vehicle
        # is heading toward y axis, so we need to rotate these -90 degrees.
        lattice = rotate_points(np.array(lattice), -90)

    print('len(lattice):', len(lattice))
    lattice = np.array(lattice)

    criterion = UncertaintyValueLoss('data/enn_A', lattice)

    image, state_vector, gt_trajectory, sample_token = dataset[1]

    print(sample_token)
    plt.clf()
    plt.imshow(image.moveaxis(0, 2))
    plt.savefig('tmp/uncertainty_value_loss_img.png')
    uncertainty_value = criterion.get_lattice_uncertainty_values(sample_token)
    print(uncertainty_value)
