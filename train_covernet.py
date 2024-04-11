import os
import torch
import pickle
import argparse

import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from trajectory_safety.datasets import get_dataloader_nuscenes, get_dataloader_carla
from trajectory_safety.datasets.utils import rotate_points
from trajectory_safety.models.covernet.backbone import ResNetBackbone
from trajectory_safety.models.covernet.covernet import CoverNet
from trajectory_safety.loss.constant_lattice import ConstantLatticeLoss



def train_model(dataset='carla', lattice_set='epsilon_4'):
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    if dataset == 'nuscenes':
        train_loader, test_loader = get_dataloader_nuscenes()
    elif dataset == 'nuscenes_mini':
        train_loader, test_loader = get_dataloader_nuscenes(split='mini')
    elif dataset == 'carla':
        train_loader, test_loader = get_dataloader_carla()
    else:
        raise NotImplementedError(f'Dataset {dataset} not supported.')

    backbone = ResNetBackbone('resnet50')

    with open(f'trajectory_safety/models/covernet/lattice/{lattice_set}.pkl', 'rb') as f:
        lattice = pickle.load(f)
        # In our data the vehicle is heading to the x axis, but the lattice is assuming the vehicle
        # is heading toward y axis, so we need to rotate these -90 degrees.
        lattice = rotate_points(np.array(lattice), -90)

    print('len(lattice):', len(lattice))
    # print(lattice.shape)
    # plt.clf()
    # for l in lattice:
    #     plt.plot(l[:, 0], l[:, 1])
    # plt.savefig('tmp/lattice.png')
    # plt.clf()
    # for i, data in enumerate(train_loader):
    #     img, state_vector, y = data
    #     for t in y:
    #         plt.plot(t[:, 0], t[:, 1])
    # plt.savefig('tmp/trajectory.png')

    # Note that the value of num_modes depends on the size of the lattice used for CoverNet.
    lattice = np.array(lattice)
    model = CoverNet(backbone, num_modes=lattice.shape[0], input_shape=(3,224,224), state_vector_shape=3)
    model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = ConstantLatticeLoss(lattice=lattice)
    #optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-5)

    save_dir = f'saves/{dataset}/resnet50_{lattice_set}/'
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=f'./runs/{dataset}/resnet50_{lattice_set}/')

    # Training loop
    num_epochs = 1000
    for epoch in range(0, num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for i, data in enumerate(train_loader):
            image, state_vector, trajectory = data

            image = image.to(device)
            state_vector = state_vector.to(device)
            trajectory = trajectory.to(device)
            optimizer.zero_grad()

            logits = model(image, state_vector)

            # print(f'image min max: {image.min()} {image.max()}')
            # print(f'state_vector min max: {state_vector.min()} {state_vector.max()}')
            # print(f'logits min max: {logits.min()} {logits.max()}')

            loss, acc = criterion(logits, trajectory)
            # print('loss', loss)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar('lr', lr, epoch)
        print(f'Epoch {epoch} training loss: {train_loss} acc: {train_acc} lr: {lr}')

        scheduler.step(train_loss)

        with torch.no_grad():
            test_loss = 0.0
            test_acc = 0.0
            for i, data in enumerate(test_loader):
                image, state_vector, trajectory = data
                image = image.to(device)
                state_vector = state_vector.to(device)
                trajectory = trajectory.to(device)
                logits = model(image, state_vector)

                loss, acc = criterion(logits, trajectory)
                test_loss += loss.item()
                test_acc += acc.item()
            test_loss = test_loss / len(test_loader)
            test_acc = test_acc / len(test_loader)
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
            print(f'Epoch {epoch} testing loss: {test_loss} acc: {test_acc}')

        if (epoch+1) % 2 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, f'optimizer_{epoch}.pth'))


def main():
    parser = argparse.ArgumentParser(
        description='Covernet trainer'
    )

    parser.add_argument('-d', '--dataset', type=str, choices=['nuscenes', 'nuscenes_mini', 'carla'])
    parser.add_argument('-g', '--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in args.gpus])

    train_model(args.dataset)

if __name__ == "__main__":
    main()
