from torchvision import transforms
import torch
from torch import nn
import numpy as np
import os
import json
import cv2
from torch.utils.data import Dataset

from .utils import rotate_points

class CarlaTrajectoryPredictionDataset(Dataset):
    def __init__(self, dataset_paths, return_metadata=False, sensor_range=50.0, bev_range=50.0):
        super().__init__()
        self.records = []
        self.return_metadata = return_metadata
        self.sensor_range = sensor_range
        self.bev_range = bev_range
        self.trajectory_length = 12
        self.frame_interval = 10        # Must match the dataset

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for path in dataset_paths:
            self.read_data_folder(path)

    def read_data_folder(self, path):
        local_records = []
        for agent in os.listdir(os.path.join(path, 'agents')):
            agent_folder = os.path.join(path, 'agents', agent)

            agent_location = {}
            for line in open(os.path.join(agent_folder, 'gt_location', 'data.jsonl'), 'r'):
                record = json.loads(line)
                agent_location[record['frame']] = {
                    'location': record['location'],
                    'rotation': record['rotation'],
                }

            vehicle_bbox_records = []
            vehicle_trajectories = {}
            for line in open(os.path.join(agent_folder, 'gt_vehicle_bbox', 'data.jsonl'), 'r'):
                record = json.loads(line)
                vehicle_bbox_records.append(record)
                frame = record['frame']
                for vehicle in record['vehicles']:
                    distance = np.linalg.norm(np.array(agent_location[frame]['location'])-np.array(vehicle['location']))
                    if distance > self.sensor_range:
                        continue
                    if vehicle['id'] not in vehicle_trajectories:
                        vehicle_trajectories[vehicle['id']] = {}
                    vehicle_trajectories[vehicle['id']][frame] = np.array(vehicle['location'])
                    local_records.append({
                        'vehicle_id': vehicle['id'],
                        'frame': frame,
                        'timestamp': record['timestamp'],
                        # 'current_action': vehicle['current_action'],
                        'distance': distance,
                        'agent_location': np.array(agent_location[frame]['location']),
                        'agent_rotation': np.array(agent_location[frame]['rotation']),
                        'vehicle_location': np.array(vehicle['location']),
                        'vehicle_rotation': np.array(vehicle['rotation']),
                        'vehicle_velocity': np.array(vehicle['velocity']),
                        'bev_image_path': os.path.join(agent_folder, 'birds_view_semantic_camera', str(frame)+'.png'),
                        'output_path': os.path.join(agent_folder, 'pred_vehicle_trajectory', 'data.jsonl'),
                    })

        records_to_delete = []
        for idx, record in enumerate(local_records):
            trajectory = []
            for f in range(record['frame'], record['frame'] + (self.trajectory_length * self.frame_interval), self.frame_interval):
                if f in vehicle_trajectories[record['vehicle_id']]:
                    trajectory.append(vehicle_trajectories[record['vehicle_id']][f])
                else:
                    records_to_delete.append(idx)
                    break
            record['trajectory'] = np.array(trajectory)

        local_records = [element for i,element in enumerate(local_records) if i not in records_to_delete]

        self.records.extend(local_records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        agent_image = cv2.imread(record['bev_image_path'])
        agent_image = cv2.cvtColor(agent_image, cv2.COLOR_BGR2RGB)
        vehicle_pitch = record['vehicle_rotation'][2]
        y = record['trajectory']
        vehicle_velocity = rotate_points(record['vehicle_velocity'], -vehicle_pitch)
        state_vector = vehicle_velocity
        width = agent_image.shape[1]
        height = agent_image.shape[0]
        assert agent_image.shape[0] == agent_image.shape[1]
        img_scale =  agent_image.shape[0] / (self.bev_range*2)

        image = agent_image
        # Rotate the image to standard rotation
        agent_pitch = record['agent_rotation'][2]
        # print(agent_pitch)
        mat = cv2.getRotationMatrix2D((width/2, height/2), -agent_pitch, 1.0)
        image = cv2.warpAffine(src=agent_image, M=mat, dsize=(width, height))

        translation = record['vehicle_location'] - record['agent_location']
        # Move the vehicle to the center
        translation_matrix = np.array([
            [1, 0, -translation[1]*img_scale],
            [0, 1, translation[0]*img_scale]
        ], dtype=np.float32)

        image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(width, height))

        # print(translation[0], translation[1])
        # print(translation[0]*img_scale, translation[1]*img_scale)

        # print(vehicle_pitch)
        mat = cv2.getRotationMatrix2D((width/2, height/2), vehicle_pitch, 1.0)
        image = cv2.warpAffine(src=image, M=mat, dsize=(width, height))

        # print(record['vehicle_id'])

        if self.transform is not None:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32)
        state_vector = torch.tensor(state_vector, dtype=torch.float32)
        y = np.array(y)
        y = (y-record['vehicle_location'])[:, [0, 1]] # take relative x and y coordinate

        y = rotate_points(y, -vehicle_pitch)
        y = torch.tensor(y, dtype=torch.float32)

        vehicle_id = record['vehicle_id']

        if self.return_metadata:
            return image, state_vector, y, self.records[index]
        return image, state_vector, y, vehicle_id
