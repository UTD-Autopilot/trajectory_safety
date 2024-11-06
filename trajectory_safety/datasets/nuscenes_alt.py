import numpy as np
from typing import List, Dict, Tuple, Any
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from pyquaternion import Quaternion
from shapely import affinity
import cv2
from nuscenes.prediction.input_representation.utils import convert_to_pixel_coords
from nuscenes import NuScenes, NuScenesExplorer
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from torchvision import transforms
from torch.utils.data import Dataset
import torch

from .utils import rotate_points

def correct_yaw(yaw: float) -> float:
    """
    nuScenes maps were flipped over the y-axis, so we need to
    add pi to the angle needed to rotate the heading.
    :param yaw: Yaw angle to rotate the image.
    :return: Yaw after correction.
    """
    if yaw <= 0:
        yaw = -np.pi - yaw
    else:
        yaw = np.pi - yaw

    return yaw

def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def pixels_to_box_corners(row_pixel: int,
                          column_pixel: int,
                          length_in_pixels: float,
                          width_in_pixels: float,
                          yaw_in_radians: float) -> np.ndarray:
    """
    Computes four corners of 2d bounding box for agent.
    The coordinates of the box are in pixels.
    :param row_pixel: Row pixel of the agent.
    :param column_pixel: Column pixel of the agent.
    :param length_in_pixels: Length of the agent.
    :param width_in_pixels: Width of the agent.
    :param yaw_in_radians: Yaw of the agent (global coordinates).
    :return: numpy array representing the four corners of the agent.
    """

    # cv2 has the convention where they flip rows and columns so it matches
    # the convention of x and y on a coordinate plane
    # Also, a positive angle is a clockwise rotation as opposed to counterclockwise
    # so that is why we negate the rotation angle
    coord_tuple = ((column_pixel, row_pixel), (length_in_pixels, width_in_pixels), -yaw_in_radians * 180 / np.pi)

    box = cv2.boxPoints(coord_tuple)

    return box

def get_track_box(annotation: Dict[str, Any],
                  center_coordinates: Tuple[float, float],
                  center_pixels: Tuple[float, float],
                  resolution: float = 0.1) -> np.ndarray:
    """
    Get four corners of bounding box for agent in pixels.
    :param annotation: The annotation record of the agent.
    :param center_coordinates: (x, y) coordinates in global frame
        of the center of the image.
    :param center_pixels: (row_index, column_index) location of the center
        of the image in pixel coordinates.
    :param resolution: Resolution pixels/meter of the image.
    """

    assert resolution > 0

    location = annotation['translation'][:2]
    yaw_in_radians = quaternion_yaw(Quaternion(annotation['rotation']))

    row_pixel, column_pixel = convert_to_pixel_coords(location,
                                                      center_coordinates,
                                                      center_pixels, resolution)

    width = annotation['size'][0] / resolution
    length = annotation['size'][1] / resolution

    # Width and length are switched here so that we can draw them along the x-axis as
    # opposed to the y. This makes rotation easier.
    return pixels_to_box_corners(row_pixel, column_pixel, length, width, yaw_in_radians)


def get_rotation_matrix(image_shape: Tuple[int, int, int], yaw_in_radians: float) -> np.ndarray:
    """
    Gets a rotation matrix to rotate a three channel image so that
    yaw_in_radians points along the positive y-axis.
    :param image_shape: (Length, width, n_channels).
    :param yaw_in_radians: Angle to rotate the image by.
    :return: rotation matrix represented as np.ndarray.
    :return: The rotation matrix.
    """

    rotation_in_degrees = angle_of_rotation(yaw_in_radians) * 180 / np.pi

    return cv2.getRotationMatrix2D((image_shape[1] / 2, image_shape[0] / 2), rotation_in_degrees, 1)

# Utility functions from nuScenes

def _is_polygon_record_in_patch(map_api,
                                token: str,
                                layer_name: str,
                                box_coords: Tuple[float, float, float, float],
                                mode: str = 'intersect') -> bool:
    """
    Query whether a particular polygon record is in a rectangular patch.
    :param layer_name: The layer name of the record.
    :param token: The record token.
    :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
    :param mode: "intersect" means it will return True if the geometric object intersects the patch and False
    otherwise, "within" will return True if the geometric object is within the patch and False otherwise.
    :return: Boolean value on whether a particular polygon record intersects or is within a particular patch.
    """
    if layer_name not in map_api.lookup_polygon_layers:
        raise ValueError('{} is not a polygonal layer'.format(layer_name))

    x_min, y_min, x_max, y_max = box_coords
    record = map_api.get(layer_name, token)
    rectangular_patch = box(x_min, y_min, x_max, y_max)

    if layer_name == 'drivable_area':
        polygons = [map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
        geom = MultiPolygon(polygons)
    else:
        geom = map_api.extract_polygon(record['polygon_token'])

    if mode == 'intersect':
        return geom.intersects(rectangular_patch)
    elif mode == 'within':
        return geom.within(rectangular_patch)

def _is_line_record_in_patch(map_api,
                                token: str,
                                layer_name: str,
                                box_coords: Tuple[float, float, float, float],
                                mode: str = 'intersect') -> bool:
    """
    Query whether a particular line record is in a rectangular patch.
    :param layer_name: The layer name of the record.
    :param token: The record token.
    :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
    :param mode: "intersect" means it will return True if the geometric object intersects the patch and False
    otherwise, "within" will return True if the geometric object is within the patch and False otherwise.
    :return: Boolean value on whether a particular line  record intersects or is within a particular patch.
    """
    if layer_name not in map_api.non_geometric_line_layers:
        raise ValueError("{} is not a line layer".format(layer_name))

    # Retrieve nodes of this line.
    record = map_api.get(layer_name, token)
    node_recs = [map_api.get('node', node_token) for node_token in record['node_tokens']]
    node_coords = [[node['x'], node['y']] for node in node_recs]
    node_coords = np.array(node_coords)

    # A few lines in Queenstown have zero nodes. In this case we return False.
    if len(node_coords) == 0:
        return False

    # Check that nodes fall inside the path.
    x_min, y_min, x_max, y_max = box_coords
    cond_x = np.logical_and(node_coords[:, 0] < x_max, node_coords[:, 0] > x_min)
    cond_y = np.logical_and(node_coords[:, 1] < y_max, node_coords[:, 1] > y_min)
    cond = np.logical_and(cond_x, cond_y)
    if mode == 'intersect':
        return np.any(cond)
    elif mode == 'within':
        return np.all(cond)

def is_record_in_patch(
    map_api,
    layer_name: str,
    token: str,
    box_coords: Tuple[float, float, float, float],
    mode: str = 'intersect') -> bool:
    """
    Query whether a particular record is in a rectangular patch.
    :param layer_name: The layer name of the record.
    :param token: The record token.
    :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
    :param mode: "intersect" means it will return True if the geometric object intersects the patch and False
    otherwise, "within" will return True if the geometric object is within the patch and False otherwise.
    :return: Boolean value on whether a particular record intersects or is within a particular patch.
    """
    if mode not in ['intersect', 'within']:
        raise ValueError("Mode {} is not valid, choice=('intersect', 'within')".format(mode))

    if layer_name in map_api.lookup_polygon_layers:
        return _is_polygon_record_in_patch(map_api, token, layer_name, box_coords, mode)
    elif layer_name in  map_api.non_geometric_line_layers:
        return _is_line_record_in_patch(map_api, token, layer_name, box_coords,  mode)
    else:
        raise ValueError("{} is not a valid layer".format(layer_name))

def get_records_in_patch(
    map_api,
    box_coords: Tuple[float, float, float, float],
    layer_names: List[str] = None,
    mode: str = 'intersect') -> Dict[str, List[str]]:
    """
    Get all the record token that intersects or within a particular rectangular patch.
    :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
    :param layer_names: Names of the layers that we want to retrieve in a particular patch.
        By default will always look for all non geometric layers.
    :param mode: "intersect" will return all non geometric records that intersects the patch,
        "within" will return all non geometric records that are within the patch.
    :return: Dictionary of layer_name - tokens pairs.
    """
    if mode not in ['intersect', 'within']:
        raise ValueError("Mode {} is not valid, choice=('intersect', 'within')".format(mode))

    records_in_patch = dict()
    for layer_name in layer_names:
        layer_records = []
        for record in getattr(map_api, layer_name):
            token = record['token']
            if is_record_in_patch(map_api, layer_name, token, box_coords, mode):
                layer_records.append(token)

        records_in_patch.update({layer_name: layer_records})

    return records_in_patch

def render_bev(nusc, nusc_map, sample_token):
    bev_size = np.array([100, 100]) # 100m x 100m centered by the ego vehicle
    image_size = np.array([200, 200]) # bev image resolution

    #sample_token = scene['first_sample_token']
    sample_record = nusc.get('sample', sample_token)
    scene_record = nusc.get('scene', sample_record['scene_token'])
    log_record = nusc.get('log', scene_record['log_token'])
    log_location = log_record['location'] # City name

    lidar_token = sample_record['data']['LIDAR_TOP']
    lidar_record = nusc.get('sample_data', lidar_token)

    ego_pose_token = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])['ego_pose_token']
    ego_pose_record = nusc.get('ego_pose', ego_pose_token)
    ego_location = ego_pose_record['translation']

    patch_radius = 50
    box_coords = [
        ego_location[0] - patch_radius,
        ego_location[1] - patch_radius,
        ego_location[0] + patch_radius,
        ego_location[1] + patch_radius,
    ]

    records_in_patch = get_records_in_patch(nusc_map, box_coords, ['drivable_area', 'traffic_light'], 'intersect')

    # Plot the bev
    scale = bev_size[0] / image_size[0] # scale ratio

    img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    patch_box = (ego_location[0], ego_location[1], bev_size[0], bev_size[1])

    ego_yaw = Quaternion(ego_pose_record['rotation']).yaw_pitch_roll[0]
    yaw_corrected = correct_yaw(ego_yaw)
    angle_in_degrees = angle_of_rotation(yaw_corrected) * 180 / np.pi

    patch_angle = angle_in_degrees  # Default orientation where North is up

    layer_names = ['drivable_area', 'road_block', 'lane_divider', 'road_divider']

    map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, image_size).astype(bool)
    # shape of map_mask: (layer, y, x).
    # Coordinate of map_mask is based on the map image, so (x, y) in nuscene coordinate becomes (-y, x).
    # We'll need to convert the coordinate back.
    map_mask = map_mask[:, ::-1, :]

    img[map_mask[0]] = [0, 255, 255]
    img[map_mask[1]] = [0, 0, 255]
    img[map_mask[2]] = [0, 255, 0]
    img[map_mask[3]] = [0, 255, 0]
    
    objects_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    for token in records_in_patch['traffic_light']:
        record = nusc_map.get('traffic_light', token)
        coord = np.array([record['pose']['tx'], record['pose']['ty'], record['pose']['tz']])
        bev_coord =  coord - np.array(ego_pose_record['translation'])
        #bev_coord = np.dot(Quaternion(ego_pose_record['rotation']).rotation_matrix.T, bev_coord)
        annotation = {
            'translation': coord,
            'rotation': Quaternion(),
            'size': [0.5, 0.5, 1.0],
        }
        track_box = get_track_box(annotation, (ego_location[0], ego_location[1]), image_size/2, scale)
        cv2.fillPoly(objects_img, pts=[np.int0(track_box)], color=(255, 255, 0))
    
    vehicle_bboxes = []

    for annotation_token in sample_record['anns']:
        annotation_record = nusc.get('sample_annotation', annotation_token)
        instance_record = nusc.get('instance', annotation_record['instance_token'])
        category_name = annotation_record['category_name']
        if category_name.startswith('vehicle'):

            track_box = get_track_box(annotation_record, (ego_location[0], ego_location[1]), image_size/2, scale)
            cv2.fillPoly(objects_img, pts=[np.int0(track_box)], color=(255, 0, 0))
    
    center_agent_yaw = quaternion_yaw(Quaternion(ego_pose_record['rotation']))
    rotation_mat = get_rotation_matrix(image_size, center_agent_yaw)
    objects_img = cv2.warpAffine(objects_img, rotation_mat, (image_size[1], image_size[0]), flags=cv2.INTER_NEAREST)
    #plt.imshow(objects_img)

    # render objects on top of map
    mask = np.repeat(np.expand_dims((objects_img != 0).any(-1), -1), 3, axis = 2)
    np.putmask(img, mask, objects_img)

    return img

class NuScenesAltTrajectoryPredictionDataset(Dataset):
    def __init__(self, dataset_path, split='train'):
        super().__init__()
        # split must be one of ['mini_train', 'mini_val', 'train', 'train_val', 'val']

        self.dataset_path = dataset_path
        if split.startswith('mini_'):
            self.nusc = NuScenes('v1.0-mini', dataroot=self.dataset_path, verbose=False)
        else:
            self.nusc = NuScenes('v1.0-trainval', dataroot=self.dataset_path, verbose=False)

        self.helper = PredictHelper(self.nusc)
        self.sample_ids = get_prediction_challenge_split(split, dataroot=self.dataset_path)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.nusc_maps = {}
    
    def __getitem__(self, index):
        instance_token, sample_token = self.sample_ids[index].split("_")
        sample_record = self.nusc.get('sample', sample_token)

        scene_record = self.nusc.get('scene', sample_record['scene_token'])
        log_record = self.nusc.get('log', scene_record['log_token'])
        log_location = log_record['location']

        # Load the map for the location
        if log_location not in self.nusc_maps:
            self.nusc_maps[log_location] = NuScenesMap(dataroot=self.dataset_path, map_name=log_location)
        nusc_map = self.nusc_maps[log_location]

        bev_image = render_bev(self.nusc, nusc_map, sample_token)

        future_xy_local = self.helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
        agent_state_vector = torch.Tensor([[self.helper.get_velocity_for_agent(instance_token, sample_token),
                                    self.helper.get_acceleration_for_agent(instance_token, sample_token),
                                    self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)]]).reshape(-1)
        agent_state_vector = torch.nan_to_num(agent_state_vector, 0)

        if self.transform is not None:
            img = self.transform(bev_image)

        future_xy_local = rotate_points(future_xy_local, -90)
        return img, agent_state_vector, future_xy_local, sample_token
