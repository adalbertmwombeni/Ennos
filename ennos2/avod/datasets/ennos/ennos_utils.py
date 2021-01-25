import csv
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from wavelab.wavedata.core.voxel_grid_2d import VoxelGrid2D

from avod.core.utils.geometry_utils import transform_points
from avod.datasets.dataset_config import DatasetConfig
from avod.datasets.dataset_utils import DatasetUtils
from avod.datasets.ennos.object_label import EnnosObjectLabel


def read_calibration(calib_dir: str) -> Dict[str, np.ndarray]:
    data_file = open(os.path.join(calib_dir, 'calib_distorted.txt'), 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    lines = [line for line in data_reader]
    data_file.close()
    calibration: Dict[str, np.ndarray] = dict()

    for i in range(len(lines)):
        line = lines[i]
        name = str(line[0])
        line = line[1:]
        line = [float(line[i]) for i in range(len(line))]
        if name in ['k_rgb:', 'r_rgb:', 'k_depth:', 'r_depth:']:
            matrix = np.reshape(np.asarray(line, dtype=np.float32), (3, 3))
            calibration[name[:-1]] = matrix
        elif name in ['t_rgb:', 't_depth:']:
            matrix = np.asarray(line, dtype=np.float32)
            calibration[name[:-1]] = matrix

    return calibration


def get_point_filter(point_cloud: np.ndarray, extents, offset_dist=2.0) -> np.ndarray:
    """
    Creates a point filter using the 3D extents and ground plane

    :param point_cloud: Point cloud in the form [[x,...],[y,...],[z,...]] (3xN)
    :param extents: 3D area in the form [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    :param offset_dist: Removes points above this offset from the ground_plane
    :returns: A binary mask for points within the extents and offset plane.
    """
    # Filter points within certain xyz range
    x_extents = extents[0]
    y_extents = extents[1]
    z_extents = extents[2]

    extents_filter = (point_cloud[0] > x_extents[0]) & (point_cloud[0] < x_extents[1]) & \
                     (point_cloud[1] > y_extents[0]) & (point_cloud[1] < y_extents[1]) & \
                     (point_cloud[2] > z_extents[0]) & (point_cloud[2] < z_extents[1])

    # Create plane filter
    plane_filter = point_cloud[2] > offset_dist

    # Combine the two filters
    point_filter = np.logical_and(extents_filter, plane_filter)

    return point_filter


class EnnosUtils(DatasetUtils):
    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
        self.bev_extents = self.area_extents[[0, 1]]
        self.calibration = read_calibration(self.dataset_config.calib_dir)
        self.inverted_camera_matrix = np.linalg.inv(self.calibration['k_depth'])

    def create_slice_filter(self,
                            point_cloud: np.ndarray,
                            ground_plane: None,
                            ground_offset_dist: float,
                            offset_dist: float):
        """
        Creates a slice filter to take a slice of the point cloud between ground_offset_dist and offset_dist above
        the ground plane.

        :param point_cloud: Point cloud in the shape (3, N)
        :param ground_plane: None. The ground is assumed to be in z=0.
        :param ground_offset_dist: min distance above the ground plane
        :param offset_dist: max distance above the ground
        :returns: A boolean mask if shape (N,) where True indicates the point should be kept and False indicates that
            the point should be removed.
        """
        # Filter points within certain xyz range and offset from ground plane
        offset_filter = get_point_filter(point_cloud, self.area_extents, offset_dist)

        # Filter points within ground_offset_dist (in meters) of the ground plane
        road_filter = get_point_filter(point_cloud, self.area_extents, ground_offset_dist)

        slice_filter = np.logical_xor(offset_filter, road_filter)
        return slice_filter

    def class_str_to_index(self, obj_label: EnnosObjectLabel):
        """
        Converts an object class type string into a integer index.

        :param obj_label: the object label from which to get the object label/class
        :returns: The corresponding integer index for a class type, starting at 1 (0 is reserved for the background
            class). Returns -1 if we don't care about that class type.
        """
        if obj_label.label in self.dataset_config.classes:
            return self.dataset_config.classes.index(obj_label.label) + 1

        raise ValueError('Invalid class string {}, not in {}'.format(obj_label.label, self.dataset_config.classes))

    def create_bev_maps(self, point_cloud: np.ndarray, ground_plane: np.ndarray):
        """
        Calculates the BEV maps.

        :param point_cloud: point cloud
        :param ground_plane: Not used. The ground plane is assumed to be z=0.
        :returns: Dictionary with entries for each type of map (e.g. height, density).
        """
        bev_maps = self.bev_generator.generate_bev('depth', point_cloud, None, self.area_extents, self.voxel_size)
        return bev_maps

    def get_point_cloud(self, depth_map: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Gets the points from the point cloud for a particular image.

        :param depth_map: The depth map from which to compute the point cloud.
        :param image_shape: image dimensions (h, w)
        :returns: The set of points in the shape (3, N).
        """
        # Points in the image coordinate systems
        pts = [(x / depth_map.shape[1], y / depth_map.shape[0], 1.0)
               for y in range(depth_map.shape[0])
               for x in range(depth_map.shape[1])
               if depth_map[y, x] > 100]
        pts = np.asarray(pts).astype(dtype=np.float32)
        z = [depth_map[y, x]
             for y in range(depth_map.shape[0])
             for x in range(depth_map.shape[1])
             if depth_map[y, x] > 100]  # Ignore all points that are 100 mm or less away from the camera
        z = np.asarray(z).reshape((-1, 1)).astype(dtype=np.float32)
        # Convert from millimeters to meters
        z /= 1000

        pts = z * np.dot(self.inverted_camera_matrix, pts.T).T

        # project points to WCS
        world_pts = transform_points(pts, self.calibration['r_depth'], self.calibration['t_depth'], inverse=True)

        return world_pts.transpose()

    def _apply_slice_filter(self, point_cloud: np.ndarray, height_lo=0.1, height_hi=2.2):
        """
        Applies a slice filter to the point cloud.

        :param point_cloud: A point cloud in the shape (3, N)
        :param height_lo: (optional) lower height for slicing
        :param height_hi: (optional) upper height for slicing
        :returns: Points filtered with a slice filter in the shape (N, 3).
        """
        slice_filter = self.create_slice_filter(point_cloud, None, height_lo, height_hi)

        # Transpose point cloud into N x 3 points
        points = np.asarray(point_cloud).T
        filtered_points = points[slice_filter]

        return filtered_points

    def create_sliced_voxel_grid_2d(self, point_cloud: np.ndarray,
                                    ground_plane: Optional[np.ndarray] = None) -> VoxelGrid2D:
        """
        Generates a filtered 2D voxel grid from point cloud data.

        Only points within the area extents are kept. The points furthermore have to be between 0.1 m and 2.4 m
        above the ground plane. (See _apply_slice_filter)

        :param point_cloud: The point cloud from which to generate the voxel grid.
        :param ground_plane: The plane equation parameters of the ground plane. Not used. Ground is assumed to be z=0.
        :returns: A 2D voxel grid from the given image.
        """
        filtered_points = self._apply_slice_filter(point_cloud)

        # Create Voxel Grid
        voxel_grid_2d = VoxelGrid2D(height_dim=self.dataset_config.height_dim)
        voxel_grid_2d.voxelize_2d(filtered_points, self.voxel_size, extents=self.area_extents, create_leaf_layout=True)

        return voxel_grid_2d

    @staticmethod
    def parse_label_file(label_dir: str, classes: List[str]) -> Dict[int, List[EnnosObjectLabel]]:
        with open(os.path.join(label_dir, 'boxes_3d.csv')) as csv_file:
            data_reader = csv.reader(csv_file, delimiter=',')
            lines: List[List[Union[str, int, float]]] = [line for line in data_reader]

            samples: Dict[int, List[EnnosObjectLabel]] = dict()
            for i in range(1, len(lines)):
                label_data = {lines[0][j]: lines[i][j] for j in range(len(lines[i]))}
                if label_data['label'] in classes:
                    sample = EnnosObjectLabel(label_data['label'])
                    sample.location = (float(label_data['cx']), float(label_data['cy']), float(label_data['cz']))
                    sample.sizes_3d = (float(label_data['dx']), float(label_data['dy']), float(label_data['dz']))
                    sample.rotation = (float(label_data['phi_x']),
                                       float(label_data['phi_y']),
                                       float(label_data['phi_z']))
                    sample.truncation = float(label_data['trunc_rgb'])
                    sample.occlusion = float(label_data['occ_rgb'])
                    frame = int(label_data['frame'])
                    if frame not in samples:
                        samples[frame] = [sample]
                    else:
                        samples[frame].append(sample)
        return samples

    def read_labels(self, split: Optional[str] = None) -> List[List[EnnosObjectLabel]]:
        # Read all labels from file
        samples = EnnosUtils.parse_label_file(self.dataset_config.label_dir, self.dataset_config.classes)
        # Filter samples based on which frames are included in the specified split
        if split is None:
            split = self.dataset_config.data_split
        sample_list = [int(sample) for sample in self.load_sample_names(split)]
        filtered_samples = [samples[idx] for idx in sample_list]
        return filtered_samples
