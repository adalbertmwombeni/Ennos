import abc
from typing import Dict

import numpy as np


class BevGenerator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_bev(self, **params) -> Dict:
        """
        Generates the bird's eye view (BEV) maps.

        :param params: additional keyword arguments for specific implementations of BevGenerator.
        :returns: A dictionary with entries for a list of height maps and one density map.
        """
        pass

    @staticmethod
    def _create_density_map(num_divisions: np.ndarray, voxel_indices_2d: np.ndarray,
                            num_pts_per_voxel: np.ndarray, norm_value: np.float64) -> np.ndarray:
        # Create empty density map
        density_map = np.zeros((num_divisions[0], num_divisions[1], num_divisions[2]), dtype=np.float32)
        density_map = density_map.squeeze()

        # Only update pixels where voxels have num_pts values
        # Density is calculated as min(1.0, log(N+1)/log(x))
        # x=64 for stereo, x=16 for lidar, x=64 for depth
        density_map[voxel_indices_2d[:, 0], voxel_indices_2d[:, 1]] = \
            np.minimum(1.0, np.log(num_pts_per_voxel + 1) / norm_value)

        density_map = np.flip(density_map.transpose(), axis=0)

        return density_map
