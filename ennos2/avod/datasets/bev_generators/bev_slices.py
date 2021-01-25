from typing import Dict, List, Optional, Union
import numpy as np

from wavelab.wavedata.core.voxel_grid_2d import VoxelGrid2D

from avod.datasets.bev_generators.bev_generator import BevGenerator
from avod.datasets.scene_config import BevGeneratorSliceConfig


class BevSlices(BevGenerator):

    NORM_VALUES = {'lidar': np.log(16), 'depth': np.log(64)}

    def __init__(self, config: BevGeneratorSliceConfig, height_dim: int, dataset_utils):
        """
        BEV maps created using slices of the point cloud.

        :param config: bev_generator config
        :param height_dim: The index of the dimension that represent height.
        """
        self.height_lo = config.height_lo
        self.height_hi = config.height_hi
        self.num_slices = config.num_slices
        self.utils = dataset_utils

        self.height_per_division = (self.height_hi - self.height_lo) / self.num_slices
        self.height_dim = height_dim

    def generate_bev(self,
                     source: str,
                     point_cloud: np.ndarray,
                     ground_plane: Optional[np.ndarray],
                     area_extents: np.ndarray,
                     voxel_size: float) -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
        """
        Generates the BEV maps dictionary.

        One height map is created for each slice of the point cloud. One density map is created for the whole point
        cloud.

        :param source: point cloud source
        :param point_cloud: point cloud (3, N)
        :param ground_plane: ground plane coefficients
        :param area_extents: 3D area extents [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        :param voxel_size: voxel size in m
        :returns: A dictionary of BEV maps (height_maps: list of height maps, density_map: density map).
        """
        all_points = np.transpose(point_cloud)
        height_maps = []
        ground_dims = [i for i in range(3) if i != self.height_dim]

        for slice_idx in range(self.num_slices):
            height_lo = self.height_lo + slice_idx * self.height_per_division
            height_hi = height_lo + self.height_per_division

            slice_filter = self.utils.create_slice_filter(point_cloud, ground_plane, height_lo, height_hi)

            # Apply slice filter
            slice_points = all_points[slice_filter]

            if len(slice_points) > 1:
                # Create Voxel Grid 2D
                voxel_grid_2d = VoxelGrid2D(self.height_dim)
                voxel_grid_2d.voxelize_2d(slice_points, voxel_size, area_extents, ground_plane,
                                          create_leaf_layout=False)

                # Remove height dimension (all 0)
                voxel_indices = voxel_grid_2d.voxel_indices[:, ground_dims]

                # Create empty BEV images
                height_map = np.zeros((voxel_grid_2d.num_divisions[ground_dims[0]],
                                       voxel_grid_2d.num_divisions[ground_dims[1]]),
                                      dtype=np.float32)

                # Only update pixels where voxels have max height values, and normalize by height of slices
                voxel_grid_2d.heights = voxel_grid_2d.heights - height_lo
                height_map[voxel_indices[:, 0], voxel_indices[:, 1]] = \
                    np.asarray(voxel_grid_2d.heights) / self.height_per_division
            else:
                print('Warning: Insufficient points during BEV slice generation. Creating empty height map.')
                min_voxel_coord, max_voxel_coord = VoxelGrid2D.compute_min_max_coordinates(voxel_size, area_extents,
                                                                                           self.height_dim)
                num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)
                height_map = np.zeros((num_divisions[ground_dims[0]], num_divisions[ground_dims[1]]), dtype=np.float32)

            height_maps.append(height_map)

        # Rotate height maps 90 degrees (transpose and flip is faster than np.rot90)
        height_maps_out = [np.flip(height_maps[map_idx].transpose(), axis=0) for map_idx in range(len(height_maps))]
        density_slice_filter = self.utils.create_slice_filter(point_cloud, ground_plane, self.height_lo, self.height_hi)
        density_points = all_points[density_slice_filter]

        # TODO this currently does not handle the case of an empty scene, i.e. no points
        # Create Voxel Grid 2D
        density_voxel_grid_2d = VoxelGrid2D(self.height_dim)
        density_voxel_grid_2d.voxelize_2d(density_points, voxel_size, area_extents, ground_plane,
                                          create_leaf_layout=False)

        # Generate density map
        density_voxel_indices_2d = density_voxel_grid_2d.voxel_indices[:, ground_dims]

        density_map = self._create_density_map(
            num_divisions=density_voxel_grid_2d.num_divisions,
            voxel_indices_2d=density_voxel_indices_2d,
            num_pts_per_voxel=density_voxel_grid_2d.num_pts_in_voxel,
            norm_value=self.NORM_VALUES[source])

        bev_maps = {'height_maps': height_maps_out, 'density_map': density_map}

        return bev_maps
