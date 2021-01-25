from typing import List, Tuple, Union
import numpy as np

from . import geometry_utils


class VoxelGrid2D:
    """
    Voxel grids represent occupancy info. The voxelize_2d method projects a point cloud onto a plane, while saving
    height and point density information for each voxel.
    """
    # Class Constants
    VOXEL_EMPTY = -1
    VOXEL_FILLED = 0

    def __init__(self, height_dim: int):
        # Quantization size of the voxel grid
        self.voxel_size = 0.0

        # Voxels at the most negative/positive xyz
        self.min_voxel_coord = np.array([])
        self.max_voxel_coord = np.array([])

        # Size of the voxel grid along each axis
        self.num_divisions = np.array([0, 0, 0])

        self.height_dim = height_dim

        # Points in sorted order, to match the order of the voxels
        self.points = []

        # Indices of filled voxels
        self.voxel_indices = []

        # Max point height in projected voxel
        self.heights = []

        # Number of points corresponding to projected voxel
        self.num_pts_in_voxel = []

        # Full occupancy grid, VOXEL_EMPTY or VOXEL_FILLED
        self.leaf_layout_2d = []

    @staticmethod
    def compute_min_max_coordinates(voxel_size: float, extents: np.ndarray, height_dim: int)\
            -> Tuple[np.ndarray, np.ndarray]:
        # Check provided extents
        extents_transpose = np.array(extents).transpose()
        if extents_transpose.shape != (2, 3):
            raise ValueError("Extents have the wrong shape {}".format(extents.shape))

        # Set voxel grid extents
        min_voxel_coord = np.floor(extents_transpose[0] / voxel_size)
        max_voxel_coord = np.ceil((extents_transpose[1] / voxel_size) - 1)

        min_voxel_coord[height_dim] = 0
        max_voxel_coord[height_dim] = 0

        return min_voxel_coord, max_voxel_coord

    def voxelize_2d(self, pts: np.ndarray, voxel_size: float, extents=None, ground_plane=None, create_leaf_layout=True):
        """
        Voxelizes the point cloud into a 2D voxel grid by projecting it down into a flat plane.

        It stores the maximum point height and the number of points corresponding to the voxel.

        :param pts: Point cloud as N x [x, y, z]
        :param voxel_size: Quantization size for the grid
        :param extents: Optional, specifies the full extents of the point cloud.
                        Used for creating same sized voxel grids.
        :param ground_plane: Plane coefficients (a, b, c, d), xz plane used if not specified
        :param create_leaf_layout: Set this to False to create an empty leaf_layout,
                                   which will save computation time.
        """
        # Check if points are 3D, otherwise early exit
        if pts.shape[1] != 3:
            raise ValueError("Points have the wrong shape: {}".format(pts.shape))

        self.voxel_size = voxel_size

        # Discretize voxel coordinates to given quantization size
        discrete_pts = np.floor(pts / voxel_size).astype(np.int32)

        x_col = discrete_pts[:, 0]
        y_col = discrete_pts[:, 1]
        z_col = discrete_pts[:, 2]
        # Use lexicographical sorting. Sort by ground location, then height.
        if self.height_dim == 1:
            # FIXME: this gives the height of the leftmost, furthest point, not the highest point!
            sort_order = (y_col, z_col, x_col)
        elif self.height_dim == 2:
            # FIXME: this gives the height of the leftmost, closest point, not the highest point!
            sort_order = (z_col, y_col, x_col)
        else:
            raise NotImplementedError('Sorting of voxels during voxel grid generation not implemented for x being the'
                                      ' height dimension.')
        sorted_order = np.lexsort(sort_order)

        # Save original points in sorted order
        self.points = pts[sorted_order]

        # Save discrete points in sorted order
        discrete_pts = discrete_pts[sorted_order]

        # Project all points to a 2D plane
        discrete_pts_2d = discrete_pts.copy()
        discrete_pts_2d[:, self.height_dim] = 0

        # Format the array to c-contiguous array for unique function
        contiguous_array = np.ascontiguousarray(discrete_pts_2d).view(
            np.dtype((np.void, discrete_pts_2d.dtype.itemsize * discrete_pts_2d.shape[1])))

        # The new coordinates are the discretized array with its unique indexes
        _, unique_indices = np.unique(contiguous_array, return_index=True)

        # Sort unique indices to preserve order -> TODO this is redundant given the current sorting of the points
        unique_indices.sort()

        voxel_coords = discrete_pts_2d[unique_indices]

        # Number of points per voxel, last voxel calculated separately
        num_points_in_voxel = np.diff(unique_indices)
        num_points_in_voxel = np.append(num_points_in_voxel, discrete_pts_2d.shape[0] - unique_indices[-1])

        if ground_plane is None:
            # Assume the voxel grid is axis aligned. So just take the value from the dimension representing height.
            height_in_voxel = self.points[unique_indices, self.height_dim]
        else:
            # Ground plane provided: Compute distance of first point in voxel to ground plane.
            height_in_voxel = geometry_utils.dist_to_plane(ground_plane, self.points[unique_indices])

        # Set the height and number of points for each voxel
        self.heights = height_in_voxel
        self.num_pts_in_voxel = num_points_in_voxel

        # Find the minimum and maximum voxel coordinates
        if extents is not None:
            self.min_voxel_coord, self.max_voxel_coord = VoxelGrid2D.compute_min_max_coordinates(voxel_size, extents,
                                                                                                 self.height_dim)
            min_coords = np.amin(voxel_coords, axis=0)
            # Check that points are bounded by new extents
            if (self.min_voxel_coord > min_coords).any():
                raise ValueError("Extents too small. Voxels exists with x smaller than extent range.")
            if (self.max_voxel_coord < np.amax(voxel_coords, axis=0)).any():
                raise ValueError("Extents too small. Voxels exists with x larger than extent range.")
        else:
            # Automatically calculate extents
            self.min_voxel_coord = np.amin(voxel_coords, axis=0)
            self.max_voxel_coord = np.amax(voxel_coords, axis=0)

        # Get the voxel grid dimensions
        self.num_divisions = ((self.max_voxel_coord - self.min_voxel_coord) + 1).astype(np.int32)

        # Bring the min voxel to the origin
        self.voxel_indices = (voxel_coords - self.min_voxel_coord).astype(int)

        if create_leaf_layout:
            # Create Voxel Object with -1 as empty/occluded, 0 as occupied
            self.leaf_layout_2d = self.VOXEL_EMPTY * np.ones(self.num_divisions.astype(int))

            # Fill out the leaf layout
            self.leaf_layout_2d[self.voxel_indices[:, 0], self.voxel_indices[:, 1], self.voxel_indices[:, 2]] =\
                self.VOXEL_FILLED

    def map_to_index(self, map_index: np.ndarray) -> Union[np.ndarray, List]:
        """
        Converts map coordinate values to 1-based discretized grid index coordinate. Note: Any values outside the
        extent of the grid will be forced to be the maximum grid coordinate.
        :param map_index: N x 2 points
        :return N x length(dim) (grid coordinate)
            [] if min_voxel_coord or voxel_size or grid_index or dim is not set
        """
        if self.voxel_size == 0 or len(self.min_voxel_coord) == 0 or len(map_index) == 0:
            return []

        ground_dims = [i for i in range(0, 3) if i != self.height_dim]

        num_divisions_2d = self.num_divisions[ground_dims]
        min_voxel_coord_2d = self.min_voxel_coord[ground_dims]

        # Truncate index (same as np.floor for positive values) and clip to valid voxel index range
        indices = np.int32(map_index / self.voxel_size - min_voxel_coord_2d)
        indices[:, 0] = np.clip(indices[:, 0], 0, num_divisions_2d[0] - 1)
        indices[:, 1] = np.clip(indices[:, 1], 0, num_divisions_2d[1] - 1)

        return indices
