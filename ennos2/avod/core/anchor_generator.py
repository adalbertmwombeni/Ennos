""" Generates 3D anchors, placing them on the ground plane

The job of the anchor generator is to create (or load) a collection of bounding boxes to be used as anchors.

Generated anchors are assumed to match some convolutional grid or list of grid shapes. For example, we might want to
generate anchors matching an 8x8 feature map and a 4x4 feature map. If we place 3 anchors per grid location on the
first feature map and 6 anchors per grid location on the second feature map, then 3*8*8 + 6*4*4 = 288 anchors are
generated in total.

To support fully convolutional settings, feature map shapes are passed dynamically at generation time. The number of
anchors to place at each location is static --- implementations of AnchorGenerator must always be able return the number
of anchors that it uses per location for each feature map.
"""
from abc import *
from typing import List, Union
import numpy as np
#import torch



class AnchorGenerator(ABC):
    def __init__(self, area_extents: np.ndarray):
        """
        Instantiates an anchor generator which creates anchors in the specified area.

        :param area_extents: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        """
        self.area_extents = area_extents
    """    
        
    @abstractmethod
    def generate(self,
                 anchor_3d_sizes: List[np.ndarray],
                 anchor_stride: np.ndarray,
                 **kwargs) -> Union[np.ndarray, torch.Tensor]:
        pass
    """



class EnnosAnchorGenerator(AnchorGenerator):
    def __init__(self, area_extents: np.ndarray):
        super().__init__(area_extents)

    def generate(self,
                 anchor_3d_sizes: List[np.ndarray],
                 anchor_stride: np.ndarray,
                 **kwargs) -> np.ndarray:
        """
        Generates 3D anchors in a grid in the provided 3d area and places them on the z=0 plane.

        Tiles anchors over the area extents by using mesh grids to generate combinations of (x, y, z), (l, w, h) and ry.

        :param anchor_3d_sizes: list of 3d anchor sizes N x (l, w, h)
        :param anchor_stride: stride lengths (x_stride, z_stride)
        :returns: A list of 3D anchors in box_3d format N x [x, y, z, l, w, h, ry].
        """
        # Convert sizes list to numpy array
        anchor_3d_sizes = np.asarray(anchor_3d_sizes, dtype=np.float32)
        anchor_rotations = np.asarray([0, np.pi / 2.0], dtype=np.float32)

        x_start = int(self.area_extents[0][0] / anchor_stride[0]) * anchor_stride[0]
        x_end = int(self.area_extents[0][1] / anchor_stride[0]) * anchor_stride[0] + anchor_stride[0] / 2.0
        x_centers = np.array(np.arange(x_start, x_end, step=anchor_stride[0]), dtype=np.float32)

        y_start = int(self.area_extents[1][0] / anchor_stride[1]) * anchor_stride[1]
        y_end = int(self.area_extents[1][1] / anchor_stride[1]) * anchor_stride[1] + anchor_stride[1] / 2.0
        y_centers = np.array(np.arange(y_start, y_end, step=anchor_stride[1]), dtype=np.float32)

        # Use ranges for substitution
        size_indices = np.arange(0, len(anchor_3d_sizes))
        rotation_indices = np.arange(0, len(anchor_rotations))

        # Generate matrix for substitution e.g. for two sizes and two rotations
        # [[x0, y0, 0, 0], [x0, y0, 0, 1], [x0, y0, 1, 0], [x0, y0, 1, 1],
        #  [x1, y0, 0, 0], [x1, y0, 0, 1], [x1, y0, 1, 0], [x1, y0, 1, 1], ...]
        before_sub = np.stack(np.meshgrid(x_centers, y_centers, size_indices, rotation_indices), axis=4).reshape(-1, 4)

        # Create empty matrix to return
        all_anchor_boxes_3d = np.zeros((before_sub.shape[0], 7), dtype=np.float32)

        # Fill in x and y.
        all_anchor_boxes_3d[:, 0:2] = before_sub[:, 0:2]

        # Fill in shapes (sizes)
        all_anchor_boxes_3d[:, 3:6] = anchor_3d_sizes[np.asarray(before_sub[:, 2], np.int32)]

        # Set z to half height of anchor.
        all_anchor_boxes_3d[:, 2] = all_anchor_boxes_3d[:, 5] / 2.0

        # Fill in rotations
        all_anchor_boxes_3d[:, 6] = anchor_rotations[np.asarray(before_sub[:, 3], np.int32)]

        return all_anchor_boxes_3d
