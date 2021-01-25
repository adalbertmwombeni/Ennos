from typing import List

import numpy as np


class BevGeneratorSliceConfig:
    def __init__(self,
                 height_lo: float,
                 height_hi: float,
                 num_slices: int):
        # slices config
        self.height_lo = height_lo
        self.height_hi = height_hi
        self.num_slices = num_slices


class SceneConfig:
    def __init__(self,
                 area_extents: List[float],
                 voxel_size: float,
                 anchor_strides: List[float],
                 bev_generator_slice_config: BevGeneratorSliceConfig):
        self.area_extents = np.reshape(np.asarray(area_extents, dtype=np.float32), (3, 2))
        self.voxel_size = voxel_size
        self.anchor_strides = anchor_strides
        self.bev_generator_slice_config = bev_generator_slice_config
