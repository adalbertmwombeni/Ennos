from typing import Callable, Dict, Optional, Union
import os



import numpy as np
##import torch.utils.data


from avod.core.anchor_generator import AnchorGenerator
from avod.datasets.dataset_config import DatasetConfig
from avod.datasets.dataset_utils import DatasetUtils
# torch.utils.data.Dataset
class Dataset():
    def __init__(self,
                 dataset_config: DatasetConfig,
                 dataset_utils: DatasetUtils,
                 anchor_generator: AnchorGenerator,
                 transform: Optional[Callable[[Dict], Dict]] = None):
        """
        :param dataset_config: The configuration information for the dataset.
        :param anchor_generator: The generator used to create the anchors for this dataset.
        :param transform: Optional transformation object that shall be applied to the
        """
        self.config = dataset_config
        self.classes = list(dataset_config.classes)
        self.num_classes = len(self.classes)
        self.transform = transform

        self._anchor_generator = anchor_generator
        self.dataset_utils = dataset_utils

        # Label Clusters
        self.clusters, self.std_devs = dataset_utils.read_all_clusters()
        self.sample_list = np.empty((1, 1), dtype=np.float32)
    """
    def get_sample_name(self, item: Union[int, torch.Tensor]) -> str:
        if torch.is_tensor(item):
            item = item.tolist()
        return self.sample_list[item]
    """
    def _set_up_classes_name(self):
        # Unique identifier for multiple classes
        if self.num_classes > 1:
            raise NotImplementedError('New unique identifier for multiple classes required')
        else:
            self.classes_name = self.classes[0]

    def get_rgb_image_path(self, sample_name: str):
        return os.path.join(self.config.image_dir, sample_name + '.png')

    def generate_anchors(self, ground_plane: Optional[np.ndarray]) -> np.ndarray:
        all_anchor_boxes_3d = []
        for class_idx in range(self.num_classes):
            # Generate anchors for all classes
            grid_anchor_boxes_3d = self._anchor_generator.generate(
                anchor_3d_sizes=self.clusters[class_idx],
                anchor_stride=self.dataset_utils.anchor_strides[class_idx],
                ground_plane=ground_plane)
            all_anchor_boxes_3d.append(grid_anchor_boxes_3d)
        if self.num_classes > 1:
            all_anchor_boxes_3d = np.asarray(all_anchor_boxes_3d)
            return all_anchor_boxes_3d.reshape((-1, all_anchor_boxes_3d.shape[-1]))
        else:
            return all_anchor_boxes_3d[0]
