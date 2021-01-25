from abc import *
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from wavelab.wavedata.core.voxel_grid_2d import VoxelGrid2D

from avod import root_dir
from avod.datasets.bev_generators.bev_slices import BevSlices
from avod.datasets.dataset_config import DatasetConfig


class DatasetUtils(ABC):
    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        scene_config = dataset_config.scene_config
        self.area_extents = scene_config.area_extents
        self.bev_extents = []
        self.voxel_size = scene_config.voxel_size
        self.anchor_strides = np.reshape(np.asarray(scene_config.anchor_strides, dtype=np.float32), (-1, 2))
        self.bev_generator = BevSlices(scene_config.bev_generator_slice_config, dataset_config.height_dim, self)

        # Check that depth maps folder exists
        if self.dataset_config.depth_dir is not None and not os.path.exists(self.dataset_config.depth_dir):
            raise FileNotFoundError('Could not find depth maps.')

    def load_sample_names(self, data_split: str) -> np.ndarray:
        """
        Load the sample names listed in this dataset's set file (e.g. train.txt, validation.txt)
        :param data_split: override the sample list to load (e.g. for clustering)
        :return A list of sample names (file names) read from the .txt file corresponding to the data split
        """
        set_file = os.path.join(self.dataset_config.dataset_dir, data_split + '.txt')
        with open(set_file, 'r') as f:
            sample_names = f.read().splitlines()

        return np.array(sample_names)

    def get_cluster_file_path(self, cls: str, num_clusters: int) -> str:
        """
        Returns a unique file path for a text file based on the dataset name, split, object class, and number of
        clusters. The file path will look like this: data/<dataset_name>/<class>_<n_clusters>.

        :param cls: The object class.
        :param num_clusters: The number of clusters for the class.
        :returns: A unique file path to text file.
        """
        file_path = os.path.join(root_dir(), 'data/label_clusters', self.dataset_config.name,
                                 '{}_{}.txt'.format(cls, num_clusters))
        return file_path

    @staticmethod
    def read_clusters_from_file(cluster_file_path: str, num_clusters: int)\
            -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        """
        Reads cluster information from a text file.

        :param cluster_file_path: The path to the file with the cluster information.
        :param num_clusters: number of clusters
        :returns: cluster centers and standard deviations.
        """
        if os.path.isfile(cluster_file_path):
            data = np.loadtxt(cluster_file_path)
            clusters = np.array(data[0:num_clusters])
            std_devs = np.array(data[num_clusters:])
            return clusters, std_devs

        return None, None

    def read_all_clusters(self):
        classes = self.dataset_config.classes
        all_clusters = [[] for _ in range(len(classes))]
        all_std_devs = [[] for _ in range(len(classes))]

        for class_idx in range(len(classes)):
            num_clusters = self.dataset_config.num_clusters
            cluster_file_path = self.get_cluster_file_path(classes[class_idx], num_clusters[class_idx])
            clusters, std_devs = self.read_clusters_from_file(cluster_file_path, num_clusters[class_idx])

            if clusters is not None:
                all_clusters[class_idx].extend(np.asarray(clusters))
                all_std_devs[class_idx].extend(np.asarray(std_devs))

        return all_clusters, all_std_devs

    @abstractmethod
    def create_slice_filter(self,
                            point_cloud: np.ndarray,
                            ground_plane: Optional[np.ndarray],
                            ground_offset_dist: float,
                            offset_dist: float):
        pass

    @abstractmethod
    def class_str_to_index(self, label):
        """
        Converts an object class type string into a integer index.

        :param label: the object label from which to get the object label/class
        :returns: The corresponding integer index for a class type, starting at 1 (0 is reserved for the background
            class). Returns -1 if we don't care about that class type.
        """
        pass

    @abstractmethod
    def create_bev_maps(self, point_cloud: np.ndarray, ground_plane: Optional[np.ndarray])\
            -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
        pass

    @abstractmethod
    def get_point_cloud(self, depth_map: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        pass

    @abstractmethod
    def create_sliced_voxel_grid_2d(self, point_cloud: np.ndarray,
                                    ground_plane: Optional[np.ndarray] = None) -> VoxelGrid2D:
        pass

    @abstractmethod
    def read_labels(self, split: Optional[str] = None) -> List:
        pass
