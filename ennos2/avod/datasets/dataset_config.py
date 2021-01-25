import os.path
from typing import List

from avod.datasets.scene_config import SceneConfig


class DatasetConfig:
    def __init__(self,
                 name: str, dataset_dir: str, data_split: str, data_split_dir: str, has_labels: bool,
                 filter_samples: bool, classes: List[str], num_clusters: List[int],
                 scene_config: SceneConfig, height_dim: int):
        self.name = name
        self.data_split = data_split
        self.dataset_dir = os.path.expanduser(dataset_dir)

        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError('Dataset path does not exist: {}'.format(self.dataset_dir))

        # Get possible data splits from txt files in dataset folder
        possible_splits = list(dict.fromkeys([os.path.splitext(file)[0] for file in os.listdir(self.dataset_dir)
                                              if os.path.isdir(os.path.join(self.dataset_dir, file)) or
                                              (file.endswith('.txt') and 'readme' not in file)]))

        if self.data_split not in possible_splits:
            raise ValueError('Invalid data split: {}, possible_splits: {}'.format(self.data_split, possible_splits))

        self.data_split_dir = os.path.join(self.dataset_dir, data_split_dir)
        if not os.path.isdir(self.data_split_dir):
            raise ValueError('Invalid data split dir: {}, possible dirs'.format(data_split_dir))

        self.filter_samples = filter_samples
        self.has_labels = has_labels
        self.classes = classes
        self.num_clusters = num_clusters
        self.height_dim = height_dim
        self.scene_config = scene_config

        # Setup Directories
        if self.name[:5] == 'ennos':
            self.image_dir = os.path.join(self.data_split_dir, 'images')
            self.calib_dir = os.path.join(self.data_split_dir, 'calibration')
            self.depth_dir = os.path.join(self.data_split_dir, 'depth')
            self.label_dir = os.path.join(self.data_split_dir, 'labels')
        else:
            raise ValueError('Unknown dataset: ' + self.name)
