from typing import Callable, Dict, Optional, Union

from PIL import Image
import numpy as np
from avod.datasets.dataset import Dataset
from avod.datasets.dataset_config import DatasetConfig
from avod.datasets.ennos.ennos_utils import *
from avod.core.anchor_generator import EnnosAnchorGenerator
from avod.core.utils.geometry_utils import compute_scene_coordinate_system, project_labels, to_projection_matrix
####import torch

"""
from avod.core import anchor_filter
from avod.core.anchor_generator import EnnosAnchorGenerator
from avod.core.box_formats import box_3d_encoder
from avod.core.utils.geometry_utils import compute_scene_coordinate_system, project_labels, to_projection_matrix
from avod.datasets.dataset import Dataset
from avod.datasets.dataset_config import DatasetConfig
from avod.datasets.ennos.ennos_utils import *
from avod.datasets.sample import Sample
"""

class EnnosDataset(Dataset):
    def __init__(self, dataset_config: DatasetConfig, transform: Optional[Callable[[Dict], Dict]] = None):
        super().__init__(dataset_config,
                         EnnosUtils(dataset_config),
                         EnnosAnchorGenerator(dataset_config.scene_config.area_extents),
                         transform)
        self._set_up_classes_name()
        self.sample_list = self.dataset_utils.load_sample_names(self.config.data_split)
        # Load objects labels that match the dataset classes
        self.object_labels: List[List[EnnosObjectLabel]] = self.dataset_utils.read_labels()\
            if self.config.has_labels else None

        all_anchors_box3d = self.generate_anchors(ground_plane=None)
        ###self.all_anchors = box_3d_encoder.box_3d_to_anchor(all_anchors_box3d, self.config.height_dim)

        calibration = read_calibration(self.config.calib_dir)
        ###scs = compute_scene_coordinate_system(calibration)
        ###self.scene_rotation, self.scene_rotation_matrix, self.scene_translation = scs[0], scs[1], scs[2]

        if self.object_labels is not None:
            # Move labels for each frame from WCS to SCS
            self.object_labels = [project_labels(labels_of_frame,
                                                 self.scene_rotation_matrix,
                                                 self.scene_translation,
                                                 self.scene_rotation) for labels_of_frame in self.object_labels]

        combined_rotation = np.matmul(calibration['r_rgb'], self.scene_rotation_matrix.transpose())
        combined_translation = calibration['t_rgb'] - np.matmul(calibration['r_rgb'],
                                                                np.matmul(self.scene_rotation_matrix.transpose(),
                                                                          self.scene_translation))

        self.projection_matrix = to_projection_matrix(calibration['k_rgb'], combined_rotation, combined_translation)

    def _set_up_classes_name(self):
        # Unique identifier for multiple classes
        if self.num_classes > 1:
            raise NotImplementedError('New unique identifier for multiple classes required')
        else:
            self.classes_name = self.classes[0]

    def __len__(self):
        return self.sample_list.shape[0]

    ####def __getitem__(self, item: Union[int, torch.Tensor]) -> Dict[str, np.ndarray]:
    def __getitem__(self, item: Union[int]) -> Dict[str, np.ndarray]:
        ###if torch.is_tensor(item):
            ###item = item.tolist()

        sample_name = self.sample_list[item]
        #sample_name = 'C:/Users/Mwomada/Desktop/ennos/object/training/depth/0000001'


        # Read RGB image from file
        image_file_path = self.get_rgb_image_path(sample_name)
        pil_image_input = Image.open(image_file_path)
        image_shape = (pil_image_input.height, pil_image_input.width)
        image_input = np.asarray(pil_image_input)

        # Get point cloud from depth image
        depth_map = np.asarray(Image.open(os.path.join(self.config.depth_dir, sample_name + '.png')))
        point_cloud = self.dataset_utils.get_point_cloud(depth_map, image_shape)
        print("point_cloud: ", point_cloud)

    def test(self): # def test(self, image_file_path):
        image_file_path = ""
        pil_image_input = Image.open(image_file_path)
        image_shape = (pil_image_input.height, pil_image_input.width)
        image_input = np.asarray(pil_image_input)
        # get point cloud from depth image
        depth_map = np.asarray(Image.open('C:/Users/Mwomada/Desktop/ennos/object/training/depth/0000001.png'))
        point_cloud = self.dataset_utils.get_point_cloud(depth_map, image_shape)
        print("=============================")
        print("point_cloud: ", point_cloud.T)
        print("type: ", type(point_cloud))
        print("Hi")

"""        
        # Move points from WCS to SCS
        point_cloud = transform_points(point_cloud.T, self.scene_rotation_matrix, self.scene_translation).transpose()

        # Create voxel grid from point cloud
        voxel_grid_2d = self.dataset_utils.create_sliced_voxel_grid_2d(point_cloud)
        # Select anchors that span non-empty voxels
        empty_filter = anchor_filter.get_empty_anchor_filter_2d(self.all_anchors, voxel_grid_2d,
                                                                height_dim=2, density_threshold=1)
        anchors = self.all_anchors[empty_filter]
        p_matrix = self.projection_matrix

        # Create BEV maps
        bev_images = self.dataset_utils.create_bev_maps(point_cloud, ground_plane=None)
        height_maps = bev_images.get('height_maps')
        density_map = bev_images.get('density_map')
        bev_input = np.stack((*height_maps, density_map), axis=0)

        # Create sample as a dictionary of all relevant data
        sample_dict = {
            Sample.IMAGE_INPUT: image_input,
            Sample.BEV_INPUT: bev_input,

            Sample.CAMERA_P_MATRIX: p_matrix,
            Sample.BEV_EXTENTS: self.dataset_utils.bev_extents,

            #Sample.SCENE_ROTATION_ANGLE: self.scene_rotation,
            #Sample.SCENE_ROTATION: self.scene_rotation_matrix,
            #Sample.SCENE_TRANSLATION: self.scene_translation,

            Sample.ANCHORS: anchors,

            Sample.IMAGE_FILE_PATH: image_file_path
        }

        # Apply transformations to sample (if any)
        if self.transform is not None:
            sample_dict = self.transform(sample_dict)

        # Reorder image channels to C,H,W order and convert to float values in range [0.0, 1.0]
        image_input = np.transpose(sample_dict[Sample.IMAGE_INPUT], axes=(2, 0, 1))
        image_input = np.asarray(image_input, dtype=np.float32)
        # Per image standardization: zero mean, unit variance
        image_input = (image_input - np.mean(image_input)) / np.std(image_input)
        sample_dict[Sample.IMAGE_INPUT] = image_input

        return sample_dict
"""
