from copy import copy
from typing import Dict, List, Tuple

import numpy as np

import wavelab.wavedata.core.calib_utils as calib_utils
from avod.datasets.ennos.object_label import EnnosObjectLabel


def to_projection_matrix(camera_matrix: np.ndarray, rotation_matrix: np.ndarray, translation: np.ndarray):
    rt = np.concatenate((rotation_matrix, translation.reshape((-1, 1))), axis=1)
    projection_matrix = np.matmul(camera_matrix, rt)
    return projection_matrix


def transform_points(pts: np.ndarray,
                     r: np.ndarray,
                     t: np.ndarray,
                     inverse: bool = False) -> np.ndarray:
    """
    Transforms a given set of 3D points into another coordinate system by translating and rotating them.

    :param pts: the 3D points to be projected of shape (N_points, 3) with 3 being [X, Y, Z]
    :param r: the rotation matrix of shape (3, 3)
    :param t: the translation vector of shape (1, 3)
    :param inverse: normally the projection is done by first rotating and then translating
                    but if we want to go into the other direction, we need to translate into the opposite direction
                    and after that rotate with the transposed rotation
    :returns: The projected points of shape (N_pts, 3).
    """
    if inverse:
        pts_proj = np.dot(r.transpose(), (pts-t).transpose()).transpose()
    else:
        pts_proj = np.dot(r, pts.transpose()).transpose() + t
    return pts_proj


def project_labels(objects: List[EnnosObjectLabel],
                   rotation: np.ndarray,
                   translation: np.ndarray,
                   rotation_angle: float,
                   inverse: bool = False):
    """Transforms object labels from one coordinate system to another."""
    objects_scs = copy(objects)
    for obj in objects_scs:
        location = np.asarray(obj.location)
        location = transform_points(location, rotation, translation, inverse)
        obj.location = (float(location[0]), float(location[1]), float(location[2]))
        if inverse:
            new_rotation = obj.rotation[2] - rotation_angle
        else:
            new_rotation = obj.rotation[2] + rotation_angle
        if new_rotation <= -np.pi:
            new_rotation += 2 * np.pi
        elif new_rotation > np.pi:
            new_rotation -= 2 * np.pi
        obj.rotation = (obj.rotation[0], obj.rotation[1], float(new_rotation))
    return objects_scs


def compute_scene_coordinate_system(calibration: Dict[str, np.ndarray]) -> Tuple[float, np.ndarray, np.ndarray]:
    r_ccs = calibration['r_rgb']
    t_ccs = calibration['t_rgb']
    k = calibration['k_rgb']

    # Get camera position in WCS
    t_wcs = np.matmul(r_ccs.transpose(), -calibration['t_rgb'])
    point_lower_center = np.dot(np.linalg.inv(k), np.asarray([0.5, 1.0, 1.0], dtype=np.float32))
    dir_vec = transform_points(point_lower_center, r_ccs, t_ccs, inverse=True) - t_wcs
    pt_floor = t_wcs - (t_wcs[2] / dir_vec[2]) * dir_vec

    north_vector: np.ndarray = np.matmul(r_ccs.transpose(),
                                         np.asarray([[0.0], [0.0], [1.0]], dtype=np.float32)).squeeze()
    north_vector[2] = 0.0
    north_vector /= np.linalg.norm(north_vector)
    rotation_value = float(np.arccos(np.dot(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), north_vector)))
    if north_vector[0] < 0:
        rotation_value *= -1  # Preserve direction of rotation
    r_scs = np.asarray([[north_vector[1], -north_vector[0], 0.0],
                        [north_vector[0], north_vector[1], 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float32)

    # Transform from WCS to SCS
    t_scs = -np.matmul(r_scs, pt_floor)

    return rotation_value, r_scs, t_scs


def project_box3d_to_image(box_3d: np.ndarray,
                           projection_matrix: np.ndarray,
                           image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Projects a box_3d to the camera image using a projection matrix. It assumes the ENNOS variant of a box_3d where the
    center of rotation is the center of the box.
    """
    rot = np.array([[+np.cos(box_3d[6]), -np.sin(box_3d[6]), 0],
                    [+np.sin(box_3d[6]), +np.cos(box_3d[6]), 0],
                    [0, 0, 1]])

    l = box_3d[3]
    w = box_3d[4]
    h = box_3d[5]

    # 3D BB corners
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    z_corners = np.array([-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

    corners_3d[0:3, :] += box_3d[0:3, np.newaxis]

    points_2d = calib_utils.project_to_image(corners_3d, projection_matrix)
    points_2d = points_2d * np.asarray(image_shape).reshape(-1, 1)

    return points_2d


def project_box3d_to_bev(box_3d: np.ndarray, bev_extents: np.ndarray, bev_shape: Tuple[int, int]) -> np.ndarray:
    """
    Projects a bounding box in the box_3d format into a bird's eye view image.

    :param box_3d: The bounding box in box_3d format that shall be projected.
    :param bev_extents: The extents (in m) of the bird's eye view in the ground plane.
    :param bev_shape: The shape of the BEV image as a tuple of width, height.
    :returns: The corners as a percentage of the map size, in the format N x [x1, y1, x2, y2].
        Origin is the bottom left corner (i.e. the corner on the left side closest to the camera)
    """
    half_dim_x = box_3d[3] / 2.0
    half_dim_y = box_3d[4] / 2.0
    # 2D corners (bottom left, top right)
    corners = [np.asarray([-half_dim_x, -half_dim_y], dtype=np.float32),
               np.asarray([+half_dim_x, -half_dim_y], dtype=np.float32),
               np.asarray([+half_dim_x, +half_dim_y], dtype=np.float32),
               np.asarray([-half_dim_x, +half_dim_y], dtype=np.float32)]

    # Create rotation matrix
    rot_angle = box_3d[6]
    rot_mat = np.asarray([[np.cos(rot_angle), -np.sin(rot_angle)],
                          [np.sin(rot_angle), np.cos(rot_angle)]], dtype=np.float32)

    center = box_3d[0:2]

    # Apply rotation and translation
    corners = np.asarray([np.matmul(rot_mat, pt) + center for pt in corners], dtype=np.float32)

    bev_min = np.asarray([bev_extents[0][0], bev_extents[1][0]], dtype=np.float32)
    bev_range = np.asarray([bev_extents[0][1], bev_extents[1][1]], dtype=np.float32) - bev_min

    # Calculate normalized box corners (for ROI pooling)
    bev_box_corners_norm = (corners - bev_min[np.newaxis, :]) / bev_range

    # Flip y coordinates (origin changes from bottom left to top left)
    bev_box_corners_norm[:, 1] = 1.0 - bev_box_corners_norm[:, 1]

    # Convert from original xy into bev xy, origin moves to bottom left of BEV extents
    bev_box_corners = bev_box_corners_norm * np.asarray(bev_shape, dtype=np.float32)

    return bev_box_corners
