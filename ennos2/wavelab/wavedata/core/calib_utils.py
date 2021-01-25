import csv
import os
from typing import List, Tuple, Union

import numpy as np
#import torch


class FrameCalibrationData:
    """ Frame Calibration Holder
        3x4    p0-p3      Camera P matrix. Contains extrinsic and intrinsic parameters.
        3x3    r0_rect    Rectification matrix, required to transform points from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from Velodyne to cam coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne.
    """
    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.tr_velodyne_to_cam = []


def read_calibration(calib_dir: str, img_idx: int) -> FrameCalibrationData:
    """
    Reads in Calibration file from Kitti Dataset.

    :param calib_dir: Directory of the calibration files.
    :param img_idx: Index of the image.
    :returns: Full calibration data for a frame.
    """
    frame_calibration_info = FrameCalibrationData()

    data_file = open(calib_dir + "/%06d.txt" % img_idx, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all: List[np.ndarray] = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(np.asarray(p, dtype=np.float32), (3, 4))
        p_all.append(p)

    frame_calibration_info.p0 = p_all[0]
    frame_calibration_info.p1 = p_all[1]
    frame_calibration_info.p2 = p_all[2]
    frame_calibration_info.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calibration_info.r0_rect = np.reshape(np.asarray(tr_rect, dtype=np.float32), (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calibration_info.tr_velodyne_to_cam = np.reshape(np.asarray(tr_v2c, dtype=np.float32), (3, 4))

    return frame_calibration_info


def krt_from_p(p: np.ndarray, fsign: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Factorize the projection matrix P as P=K*[R;t] and enforce the sign of the focal length to be fsign.

    :param p: 3x4 camera matrix
    :param fsign: Sign of the focal length.
    :returns: k: 3x3 Intrinsic calibration matrix,
              r: 3x3 Extrinsic rotation matrix.
              t: 1x3 Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t

"""
def project_to_image(points_3d: Union[np.ndarray, torch.Tensor], p: Union[np.ndarray, torch.Tensor])\
        -> Union[np.ndarray, torch.Tensor]:    
    
    ###Projects a 3D point cloud to 2D points for plotting.

    ###:param points_3d: 3D point cloud (3, N)
    ###:param p: Camera matrix (3, 4)
    ###:returns: The projected 2D points in image coordinates in the shape (2, N).
    
    if isinstance(points_3d, torch.Tensor):
        ones = torch.ones((1, points_3d.shape[1]), device=points_3d.device)
        points_3d_h = torch.cat([points_3d, ones], dim=0)
        points_2d = torch.matmul(p, points_3d_h)
        pts_2d = points_2d[0:2, :] / points_2d[2, :]
    else:
        pts_2d = np.dot(p, np.append(points_3d, np.ones((1, points_3d.shape[1]), dtype=np.float32), axis=0))

        pts_2d[0:2, :] = pts_2d[0:2, :] / pts_2d[2, :]
        pts_2d = np.delete(pts_2d, 2, axis=0)
    return pts_2d
"""

def read_lidar(velo_dir: str, img_idx: int) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], List]:
    """
    Reads in PointCloud from Kitti Dataset.

    :param velo_dir: Directory of the velodyne files.
    :param img_idx: Index of the image.
    :returns: x: Contains the x coordinates of the point cloud.
              y: Contains the y coordinates of the point cloud.
              z: Contains the z coordinates of the point cloud.
              i: Contains the intensity values of the point cloud.
              []: if file is not found
    """
    velo_dir = velo_dir + "/%06d.bin" % img_idx

    if os.path.exists(velo_dir):
        with open(velo_dir, 'rb') as fid:
            data_array = np.fromfile(fid, np.single)

        xyzi = data_array.reshape(-1, 4)

        x = xyzi[:, 0]
        y = xyzi[:, 1]
        z = xyzi[:, 2]
        i = xyzi[:, 3]

        return x, y, z, i
    else:
        return []


def lidar_to_cam_frame(xyz_lidar: np.ndarray, frame_calib: FrameCalibrationData) -> np.ndarray:
    """
    Transforms the point clouds to the camera 0 frame.

    :param xyz_lidar: Contains the x,y,z coordinates of the LIDAR point cloud
    :param frame_calib: Contains calibration information for a given frame
    :returns: The xyz coordinates of the transformed point cloud.
    """
    # Pad the r0_rect matrix to a 4x4
    r0_rect_mat = frame_calib.r0_rect
    r0_rect_mat = np.pad(r0_rect_mat, ((0, 1), (0, 1)), 'constant', constant_values=0)
    r0_rect_mat[3, 3] = 1

    # Pad the tr_vel_to_cam matrix to a 4x4
    tf_mat = frame_calib.tr_velodyne_to_cam
    tf_mat = np.pad(tf_mat, ((0, 1), (0, 0)), 'constant', constant_values=0)
    tf_mat[3, 3] = 1

    # Pad the point cloud with 1's for the transformation matrix multiplication
    one_pad = np.ones(xyz_lidar.shape[0], dtype=np.float32).reshape(-1, 1)
    xyz_lidar = np.append(xyz_lidar, one_pad, axis=1)

    # p_cam = P2 * R0_rect * Tr_velo_to_cam * p_velo
    rectified = np.dot(r0_rect_mat, tf_mat)
    ret_xyz = np.dot(rectified, xyz_lidar.T)

    # Change to N x 3 array for consistency.
    return ret_xyz[0:3].T
