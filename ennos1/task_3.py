# this file is dataset_utils
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import csv

def read_calibration(calib_dir: str) -> Dict[str, np.ndarray]:
    data_file = open(os.path.join('C:/Users/Mwomada/Desktop/ennos/object/training/calibration/calib_distorted.txt'), 'r')
    data_reader =csv.reader(data_file, delimiter='')
    lines = [line for line in data_reader]
    data_file.close()
    calibration: Dict[str, np.ndarray] = dict()

    for i in range(len(lines)):
        line = lines[i]
        name = str(line[0])
        line = line[1:]
        line = [float(line[i]) for i in range(len(line))]
        if name in ['k_rgb:', 'r_rgb:', 'k_depth:', 'r_depth:']:
            matrix = np.reshape(np.asarray(line, dtype=np.float32),(3,3))
            calibration[name[:-1]] = matrix
        elif name in ['t_rgb:', 't_depth:']:
            matrix = np.asarray(line, dtype=np.float32)
            calibration[name[:-1]] = matrix
    return calibration


def get_point_cloud(depth_map: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Get the point cloud from a particular image
    :param depth_map: the depthmap from which to compute the point cloud
    :param image_shape: image dimensions(h, w)
    : returns: the set of points in the shape (3, N)
    """
    # Points in the image coordinate systems
    pts = [(x / depth_map.shape[1], y / depth_map.shape[0], 1.0)
           for y in range(depth_map.shape[0])
           for x in range(depth_map.shape[1])
           if depth_map[y,x]> 100]
    pts = np.asarray(pts).astype(dtype=np.float32)
    z = [depth_map[y,x]
         for y in range(depth_map.shape[0])
         for x in range(depth_map.shape[1])
         if depth_map[y,x]> 100] # Ignore all points that are in 100 mm 0r less away from the camera
    z = np.asarray(z).reshape((-1, 1)).astype(dtype=np.float32)
    # Convert from millimeters to meters
    z /= 1000

    #calibration_applied = read_calibration('C:/Users/Mwomada/Desktop/ennos/object/training/calibration/calib_distorted.txt')
    #inverted_camera_matrix = np.linalg.inv(calibration_applied)
    pts = z.T
    """
    pts = z * np.dot(inverted_camera_matrix, pts.T).T
    
    # Project points to WCS
    word_points = transform_points(calibration_applied['r_depth'], calibration_applied['t_depth'], inverse=True)
    """
    return pts
    #return word_points.transpose()