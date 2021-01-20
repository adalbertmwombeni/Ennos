# This is like geometry_utils
import numpy as np

def transform_points(pts: np.ndarray,
                     r: np.ndarray,
                     t: np.ndarray,
                     inverse: bool = False) -> np.ndarray:

    if inverse:
        pts_proj = np.dot(r.transpose(), (pts-t).transpose()).transpose()
    else:
        pts_proj = np.dot(r, pts.transpose()).transpose() +t
    return pts_proj