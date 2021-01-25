from typing import List, Tuple
import numpy as np


def dist_to_plane(plane: Tuple[float, float, float, float], points: List[List[float]]) -> float:
    """
    Calculates the signed distance from a 3D plane to each point in a list of points.

    :param plane: Coefficients of the plane equation (a, b, c, d)
    :param points: List of points
    :returns: Signed distance of each point to the plane
    """
    a, b, c, d = plane

    points = np.array(points)
    x: float = points[:, 0]
    y: float = points[:, 1]
    z: float = points[:, 2]

    return (a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)
