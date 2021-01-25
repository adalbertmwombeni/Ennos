
class EnnosObjectLabel:
    """
    Object label class for the ENNOS dataset
    """
    def __init__(self, label: str):
        self.label = label  # Object label (class) of the object, e.g. 'person'
        self.location = (0., 0., 0.)  # location in 3D space (x, y, z) in meters
        self.sizes_3d = (0., 0., 0.)  # object size in 3D (width (dx), length (dy), height (dz)) in meters
        self.rotation = (0., 0., 0.)  # rotation around x, y and z axis
        self.truncation = 0.
        self.occlusion = 0.
