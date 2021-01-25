import numpy as np
from PIL import Image
from task_3 import *

# read RGB image from file
pil_image_input = Image.open('C:/Users/Mwomada/Desktop/ennos/object/training/image/0000001.bmp')
image_shape = (pil_image_input.height, pil_image_input.width)
image_input = np.asarray(pil_image_input)
print("rgb_image: ", image_input)

# get point cloud from depth image
depth_map = np.asarray(Image.open('C:/Users/Mwomada/Desktop/ennos/object/training/depth/0000001.png'))
point_cloud = get_point_cloud(depth_map, image_shape)
print("=============================")
print("point_cloud: ", point_cloud.T)
print("type: ", type(point_cloud))

img_size=(180, 240)
image=np.zeros(img_size)
points = point_cloud.T
for point in points:
    pass
    #each point = [x,y,z,v]
    #image[tuple(point[0:2])] += point[3]
from PIL import Image
import numpy as np

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def birds_eye_point_cloud(points,
                          side_range=(-10, 10),
                          fwd_range=(-10,10),
                          res=0.1,
                          min_height = -2.73,
                          max_height = 1.27,
                          saveto=None):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """
    x_tof = points[:, 0]
    y_tof = points[:, 1]
    z_tof = points[:, 2]
    # r_tof = points[:, 3]  # Reflectance

    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in TOF coordinates
    ff = np.logical_and((x_tof > fwd_range[0]), (x_tof < fwd_range[1]))
    ss = np.logical_and((y_tof > -side_range[1]), (y_tof < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff,ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_tof[indices]/res).astype(np.int32) # x axis is -y in TOF
    y_img = (x_tof[indices]/res).astype(np.int32)  # y axis is -x in TOF
                                                     # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0]/res))
    y_img -= int(np.floor(fwd_range[0]/res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a = z_tof[indices],
                           a_min=min_height,
                           a_max=max_height)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values  = scale_to_255(pixel_values, min=min_height, max=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[-y_img, x_img] = pixel_values # -y because images start from top left

    # Convert from numpy array to a PIL image
    im = Image.fromarray(im)

    # SAVE THE IMAGE
    if saveto is not None:
        im.save(saveto)
    else:
        im.show()

birds_eye_point_cloud(points,
                          side_range=(-10, 10),
                          fwd_range=(-10,10),
                          res=0.1,
                          min_height = -2.73,
                          max_height = 1.27,
                          saveto=None)

print("points: ",points)
print("type: ", type(points))