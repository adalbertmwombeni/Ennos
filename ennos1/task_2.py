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
print("point_cloud: ", point_cloud)


