
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer

import cv2
import math
import time



class VispyRenderer:

    def __init__(self, width=640, height=360, camera_matrix=None, ply_model_path="bunny.ply"):

        self.ren = renderer.create_renderer(
            width, height, renderer_type="vispy", mode="depth"
        )
    
        self.obj_id = 0
        self.ren.add_object(self.obj_id, ply_model_path)

        # random default values
        self.fx = 497.8242492675781
        self.fy = 498.3829345703125
        self.cx = width//2
        self.cy = height//2

        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

        if camera_matrix is not None:
            self.K = camera_matrix
            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]



    def render_mask(self, R, t):
        # test_t = np.array((0,0,15 + 5*math.sin(3*time.time())))
        # test_R = np.array(((1,0,0),(0,math.cos(time.time()),-math.sin(time.time())),(0,math.sin(time.time()),math.cos(time.time()))))

        depth_gt = self.ren.render_object(self.obj_id, R, t, self.fx, self.fy, self.cx, self.cy)["depth"]
        dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, self.K)

        # Mask of the full object silhouette.
        mask = dist_gt > 0

        return mask


    def render_mask_uint8(self, R, t):
        mask = self.render_mask(R, t)
        uint8_image = mask.astype(np.uint8) * 255
        return uint8_image
