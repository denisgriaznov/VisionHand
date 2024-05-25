import time
import mujoco.viewer
from utils import *

m = mujoco.MjModel.from_xml_path('mujoco_models/scene.xml')
d = mujoco.MjData(m)

depth_renderer = mujoco.Renderer(m)
depth_renderer.enable_depth_rendering()

segm_renderer = mujoco.Renderer(m)
segm_renderer.enable_segmentation_rendering()

cv2.namedWindow("Depth")
cv2.namedWindow("Segmentation")

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while True:
        step_start = time.time()
        mujoco.mj_step(m, d)
        depth_renderer.update_scene(d, camera='hand_camera')
        segm_renderer.update_scene(d, camera='hand_camera')

        cv2.imshow('Depth', get_depth_pixels(depth_renderer.render()))
        cv2.imshow('Segmentation', get_segmentation_pixels(segm_renderer.render()))
        cv2.waitKey(1)
        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
