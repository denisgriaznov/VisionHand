import time
import numpy as np
import mujoco
import cv2
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    m = mujoco.MjModel.from_xml_path('shadow_hand/scene_right.xml')
    d = mujoco.MjData(m)

    depth_renderer = mujoco.Renderer(m)
    depth_renderer.enable_depth_rendering()

    segm_renderer = mujoco.Renderer(m)
    segm_renderer.enable_segmentation_rendering()

    cv2.namedWindow("Depth")
    cv2.namedWindow("Segmentation")

    mujoco.mj_step(m, d)
    depth_renderer.update_scene(d, camera='hand_camera')
    segm_renderer.update_scene(d, camera='hand_camera')

    #depth
    depth = depth_renderer.render()
    depth -= depth.min()
    depth /= 2 * depth[depth <= 1].mean()
    pixels = 255 * np.clip(depth, 0, 1)
    pixels = cv2.convertScaleAbs(pixels)
    cv2.imshow('Depth', pixels)
    cv2.waitKey(0);

    #segmentation
    seg = segm_renderer.render()
    geom_ids = seg[:, :, 0]
    geom_ids = geom_ids.astype(np.float64) + 1
    geom_ids = geom_ids / geom_ids.max()
    pixels = 255 * geom_ids
    cv2.imshow('Segmentation', pixels)
    cv2.waitKey(0);