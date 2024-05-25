import time
import mujoco.viewer
import cv2
import numpy as np

m = mujoco.MjModel.from_xml_path('mujoco_models/scene.xml')
d = mujoco.MjData(m)

depth_renderer = mujoco.Renderer(m)
depth_renderer.enable_depth_rendering()

segm_renderer = mujoco.Renderer(m)
segm_renderer.enable_segmentation_rendering()


def render_depth():
    depth = depth_renderer.render()
    depth -= depth.min()
    depth /= 2 * depth[depth <= 1].mean()
    pixels = 255 * np.clip(depth, 0, 1)
    pixels = cv2.convertScaleAbs(pixels)
    cv2.imshow('Depth', pixels)
    cv2.waitKey(1);


def render_segmentation():
    seg = segm_renderer.render()
    geom_ids = seg[:, :, 0]
    geom_ids = geom_ids.astype(np.float64) + 1
    geom_ids = geom_ids / geom_ids.max()
    pixels = 255 * geom_ids
    pixels = cv2.convertScaleAbs(pixels)
    print('Camera resolution: ', pixels.shape)
    cv2.imshow('Segmentation', pixels)
    cv2.waitKey(1);


with mujoco.viewer.launch_passive(m, d) as viewer:
    cv2.namedWindow("Depth")
    cv2.namedWindow("Segmentation")

    start = time.time()
    while True:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)
        depth_renderer.update_scene(d, camera='hand_camera')
        segm_renderer.update_scene(d, camera='hand_camera')

        render_depth()
        render_segmentation()

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
