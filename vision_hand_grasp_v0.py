import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
import os
from utils import *


class VisionHandGraspEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, episode_len=500, render_mode='RGB', **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        Box(0, 255, (210, 160, 3), np.uint8)
        observation_space = Dict(dict(camera=Box(0, 255, (240, 320), np.uint8),
                                      space=Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)))

        MujocoEnv.__init__(
            self,
            os.path.abspath("mujoco_models/scene.xml"),
            5,
            observation_space=observation_space,
            render_mode=render_mode
        )
        self.step_number = 0
        self.episode_len = episode_len

        self.depth_renderer = mujoco.Renderer(self.model)
        self.depth_renderer.enable_depth_rendering()
        self.segmentation_renderer = mujoco.Renderer(self.model)
        self.segmentation_renderer.enable_segmentation_rendering()
        self.rgb_renderer= mujoco.Renderer(self.model)

        self.depth_camera = np.zeros((240, 320))
        self.segmentation_camera = np.zeros((240, 320))
        self.rgb_camera = np.zeros((240, 320))

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        self.step_number += 1

        self.depth_renderer.update_scene(self.data, camera='hand_camera')
        self.segmentation_renderer.update_scene(self.data, camera='hand_camera')
        self.rgb_renderer.update_scene(self.data, camera='hand_camera')

        self.depth_camera = self.depth_renderer.render()
        self.segmentation_camera = self.segmentation_renderer.render()
        self.rgb_camera = self.rgb_renderer.render()

        obs = self._get_obs()
        # done = bool(not np.isfinite(obs).all() or (obs[2] < 0))
        done = False
        truncated = self.step_number > self.episode_len
        return obs, reward, done, truncated, {}

    def reset_model(self):
        self.step_number = 0
        return self._get_obs()

    def _get_obs(self):

        obs = {'camera': self.depth_camera,
               'space': np.zeros(10)}

        return obs

    def render(self):
        depth_pixels = get_depth_pixels(self.depth_camera)
        segmentation_pixels = get_segmentation_pixels(self.segmentation_camera)
        side_pixels = cv2.resize(self.mujoco_renderer.render(render_mode='rgb_array', camera_name='side_camera'), (720, 720))

        depth_pixels = np.stack((depth_pixels,)*3, axis=-1)
        segmentation_pixels = np.stack((segmentation_pixels,) * 3, axis=-1)
        render_pixels = np.concatenate((depth_pixels, segmentation_pixels, self.rgb_camera), axis=0)
        render_pixels = np.concatenate((render_pixels, side_pixels), axis=1)
        return render_pixels
