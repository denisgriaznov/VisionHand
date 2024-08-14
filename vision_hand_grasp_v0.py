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

    def __init__(self, episode_len=500, obs_mode='coordinates', path=os.path.abspath("mujoco_models/scene.xml"), **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        # check if obs mode is valid
        valid_obs = {'coordinates', 'vision_rgb', 'vision_depth', 'vision_segmentation'}
        if obs_mode not in valid_obs:
            raise ValueError("obs must be one of %r." % valid_obs)
        else:
            self.obs_mode = obs_mode

        # create observation space depending on the mode
        if self.obs_mode == 'coordinates':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(73,), dtype=np.float64)
        else:
            self.observation_space = Box(0, 255, (240, 320), np.uint8)

        MujocoEnv.__init__(
            self,
            path,
            5,
            observation_space=self.observation_space,
            render_mode='RGB'
        )
        self.step_number = 0
        self.episode_len = episode_len

        self.depth_renderer = mujoco.Renderer(self.model)
        self.depth_renderer.enable_depth_rendering()
        self.segmentation_renderer = mujoco.Renderer(self.model)
        self.segmentation_renderer.enable_segmentation_rendering()
        self.rgb_renderer = mujoco.Renderer(self.model)

        self.depth_camera = np.zeros((240, 320))
        self.segmentation_camera = np.zeros((240, 320))
        self.rgb_camera = np.zeros((240, 320))

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        self.step_number += 1
        self.update_renderers(for_obs=True)

        reward = self.get_reward()
        obs = self.get_obs()
        done = False
        truncated = self.step_number > self.episode_len

        return obs, reward, done, truncated, {}

    def reset_model(self):
        self.step_number = 0
        return self.get_obs()

    def get_obs(self):
        if self.obs_mode == 'coordinates':
            obs = get_joints_state(self.data)
        elif self.obs_mode == 'vision_rgb':
            obs = self.rgb_camera
        elif self.obs_mode == 'vision_depth':
            obs = self.depth_camera
        elif self.obs_mode == 'vision_segmentation':
            obs = self.segmentation_camera
        else:
            return
        return obs

    def get_reward(self):
        r_pick = -np.abs(self.data.body("rh_palm").xpos[2] - self.data.joint("target").qpos[2])
        r_up = -np.abs(0.05 - self.data.joint("target").qpos[2])
        reward = r_pick + r_up * 4
        return reward

    def update_renderers(self, for_obs):
        if self.obs_mode == 'vision_rgb' or not for_obs:
            self.rgb_renderer.update_scene(self.data, camera='hand_camera')
            self.rgb_camera = self.rgb_renderer.render()
        if self.obs_mode == 'vision_depth' or not for_obs:
            self.depth_renderer.update_scene(self.data, camera='hand_camera')
            self.depth_camera = self.depth_renderer.render()
        if self.obs_mode == 'vision_segmentation' or not for_obs:
            self.segmentation_renderer.update_scene(self.data, camera='hand_camera')
            self.segmentation_camera = self.segmentation_renderer.render()

    def render(self):
        self.update_renderers(for_obs=False)
        depth_pixels = get_depth_pixels(self.depth_camera)
        segmentation_pixels = get_segmentation_pixels(self.segmentation_camera)
        side_pixels = cv2.resize(self.mujoco_renderer.render(render_mode='rgb_array', camera_name='side_camera'),
                                 (720, 720))

        depth_pixels = np.stack((depth_pixels,) * 3, axis=-1)
        segmentation_pixels = np.stack((segmentation_pixels,) * 3, axis=-1)
        render_pixels = np.concatenate((depth_pixels, segmentation_pixels, self.rgb_camera), axis=0)
        render_pixels = np.concatenate((render_pixels, side_pixels), axis=1)
        return render_pixels
