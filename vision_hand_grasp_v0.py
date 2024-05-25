import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
import os


class ShadowVisionEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 60,
    }

    def __init__(self, episode_len=500, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        Box(0, 255, (210, 160, 3), np.uint8)
        observation_space = Dict(dict(camera=Box(0, 255, (240, 320), np.uint8),
                                      space=Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)))

        MujocoEnv.__init__(
            self,
            os.path.abspath("mujoco_models/scene.xml"),
            5,
            observation_space=observation_space
        )
        self.step_number = 0
        self.episode_len = episode_len

        self.depth_renderer = mujoco.Renderer(self.model)
        self.depth_renderer.enable_depth_rendering()
        self.segm_renderer = mujoco.Renderer(self.model)
        self.segm_renderer.enable_segmentation_rendering()

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        self.step_number += 1

        obs = self._get_obs()
        # done = bool(not np.isfinite(obs).all() or (obs[2] < 0))
        done = False
        truncated = self.step_number > self.episode_len
        return obs, reward, done, truncated, {}

    def reset_model(self):
        self.step_number = 0
        return self._get_obs()

    def _get_obs(self):
        self.depth_renderer.update_scene(self.data, camera='hand_camera')
        obs = {'camera': self.depth_renderer.render(),
               'space': np.zeros(10)}

        return obs
