import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class ShadowVisionEnv(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 60,
    }

    def __init__(self):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            "humanoid.xml",
            5,
            observation_space=observation_space
        )
