from vision_hand_grasp_v0 import VisionHandGraspEnv
from stable_baselines3 import PPO
import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="Path towards model xml file",
        required=True
    )
    parser.add_argument(
        "--obs_mode",
        type=str,
        help="Mode of observation",
        required=True
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    path = args.path
    obs_mode = args.obs_mode
    env = VisionHandGraspEnv(obs_mode='obs_mode')

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_vision_hand_grasp_coordinates")
