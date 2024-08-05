from vision_hand_grasp_v0 import VisionHandGraspEnv
from stable_baselines3 import PPO

env = VisionHandGraspEnv(obs_mode='coordinates')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_vision_hand_grasp_coordinates")