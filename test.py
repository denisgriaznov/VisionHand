from vision_hand_grasp_v0 import VisionHandGraspEnv
from stable_baselines3 import PPO
import cv2

model = PPO.load("ppo_vision_hand_grasp")

env = VisionHandGraspEnv(obs_mode='coordinates')
obs, info = env.reset()
for _ in range(500):
    action, _state = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if _ % 5 == 0:
        cv2.imshow("image", env.render())
        cv2.waitKey(1)

    if done or truncated:
        obs, info = env.reset()