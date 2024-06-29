import numpy as np
import cv2
from vision_hand_grasp_v0 import VisionHandGraspEnv


env = VisionHandGraspEnv()
obs, info = env.reset()
for _ in range(5):
    action = np.random.rand(26) / 100
    obs, reward, done, truncated, info = env.step(action)
    #print(obs)
    #print(obs)
    if _ % 5 == 0:
        cv2.imshow("image", env.render())
        cv2.waitKey(1)

    if done or truncated:
        obs, info = env.reset()