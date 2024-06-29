import numpy as np
import cv2


def get_depth_pixels(depth):
    depth -= depth.min()
    mean = depth[depth <= 1].mean()
    if mean == 0.0:
        mean = 1
    depth /= 2 * mean
    pixels = 255 * np.clip(depth, 0, 1)
    pixels = cv2.convertScaleAbs(pixels)
    return pixels


def get_segmentation_pixels(seg):
    geom_ids = seg[:, :, 0]
    geom_ids = geom_ids.astype(np.float64) + 1
    geom_ids = geom_ids / geom_ids.max()
    pixels = 255 * geom_ids
    pixels = cv2.convertScaleAbs(pixels)
    return pixels


def get_joints_state(data):
    joint_names = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz", "rh_WRJ1", "rh_WRJ2",
                   "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
                   "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
                   "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
                   "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
                   "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1"]
    hand_pos = list()
    hand_vel = list()
    for joint_name in joint_names:
        hand_pos.append(data.joint(joint_name).qpos[0])
        hand_vel.append(data.joint(joint_name).qvel[0])

    pos = np.concatenate((np.array(hand_pos), np.array(data.joint("target").qpos)), axis=0)
    vel = np.concatenate((np.array(hand_vel), np.array(data.joint("target").qvel)), axis=0)
    return np.concatenate((pos, vel), axis=0)
