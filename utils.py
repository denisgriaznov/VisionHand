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
