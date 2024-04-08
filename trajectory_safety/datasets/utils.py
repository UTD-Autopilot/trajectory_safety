import numpy as np

def rotate_points(points, angel):
    rotated = np.zeros(points.shape, dtype=points.dtype)
    rotated[...,0] = np.cos(np.deg2rad(angel)) * points[...,0] - np.sin(np.deg2rad(angel)) * points[...,1]
    rotated[...,1] = np.sin(np.deg2rad(angel)) * points[...,0] + np.cos(np.deg2rad(angel)) * points[...,1]
    return rotated
