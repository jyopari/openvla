"""Rotation conversion utilities."""

import numpy as np
from scipy.spatial.transform import Rotation


def axangle2euler(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to euler angles (roll, pitch, yaw).
    
    Args:
        axis_angle: (3,) array where the direction represents the rotation axis
                   and the magnitude represents the rotation angle in radians
    
    Returns:
        (3,) array of euler angles [roll, pitch, yaw] in radians
    """
    # Get angle (magnitude) and axis (direction)
    angle = np.linalg.norm(axis_angle)
    if np.isclose(angle, 0.0):
        return np.zeros(3)
    
    axis = axis_angle / angle
    
    # Convert to rotation matrix using scipy
    rot = Rotation.from_rotvec(axis_angle)
    
    # Convert to euler angles
    euler = rot.as_euler('xyz', degrees=False)
    return euler


def euler2axangle(roll: float, pitch: float, yaw: float) -> tuple[np.ndarray, float]:
    """Convert euler angles to axis-angle representation.
    
    Args:
        roll: rotation around x-axis in radians
        pitch: rotation around y-axis in radians 
        yaw: rotation around z-axis in radians
    
    Returns:
        tuple of:
        - (3,) unit vector representing rotation axis
        - float angle in radians
    """
    # Convert to rotation matrix using scipy
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    
    # Get axis-angle representation
    rotvec = rot.as_rotvec()
    angle = np.linalg.norm(rotvec)
    
    if np.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 1.0]), 0.0
    
    axis = rotvec / angle
    return axis, angle
