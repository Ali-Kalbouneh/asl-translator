import numpy as np


def normalize_landmarks(landmarks):
    
    points = np.array(landmarks, dtype=np.float32)

    # Use wrist as base
    wrist = points[0]
    points -= wrist

    # Use distance between writs to upper hand as hand size
    scale = np.linalg.norm(points[9])
 
    # If scale is too small, default to 1
    if scale < 1e-6:
        scale = 1.0
    
    # Normalize
    points = points / scale

    return points.flatten()
