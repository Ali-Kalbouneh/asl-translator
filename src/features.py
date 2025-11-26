import numpy as np


def normalize_landmarks(landmarks):
    
    points = np.array(landmarks, dtype=np.float32)

    wrist = points[0]
    points -= wrist

    scale = np.linalg.norm(points[9])

    if scale < 1e-6:
        scale = 1.0
    
    points = points / scale

    return points.flatten
