import numpy as np

def normalize_landmarks(landmarks_list):
    """
    Normalizes a list of 21 landmarks (dicts with x, y, z).
    Returns a flattened 63D numpy array.
    """
    landmarks = np.zeros((21, 3))
    for i, lm in enumerate(landmarks_list):
        landmarks[i] = [lm['x'], lm['y'], lm['z']]
        
    # Normalization Step 1: Translation invariance (make wrist the origin)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    
    # Normalization Step 2: Scale invariance (scale by max distance from wrist)
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist
        
    # Flatten to 63D vector for the MLP
    return landmarks.flatten()
