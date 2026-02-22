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


def normalize_both_hands(left_landmarks, right_landmarks):
    """
    Normalizes both hands independently and concatenates into a 126D vector.
    Missing hands (empty list) are zero-padded.
    Used by the dynamic model for two-hand phrase signs.
    """
    if left_landmarks and len(left_landmarks) == 21:
        left = normalize_landmarks(left_landmarks)
    else:
        left = np.zeros(63, dtype=np.float32)

    if right_landmarks and len(right_landmarks) == 21:
        right = normalize_landmarks(right_landmarks)
    else:
        right = np.zeros(63, dtype=np.float32)

    return np.concatenate([left, right])
