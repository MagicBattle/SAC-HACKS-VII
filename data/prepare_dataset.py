import json
import numpy as np
import os
from src.data.preprocess import normalize_landmarks


def _convert_landmarks(landmarks):
    """Convert a list of landmark dicts to normalized feature vector."""
    lm_list = [{"x": lm['x'], "y": lm['y'], "z": lm['z']} for lm in landmarks]
    return normalize_landmarks(lm_list)


def prepare_static_dataset(json_path="frontend/asl_dataset.json", output_dir="data"):
    """Convert static sign JSON data to numpy arrays."""
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        dataset = json.load(f)

    X, y = [], []
    for sample in dataset:
        features = _convert_landmarks(sample['landmarks'])
        X.append(features)
        y.append(sample['label'])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    print(f"Static: {len(X)} samples saved to {output_dir}/X.npy, y.npy")


def prepare_dynamic_dataset(json_path="frontend/asl_dynamic_dataset.json",
                            output_dir="data", seq_len=30):
    """Convert motion sign JSON data to numpy arrays.

    Each sample is a sequence of landmark frames, padded or truncated to seq_len.
    Output shape: X_dynamic (N, seq_len, 63), y_dynamic (N,)
    """
    if not os.path.exists(json_path):
        print(f"No dynamic dataset found at {json_path} — skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        dataset = json.load(f)

    X, y = [], []
    for sample in dataset:
        sequence = sample['sequence']
        frames = []
        for frame_landmarks in sequence:
            features = _convert_landmarks(frame_landmarks)
            frames.append(features)

        # Pad or truncate to seq_len
        frames_arr = np.array(frames, dtype=np.float32)  # (T, 63)
        if len(frames_arr) >= seq_len:
            frames_arr = frames_arr[:seq_len]
        else:
            pad = np.zeros((seq_len - len(frames_arr), 63), dtype=np.float32)
            frames_arr = np.concatenate([frames_arr, pad], axis=0)

        X.append(frames_arr)
        y.append(sample['label'])

    X = np.array(X, dtype=np.float32)  # (N, seq_len, 63)
    y = np.array(y, dtype=np.int64)

    np.save(os.path.join(output_dir, "X_dynamic.npy"), X)
    np.save(os.path.join(output_dir, "y_dynamic.npy"), y)
    print(f"Dynamic: {len(X)} samples (seq_len={seq_len}) saved to {output_dir}/X_dynamic.npy, y_dynamic.npy")


def prepare_dataset():
    """Prepare both static and dynamic datasets."""
    prepare_static_dataset()
    prepare_dynamic_dataset()


if __name__ == "__main__":
    prepare_dataset()
