import json
import glob
import numpy as np
import os
from src.data.preprocess import normalize_landmarks, normalize_both_hands


def _convert_landmarks(landmarks):
    """Convert a list of landmark dicts to normalized 63D feature vector."""
    lm_list = [{"x": lm['x'], "y": lm['y'], "z": lm['z']} for lm in landmarks]
    return normalize_landmarks(lm_list)


def _convert_two_hand_frame(frame):
    """Convert a two-hand frame dict to normalized 126D feature vector."""
    left = frame.get('left_hand', [])
    right = frame.get('right_hand', [])
    left_lm = [{"x": lm['x'], "y": lm['y'], "z": lm['z']} for lm in left] if left else []
    right_lm = [{"x": lm['x'], "y": lm['y'], "z": lm['z']} for lm in right] if right else []
    return normalize_both_hands(left_lm, right_lm)


def prepare_static_dataset(json_dir="frontend", output_dir="data"):
    """Convert all static sign JSON files in json_dir to numpy arrays."""
    pattern = os.path.join(json_dir, "asl_dataset*.json")
    files = sorted([f for f in glob.glob(pattern) if 'dynamic' not in f and 'Zone' not in f])

    if not files:
        print(f"Error: No static dataset JSON files found matching {pattern}")
        return

    os.makedirs(output_dir, exist_ok=True)

    X, y = [], []
    for json_path in files:
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        for sample in dataset:
            features = _convert_landmarks(sample['landmarks'])
            X.append(features)
            y.append(sample['label'])
        print(f"  Loaded {len(dataset)} samples from {json_path}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    print(f"Static: {len(X)} total samples saved to {output_dir}/X.npy, y.npy")


def prepare_dynamic_dataset(json_dir="frontend", output_dir="data", seq_len=30, json_files=None):
    """Convert motion sign JSON files to numpy arrays.

    Args:
        json_dir: Directory to search for JSON files (used when json_files is None).
        output_dir: Directory to save numpy arrays.
        seq_len: Sequence length for padding/truncation.
        json_files: Optional list of specific JSON file paths to use.
                    If None, all asl_dynamic_dataset*.json in json_dir are used.

    Each sample is a sequence of two-hand frames, padded or truncated to seq_len.
    Output shape: X_dynamic (N, seq_len, 126), y_dynamic (N,)
    """
    if json_files:
        files = [f for f in json_files if os.path.exists(f)]
    else:
        pattern = os.path.join(json_dir, "asl_dynamic_dataset*.json")
        files = sorted([f for f in glob.glob(pattern) if 'Zone' not in f])

    if not files:
        print(f"No dynamic dataset JSON files found matching {pattern} — skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    X, y = [], []
    for json_path in files:
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        count = 0
        for sample in dataset:
            sequence = sample['sequence']
            frames = []
            for frame in sequence:
                # Support both old format (list of landmarks) and new format ({left_hand, right_hand})
                if isinstance(frame, dict) and ('left_hand' in frame or 'right_hand' in frame):
                    features = _convert_two_hand_frame(frame)
                else:
                    # Legacy single-hand: put into right hand, zero-pad left
                    features = _convert_two_hand_frame({'right_hand': frame})
                frames.append(features)

            # Pad or truncate to seq_len
            frames_arr = np.array(frames, dtype=np.float32)  # (T, 126)
            if len(frames_arr) >= seq_len:
                frames_arr = frames_arr[:seq_len]
            else:
                pad = np.zeros((seq_len - len(frames_arr), 126), dtype=np.float32)
                frames_arr = np.concatenate([frames_arr, pad], axis=0)

            X.append(frames_arr)
            y.append(sample['label'])
            count += 1
        print(f"  Loaded {count} sequences from {json_path}")

    X = np.array(X, dtype=np.float32)  # (N, seq_len, 126)
    y = np.array(y, dtype=np.int64)

    np.save(os.path.join(output_dir, "X_dynamic.npy"), X)
    np.save(os.path.join(output_dir, "y_dynamic.npy"), y)
    print(f"Dynamic: {len(X)} total samples (seq_len={seq_len}) saved to {output_dir}/X_dynamic.npy, y_dynamic.npy")


def prepare_dataset():
    """Prepare both static and dynamic datasets."""
    prepare_static_dataset()
    prepare_dynamic_dataset()


if __name__ == "__main__":
    prepare_dataset()
