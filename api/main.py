from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from collections import deque
import torch
import numpy as np
import os
from src.models.sign_model import ASLClassifier, ASLDynamicClassifier
from src.data.preprocess import normalize_landmarks

app = Flask("ASL Recognition API")
app.config["DESCRIPTION"] = "Real-time ASL Alphabet Classification with Motion + Phrases"
CORS(app)

# --- Static model (A-Z static signs) ---
static_model = ASLClassifier()
static_model_path = "models/best_model.pth"
if os.path.exists(static_model_path):
    static_model.load_state_dict(torch.load(static_model_path, map_location='cpu'))
static_model.eval()

# --- Dynamic model (J, Z, Hello, Goodbye, Please, Thank You) ---
DYNAMIC_CLASSES = ['J', 'Z', 'Hello', 'Goodbye', 'Please', 'Thank You']
SEQ_LEN = 30
dynamic_model = None
dynamic_model_path = "models/best_dynamic_model.pth"
if os.path.exists(dynamic_model_path):
    dynamic_model = ASLDynamicClassifier(num_classes=len(DYNAMIC_CLASSES))
    dynamic_model.load_state_dict(torch.load(dynamic_model_path, map_location='cpu'))
    dynamic_model.eval()

classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# --- Frame buffer for motion detection ---
frame_buffer = deque(maxlen=SEQ_LEN)
# --- Phrase state ---
phrase_letters = []


def _normalize_payload(landmarks):
    lm_list = [{"x": lm["x"], "y": lm["y"], "z": lm["z"]} for lm in landmarks]
    return normalize_landmarks(lm_list)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "static_model_loaded": os.path.exists(static_model_path),
        "dynamic_model_loaded": dynamic_model is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Predict a static sign from a single frame of 21 landmarks."""
    data = request.get_json()
    landmarks = data.get("landmarks", [])
    if len(landmarks) != 21:
        abort(400, description="Must provide exactly 21 landmarks")

    features = _normalize_payload(landmarks)
    tensor = torch.FloatTensor(features).unsqueeze(0)

    with torch.no_grad():
        outputs = static_model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)

    return jsonify({
        "prediction": classes[predicted.item()],
        "confidence": float(conf.item()),
        "type": "static",
    })


@app.route("/predict_dynamic", methods=["POST"])
def predict_dynamic():
    """Predict a dynamic/motion sign (J or Z) from a sequence of frames."""
    if dynamic_model is None:
        abort(503, description="Dynamic model not loaded. Train it first.")

    data = request.get_json()
    sequence = data.get("sequence", [])

    # Normalize each frame in the sequence
    seq = []
    for frame in sequence:
        if len(frame) != 21:
            abort(400, description="Each frame must have 21 landmarks")
        features = _normalize_payload(frame)
        seq.append(features)

    # Pad or truncate to SEQ_LEN
    seq = np.array(seq, dtype=np.float32)
    if len(seq) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(seq), 63), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    elif len(seq) > SEQ_LEN:
        seq = seq[:SEQ_LEN]

    tensor = torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, 63)

    with torch.no_grad():
        outputs = dynamic_model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)

    return jsonify({
        "prediction": DYNAMIC_CLASSES[predicted.item()],
        "confidence": float(conf.item()),
        "type": "dynamic",
    })


@app.route("/phrase", methods=["POST"])
def phrase():
    """Manage the phrase being built letter by letter."""
    global phrase_letters

    data = request.get_json()
    action = data.get("action")
    letter = data.get("letter")

    if action == "add" and letter:
        phrase_letters.append(letter)
    elif action == "space":
        phrase_letters.append(" ")
    elif action == "backspace":
        if phrase_letters:
            phrase_letters.pop()
    elif action == "clear":
        phrase_letters = []
    elif action == "get":
        pass  # Just return the current phrase
    else:
        abort(400, description="Invalid action")

    return jsonify({"phrase": "".join(phrase_letters)})
