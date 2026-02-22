from flask import Flask, jsonify, request, abort, render_template
from flask_cors import CORS
from collections import deque
import torch
import numpy as np
import os
from src.models.sign_model import ASLClassifier, ASLDynamicClassifier
from src.data.preprocess import normalize_landmarks, normalize_both_hands

app = Flask("ASL Recognition API",
            template_folder=os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
app.config["DESCRIPTION"] = "Real-time ASL Alphabet Classification with Motion + Phrases"
CORS(app)

# --- Static model (A-Z static signs) ---
static_model = ASLClassifier()
static_model_path = "models/best_model.pth"
if os.path.exists(static_model_path):
    static_model.load_state_dict(torch.load(static_model_path, map_location='cpu'))
static_model.eval()

# --- Dynamic model (J, Z, Hello, Goodbye, Thank You, My, name, I, love, you) ---
# Labels: 0=J, 1=Z, 2=Hello, 3=Goodbye, 4=unused, 5=Thank You, 6=My, 7=name, 8=I, 9=love, 10=you
DYNAMIC_CLASSES = ['J', 'Z', 'Hello', 'Goodbye', '(unused)', 'Thank You', 'My', 'name', 'I', 'love', 'you']
SEQ_LEN = 30
dynamic_model = None
dynamic_model_path = "models/best_dynamic_model.pth"
if os.path.exists(dynamic_model_path):
    try:
        dynamic_model = ASLDynamicClassifier(num_classes=len(DYNAMIC_CLASSES))
        dynamic_model.load_state_dict(torch.load(dynamic_model_path, map_location='cpu'))
        dynamic_model.eval()
    except RuntimeError as e:
        print(f"Warning: Could not load dynamic model (class mismatch). Retrain needed. Error: {e}")
        dynamic_model = None

classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# --- Frame buffer for motion detection ---
frame_buffer = deque(maxlen=SEQ_LEN)
# --- Phrase state ---
phrase_letters = []


def _normalize_payload(landmarks):
    """Convert landmarks from various formats and normalize to 63D."""
    lm_list = []
    for lm in landmarks:
        if isinstance(lm, dict):
            lm_list.append({"x": float(lm["x"]), "y": float(lm["y"]), "z": float(lm["z"])})
        elif isinstance(lm, (list, tuple)):
            lm_list.append({"x": float(lm[0]), "y": float(lm[1]), "z": float(lm[2])})
        else:
            lm_list.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})
    return normalize_landmarks(lm_list)


def _parse_landmarks(landmarks):
    """Convert landmarks from various input formats to list of dicts."""
    if not landmarks:
        return []
    lm_list = []
    for lm in landmarks:
        if isinstance(lm, dict):
            lm_list.append({"x": float(lm["x"]), "y": float(lm["y"]), "z": float(lm["z"])})
        elif isinstance(lm, (list, tuple)):
            lm_list.append({"x": float(lm[0]), "y": float(lm[1]), "z": float(lm[2])})
        else:
            lm_list.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})
    return lm_list


@app.route("/", methods=["GET"])
def index():
    """Serve the main ASL Translator frontend."""
    return render_template("asl_translator.html")


@app.route("/collect", methods=["GET"])
def collect():
    """Serve the data collection page."""
    return render_template("collect.html")


@app.route("/health", methods=["GET"])
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
    """Predict a dynamic/motion sign from a sequence of two-hand frames (126D per frame)."""
    if dynamic_model is None:
        abort(503, description="Dynamic model not loaded. Train it first.")

    data = request.get_json()
    sequence = data.get("sequence", [])

    # Each frame has {left_hand: [...], right_hand: [...]}
    seq = []
    for frame in sequence:
        left = _parse_landmarks(frame.get("left_hand", []))
        right = _parse_landmarks(frame.get("right_hand", []))
        if not left and not right:
            abort(400, description="Each frame must have at least one hand")
        features = normalize_both_hands(left, right)
        seq.append(features)

    # Pad or truncate to SEQ_LEN
    seq = np.array(seq, dtype=np.float32)
    if len(seq) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(seq), 126), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    elif len(seq) > SEQ_LEN:
        seq = seq[:SEQ_LEN]

    tensor = torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, 126)

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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
