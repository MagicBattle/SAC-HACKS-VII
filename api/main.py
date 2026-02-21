from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from collections import deque
import torch
import numpy as np
import os
from src.models.sign_model import ASLClassifier, ASLDynamicClassifier
from src.data.preprocess import normalize_landmarks

app = FastAPI(title="ASL Recognition API", description="Real-time ASL Alphabet Classification with Motion + Phrases")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class Landmark(BaseModel):
    x: float
    y: float
    z: float

class LandmarkPayload(BaseModel):
    landmarks: List[Landmark]

class SequencePayload(BaseModel):
    sequence: List[List[Landmark]]  # List of frames, each frame is 21 landmarks

class PhraseAction(BaseModel):
    action: str  # "add", "space", "backspace", "clear", "get"
    letter: Optional[str] = None


def _normalize_payload(landmarks: List[Landmark]):
    lm_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
    return normalize_landmarks(lm_list)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "static_model_loaded": os.path.exists(static_model_path),
        "dynamic_model_loaded": dynamic_model is not None,
    }


@app.post("/predict")
async def predict(payload: LandmarkPayload):
    """Predict a static sign from a single frame of 21 landmarks."""
    if len(payload.landmarks) != 21:
        raise HTTPException(status_code=400, detail="Must provide exactly 21 landmarks")

    features = _normalize_payload(payload.landmarks)
    tensor = torch.FloatTensor(features).unsqueeze(0)

    with torch.no_grad():
        outputs = static_model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)

    return {
        "prediction": classes[predicted.item()],
        "confidence": float(conf.item()),
        "type": "static",
    }


@app.post("/predict_dynamic")
async def predict_dynamic(payload: SequencePayload):
    """Predict a dynamic/motion sign (J or Z) from a sequence of frames."""
    if dynamic_model is None:
        raise HTTPException(status_code=503, detail="Dynamic model not loaded. Train it first.")

    # Normalize each frame in the sequence
    seq = []
    for frame in payload.sequence:
        if len(frame) != 21:
            raise HTTPException(status_code=400, detail="Each frame must have 21 landmarks")
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

    return {
        "prediction": DYNAMIC_CLASSES[predicted.item()],
        "confidence": float(conf.item()),
        "type": "dynamic",
    }


@app.post("/phrase")
async def phrase(action: PhraseAction):
    """Manage the phrase being built letter by letter."""
    global phrase_letters

    if action.action == "add" and action.letter:
        phrase_letters.append(action.letter)
    elif action.action == "space":
        phrase_letters.append(" ")
    elif action.action == "backspace":
        if phrase_letters:
            phrase_letters.pop()
    elif action.action == "clear":
        phrase_letters = []
    elif action.action == "get":
        pass  # Just return the current phrase
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

    return {"phrase": "".join(phrase_letters)}
