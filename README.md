# SignFlow

Real-time American Sign Language to text, in the browser. Built at SacHacks VII (Feb 2026).

Your webcam runs MediaPipe hand tracking in the browser. The 21 hand landmarks per frame are sent to a small Flask API, where two PyTorch models turn them into letters and signs, and a phrase builder assembles the sentence.

- **Static letters**: an MLP over a single normalized hand pose
- **Motion signs** (J, Z, Hello, Thank You, etc.): an LSTM over 30-frame two-hand sequences
- **Dataset**: 12,919 hand poses and 749 motion sequences, self-collected with the built-in collector page

**Live:** https://sac-hacks-vii.onrender.com/translator

## Run locally

```bash
pip install -r requirements.txt
python -m src.api.main
```

Then open http://localhost:5000/translator. The collector page is at `/collect`.

The translator page points at the Render API by default. To hit your local server instead, change `API_BASE` in `frontend/asl_translator.html` to `http://localhost:5000`.

## Retrain the models

```bash
python -m src.data.prepare_dataset   # JSON captures in frontend/ -> data/*.npy
python -m src.pipelines.train        # writes models/best_model.pth and best_dynamic_model.pth
```

## Layout

- `src/api/main.py`: Flask API (`/predict`, `/predict_dynamic`, `/phrase`) and page serving
- `src/models/sign_model.py`: the MLP and LSTM
- `src/data/`: landmark normalization and dataset prep
- `frontend/`: translator and collector pages, plus the raw JSON captures
- `models/`: trained checkpoints
