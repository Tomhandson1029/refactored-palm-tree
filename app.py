import os
import json
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import textstat


# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "roberta-goemotions-final")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MAX_LENGTH = 128
DEFAULT_THRESHOLD = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Load model, tokenizer, labels
# ----------------------------
print("Loading model and tokenizer from:", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

labels_path = os.path.join(MODEL_DIR, "labels.json")
with open(labels_path, "r") as f:
    label_names = json.load(f)

id2label = {i: name for i, name in enumerate(label_names)}

print("Loaded", len(label_names), "labels.")


# ----------------------------
# Helper functions
# ----------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def readability_flesch(text: str) -> float:
    """Flesch Reading Ease score (higher = easier)."""
    return textstat.flesch_reading_ease(text)


def readability_grade(text: str) -> float:
    """Flesch–Kincaid Grade Level (approx school grade)."""
    return textstat.flesch_kincaid_grade(text)


def analyze_text(text: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Run RoBERTa emotion model + readability metrics on a single text."""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    ).to(device)

    # Model forward
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0].cpu().numpy()

    probs = sigmoid(logits)

    # Multi-label selection
    pred_indices = np.where(probs >= threshold)[0]
    if len(pred_indices) == 0:
        # if nothing passes threshold, take best label
        pred_indices = [int(np.argmax(probs))]

    emotions = [
        {"label": id2label[i], "score": float(probs[i])}
        for i in sorted(pred_indices, key=lambda j: probs[j], reverse=True)
    ]

    # Readability metrics
    fres = readability_flesch(text)
    grade = readability_grade(text)

    if fres >= 80:
        desc = "very easy to read"
        clarity = "emotion is very easy to understand"
    elif fres >= 60:
        desc = "easy to read"
        clarity = "emotion should be easy to understand"
    elif fres >= 40:
        desc = "fairly difficult"
        clarity = "emotion might require some effort to understand"
    else:
        desc = "difficult or very difficult"
        clarity = "emotion may be hard to interpret from the text"

    return {
        "text": text,
        "emotions": emotions,
        "readability_flesch": fres,
        "readability_grade": grade,
        "readability_description": desc,
        "emotion_clarity_comment": clarity,
    }


# ----------------------------
# FastAPI app setup
# ----------------------------
app = FastAPI(
    title="Emotion + Readability API",
    version="1.0.0",
    description="RoBERTa GoEmotions + readability analysis",
)

# CORS (so the browser can call the API from other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    threshold: Optional[float] = None


# ----------------------------
# Static file serving (UI)
# ----------------------------

# Serve /static/* from the "static" folder
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    """
    Serve the UI (index.html) at the root URL.
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "UI not found. Place index.html in the 'static' folder."}


# ----------------------------
# API endpoints
# ----------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Emotion + Readability API is running."}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    Analyze the emotion and readability of the provided text.
    """
    thr = req.threshold if req.threshold is not None else DEFAULT_THRESHOLD
    result = analyze_text(req.text, threshold=thr)
    return result
