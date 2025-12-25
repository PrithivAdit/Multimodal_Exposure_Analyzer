from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
import re
import spacy
from transformers import pipeline
import easyocr
import exifread
import librosa
import whisper

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_np(obj):
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

# Text analysis setup

def load_models():
    nlp = spacy.load("en_core_web_sm")
    hf_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return nlp, hf_ner, sentiment

nlp, hf_ner, sentiment_pipe = load_models()

PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9._%-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"\b\d{6,15}\b"),
    "url": re.compile(r"https?://[^\s]+"),
    "creditcardlike": re.compile(r"\b\d{13,16}\b"),
}

WEIGHTS = {
    "email": 25, "phone": 25, "creditcardlike": 40, "url": 10,
    "addresslike": 20, "PERSON": 15, "GPE": 15, "LOC": 12, "ORG": 12,
    "DATE": 8, "TIME": 6, "MONEY": 20, "defaultentity": 5,
}

def detect_pii_regex(text):
    hits = []
    for label, pattern in PII_PATTERNS.items():
        for m in pattern.finditer(text):
            hits.append({"source": "regex", "type": label, "text": m.group(0), "span": m.span()})
    return hits

def run_spacy_ner(text):
    doc = nlp(text)
    return [{"source": "spacy", "label": ent.label_, "text": ent.text, "span": (ent.start_char, ent.end_char)} for ent in doc.ents]

def run_hf_ner(text):
    out = hf_ner(text)
    entities = []
    for e in out:
        label = e["entity_group"]
        word = e["word"]
        score = float(e.get("score", 1.0))
        entities.append({"source": "hf", "label": label, "text": word, "score": score})
    return entities

def extract_evidence(text):
    evidence = []
    evidence.extend(detect_pii_regex(text))
    evidence.extend(run_spacy_ner(text))
    evidence.extend(run_hf_ner(text))
    return evidence

def compute_exposure_score(evidence):
    total = 0
    details = []
    for e in evidence:
        t = e.get("type") or e.get("label") or ""
        w = WEIGHTS.get(t, WEIGHTS.get(t.upper(), WEIGHTS["defaultentity"]))
        score_factor = float(e.get("score", 1.0))
        text_len_factor = min(1.0, len(e.get("text", ""))/30.0 + 0.2)
        contrib = float(w) * score_factor * text_len_factor
        details.append({
            "evidence": e,
            "weight": float(w),
            "factor": float(score_factor * text_len_factor),
            "contribution": float(contrib)
        })
        total += contrib
    return {
        "rawscore": float(total),
        "exposurescore": min(100, round(float(total), 2)),
        "details": details
    }

def analyze_text(text):
    if not text.strip():
        return {"error": "empty text"}
    evidence = extract_evidence(text)
    score = compute_exposure_score(evidence)
    sentiment = sentiment_pipe(text[:512])[0]
    sentiment["score"] = float(sentiment["score"])
    result = {
        "inputtext": text,
        "sentiment": sentiment,
        "evidence": evidence,
        "score": score,
    }
    return convert_np(result)

@app.post("/analyze_text/")
async def analyze_text_api(text: str = Form(...)):
    return analyze_text(text)

# Image analysis

def analyze_image(image_path):
    evidence = {}
    with open(image_path, 'rb') as f:
        exif = exifread.process_file(f)
    if exif:
        evidence['exif'] = {tag: str(exif[tag]) for tag in exif.keys() if tag.startswith('GPS') or tag.startswith('Image DateTime')}
    reader = easyocr.Reader(['en'])
    ocr_results = reader.readtext(image_path)
    texts = []
    for res in ocr_results:
        conf = float(res[2]) if isinstance(res[2], (float, np.floating)) else res[2]
        texts.append({"text": res[1], "confidence": conf})
    if texts:
        evidence['ocr_texts'] = texts
    score = 0.0
    details = []
    if 'exif' in evidence:
        score += 20.0
        details.append({"feature": "EXIF metadata found", "weight": 20.0})
    if 'ocr_texts' in evidence:
        score += 30.0
        details.append({"feature": f"OCR texts found: {len(texts)}", "weight": 30.0})
    result = {"score": float(score), "details": details, "evidence": evidence}
    return convert_np(result)

@app.post("/analyze_image/")
async def analyze_image_api(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    contents = await file.read()
    with open(temp_file, "wb") as f:
        f.write(contents)
    result = analyze_image(temp_file)
    os.remove(temp_file)
    return JSONResponse(content=result)

# Audio analysis using librosa instead of pydub

audio_model = whisper.load_model("base")  # Load once globally

SENSITIVE_AUDIO_KEYWORDS = ["password", "ssn", "credit card", "secret", "confidential"]

def extract_audio_metadata(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    duration_seconds = librosa.get_duration(y=y, sr=sr)
    metadata = {
        "duration_seconds": duration_seconds,
        "sampling_rate": sr,
        "channels": 1 if y.ndim == 1 else y.shape[0]
    }
    return metadata

def transcribe_audio(audio_path):
    result = audio_model.transcribe(audio_path)
    return result["text"]

def compute_audio_exposure_score(metadata, transcript):
    score = 0
    details = []
    if metadata["duration_seconds"] > 60:
        score += 10
        details.append({"feature": "long duration", "contribution": 10})
    keyword_count = sum(transcript.lower().count(kw) for kw in SENSITIVE_AUDIO_KEYWORDS)
    if keyword_count:
        contribution = min(50, keyword_count * 10)
        score += contribution
        details.append({"feature": "sensitive keywords", "contribution": contribution})
    score = min(score, 100)
    return {"exposure_score": score, "details": details}

def analyze_audio(audio_path):
    metadata = extract_audio_metadata(audio_path)
    transcript = transcribe_audio(audio_path)
    score_details = compute_audio_exposure_score(metadata, transcript)
    return {"metadata": metadata, "transcript": transcript, "score": score_details}

@app.post("/analyze_audio/")
async def analyze_audio_api(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    contents = await file.read()
    with open(temp_file, "wb") as f:
        f.write(contents)
    try:
        result = analyze_audio(temp_file)
    finally:
        os.remove(temp_file)
    return JSONResponse(content=convert_np(result))

@app.get("/")
async def root():
    return {"message": "Exposure analyzer API running. Use POST /analyze_text/, /analyze_image/, or /analyze_audio/"}
