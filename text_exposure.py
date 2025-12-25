import re
import spacy
from transformers import pipeline

# Load spaCy and Hugging Face models
def load_models():
    nlp = spacy.load("en_core_web_sm")
    hf_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return nlp, hf_ner, sentiment

nlp, hf_ner, sentiment_pipe = load_models()

# PII patterns for regex detection
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
        score = e.get("score", 1.0)
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
        contrib = w * score_factor * text_len_factor
        details.append({"evidence": e, "weight": w, "factor": score_factor * text_len_factor, "contribution": contrib})
        total += contrib
    return {"rawscore": total, "exposurescore": min(100, round(total, 2)), "details": details}

def analyze_text(text):
    if not text.strip():
        return {"error": "empty text"}
    evidence = extract_evidence(text)
    score = compute_exposure_score(evidence)
    sentiment = sentiment_pipe(text[:512])[0]
    return {
        "inputtext": text,
        "sentiment": sentiment,
        "evidence": evidence,
        "score": score,
    }

# Example usage:
if __name__ == "__main__":
    sample = """
    My name is Srinivas, and I have been working at Infosys Technologies in Bangalore.
    You can reach me at srinivas.reddy@example.com or call me at 91 9876512345.
    Every morning, I leave my apartment around 8:00 AM to go to the office.
    I traveled to New York on 12th August 2023 for a meeting.
    Check my site https://srinivasportfolio.com
    """
    res = analyze_text(sample)
    print("Exposure Score:", res["score"]["exposurescore"])
    print("Sentiment:", res["sentiment"])
    print("Evidence found:")
    for d in res["score"]["details"]:
        e = d["evidence"]
        print("-", e.get("source"), e.get("text"), e.get("type", e.get("label")), round(d["contribution"],2))
