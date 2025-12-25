# audio_analyzer.py

from pydub import AudioSegment
import whisper

# Load whisper model once globally
audio_model = whisper.load_model("base")  # Change model size as needed: "small", "medium", "large"

SENSITIVE_AUDIO_KEYWORDS = ["password", "ssn", "credit card", "secret", "confidential"]

def extract_audio_metadata(audio_path):
    audio = AudioSegment.from_file(audio_path)
    metadata = {
        "duration_seconds": len(audio) / 1000,
        "channels": audio.channels,
        "framerate": audio.frame_rate,
        "sample_width": audio.sample_width
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python audio_analyzer.py <audiofile>")
        sys.exit(1)
    audio_file = sys.argv[1]
    result = analyze_audio(audio_file)
    print("Exposure Score:", result["score"]["exposure_score"])
    print("Details:", result["score"]["details"])
    print("Metadata:", result["metadata"])
    print("Transcript:", result["transcript"])
