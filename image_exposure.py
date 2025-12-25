import easyocr
import exifread

def extract_metadata(image_path):
    metadata = {}
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
        for tag in ["GPS GPSLatitude", "GPS GPSLongitude", "Image DateTime"]:
            if tag in tags:
                metadata[tag] = str(tags[tag])
    return metadata

def extract_text(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    texts = []
    for bbox, text, prob in results:
        if prob > 0.3:
            texts.append(text)
    return texts

def analyze_image(image_path):
    evidence = {}

    # Extract metadata
    metadata = extract_metadata(image_path)
    if metadata:
        evidence["metadata"] = metadata

    # Extract OCR text
    ocr_texts = extract_text(image_path)
    if ocr_texts:
        evidence["ocr_texts"] = ocr_texts

    # Simple exposure scoring
    score = 0
    details = []
    if "metadata" in evidence:
        score += 20
        details.append({"feature": "Metadata detected", "weight": 20})
    if "ocr_texts" in evidence:
        score += 25
        details.append({"feature": f"OCR texts found: {len(ocr_texts)}", "weight": 25})

    return {"score": score, "details": details, "evidence": evidence}

# Example usage:
if __name__ == "__main__":
    test_image_path = "image.jpg"  # Replace with your test image path
    results = analyze_image(test_image_path)
    print("Image exposure score:", results["score"])
    print("Details:", results["details"])
    print("Evidence:", results["evidence"])
