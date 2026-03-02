# /// script
# dependencies = ["transformers", "torch", "Pillow"]
# ///

import os
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image

def classify_ecg_zero_shot(image_path: str, labels: list):
    """
    Uses MedSigLIP-448 for zero-shot ECG classification.
    Labels are clinical findings like ['ST-segment elevation', 'Normal sinus rhythm', 'Atrial fibrillation'].
    """
    model_id = "google/medsiglip-448"
    token = os.environ.get("HF_TOKEN")

    print(f"Loading MedSigLIP for Zero-Shot Classification: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = AutoModel.from_pretrained(model_id, token=token)
    
    # Pre-process labels for MedSigLIP
    # It prefers short, descriptive clinical findings.
    processed_labels = [f"This ECG shows {label}." for label in labels]

    image = Image.open(image_path)
    
    # Prepare inputs
    inputs = processor(
        text=processed_labels,
        images=[image],
        padding="max_length",
        max_length=64, # MedSigLIP limit
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # Softmax over the text-image similarity logits
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    results = {label: float(prob) for label, prob in zip(labels, probs[0])}
    return results

if __name__ == "__main__":
    # Example usage
    test_img = "test_ecg.png"
    if not os.path.exists(test_img):
        img = Image.new('RGB', (448, 448), color = (255, 255, 255))
        img.save(test_img)
    
    findings_to_check = [
        "Normal Sinus Rhythm",
        "ST-Segment Elevation Myocardial Infarction (STEMI)",
        "Atrial Fibrillation",
        "T-Wave Inversion",
        "Pathological Q Waves"
    ]

    try:
        probabilities = classify_ecg_zero_shot(test_img, findings_to_check)
        print("
--- MEDSIGLIP ZERO-SHOT PROBABILITIES ---")
        for finding, prob in probabilities.items():
            print(f"{finding}: {prob:.2%}")
    except Exception as e:
        print(f"Error during MedSigLIP inference: {e}")
