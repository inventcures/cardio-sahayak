# /// script
# dependencies = ["transformers", "torch", "Pillow", "accelerate", "bitsandbytes"]
# ///

import os
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests

def generate_cardio_vlm_report(image_path_or_url: str):
    """
    Uses MedGemma-27B-it (Gemma 3 Multimodal) to analyze an ECG and generate a report.
    """
    model_id = "google/medgemma-27b-it"
    token = os.environ.get("HF_TOKEN")

    print(f"Loading MedGemma Multimodal model: {model_id}")
    
    # 4-bit quantization for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token
    )

    # Load image
    if image_path_or_url.startswith("http"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw)
    else:
        image = Image.open(image_path_or_url)

    # Prepare Prompt
    prompt = (
        "You are an expert cardiologist. Analyze this 12-lead ECG image. "
        "Provide a detailed clinical report including: "
        "1. Heart rate and rhythm. "
        "2. Intervals (PR, QRS, QT). "
        "3. Morphological findings (ST elevation/depression, T-wave changes). "
        "4. Final clinical impression and suggested next steps for a South Asian patient."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Process inputs
    # processor.apply_chat_template handles the multimodal formatting for Gemma 3
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=input_text, images=[image], return_tensors="pt").to(model.device)

    print("Generating report...")
    output = model.generate(**inputs, max_new_tokens=1024)
    
    # Decode only the generated part
    generated_text = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_text

if __name__ == "__main__":
    # Test with a dummy image if no real one provided
    test_img = "test_ecg.png"
    if not os.path.exists(test_img):
        img = Image.new('RGB', (896, 896), color = (255, 255, 255))
        img.save(test_img)
    
    try:
        report = generate_cardio_vlm_report(test_img)
        print("\n--- CLINICAL ECG REPORT ---")
        print(report)
    except Exception as e:
        print(f"Error during VLM inference: {e}")
