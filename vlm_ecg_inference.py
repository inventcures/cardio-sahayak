# /// script
# dependencies = ["transformers", "torch", "Pillow", "qwen-vl-utils"]
# ///

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

def generate_ecg_report(image_path: str):
    """
    Takes an ECG image (12-lead graph) and generates a detailed clinical report.
    This assumes the model has been instruction-tuned on ECG image-report pairs (e.g., using MEIT framework).
    """
    # 1. Load the fine-tuned VLM (e.g., Qwen2-VL-7B fine-tuned on ECGInstruct)
    # Using the base model here as a placeholder for the eventually fine-tuned 'cardio-sahayak-vision'
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    # In production, we'd load this with 4-bit quantization or on a dedicated GPU
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 2. Formulate the prompt requiring rigorous clinical analysis
    prompt = (
        "You are an expert cardiologist analyzing an electrocardiogram (ECG) for a patient of South Asian descent. "
        "Analyze the provided 12-lead ECG image carefully. "
        "1. Identify the rhythm and calculate the approximate heart rate. "
        "2. Analyze the P wave, PR interval, QRS complex, and QT interval. "
        "3. Look for specific morphological abnormalities such as ST-segment elevation/depression, T-wave inversion, or pathological Q waves. "
        "4. Provide a detailed, scientifically accurate, and rigorous clinical interpretation."
    )

    # 3. Prepare the multimodal input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 4. Process inputs for the VLM
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # 5. Generate the interpretation report
    print("Analyzing ECG image and generating report...")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    # Trim the prompt from the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    report = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return report

if __name__ == "__main__":
    # Example usage (Requires an actual ECG image to run)
    # fake path for demonstration
    sample_ecg = "sample_12_lead_ecg.jpg" 
    
    # Create a dummy image just so the script can compile if tested
    dummy_image = Image.new('RGB', (1024, 768), color = (255, 255, 255))
    dummy_image.save(sample_ecg)
    
    try:
        clinical_report = generate_ecg_report(sample_ecg)
        print("
=== Generated Clinical ECG Report ===")
        print(clinical_report)
    except Exception as e:
        print(f"Inference requires GPU and model weights. Error: {e}")
