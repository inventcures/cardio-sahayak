# /// script
# dependencies = ["gradio", "transformers", "peft", "accelerate", "bitsandbytes", "torch", "Pillow", "huggingface_hub"]
# ///

import os
import gradio as gr
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

# Global variables for model and processor
model = None
processor = None

def load_model():
    global model, processor
    if model is not None:
        return

    print("Loading VLM Model and Adapters...")
    base_model_id = "google/medgemma-27b-it"
    adapter_model_id = "tp53/cardio-sahayak-vlm"
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not found in environment. May fail to load gated models.")

    processor = AutoProcessor.from_pretrained(base_model_id, token=token)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token
    )
    
    try:
        model = PeftModel.from_pretrained(base_model, adapter_model_id, token=token)
        print("Successfully loaded fine-tuned VLM adapters.")
    except Exception as e:
        print(f"Could not load adapters, falling back to base model: {e}")
        model = base_model


def analyze_ecg(image, custom_prompt):
    if image is None:
        return "Please upload an ECG image."
    
    # Lazy load the model on first request
    if model is None:
        try:
            load_model()
        except Exception as e:
            return f"Error loading model (requires GPU and HF_TOKEN): {e}"

    prompt_text = custom_prompt if custom_prompt.strip() else "Write a diagnostic report for this ECG image."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    try:
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=input_text, images=[image], return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512)
            
        generated_text = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        return f"Error during inference: {e}"

# Build the Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown("# 🫀 Cardio-Sahayak India - ECG VLM Assistant")
    gr.Markdown("Upload a 12-lead ECG image. The fine-tuned **MedGemma-27B** VLM will analyze the waveforms and generate a comprehensive clinical report tailored for South Asian cardiology guidelines.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload ECG Image")
            input_prompt = gr.Textbox(
                label="Custom Instructions (Optional)", 
                placeholder="e.g., 'Focus specifically on the ST segments.'",
                lines=2
            )
            analyze_btn = gr.Button("Analyze ECG", variant="primary")
            
        with gr.Column(scale=1):
            output_report = gr.Textbox(label="Generated Clinical Report", lines=15, show_copy_button=True)
            
    analyze_btn.click(fn=analyze_ecg, inputs=[input_image, input_prompt], outputs=output_report)

    gr.Markdown("---")
    gr.Markdown("**Note:** This is a research prototype. Do not use for actual medical diagnosis without physician supervision.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
