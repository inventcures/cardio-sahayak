import os
import gradio as gr
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import traceback

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
    
    # Check if GPU is available to avoid bitsandbytes errors on CPU-only Spaces
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected! A GPU is required to load this 27B parameter model in 4-bit quantization. If you are on a free Hugging Face Space, the container will crash due to Out-Of-Memory (OOM) before it finishes loading. Please duplicate this space and upgrade to an A10G or L4 GPU hardware tier.")

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
        yield "Please upload an ECG image."
        return
    
    yield "Initializing inference...\nStep 1/3: Checking hardware and loading 27B model weights (this may take a few minutes if cold-starting)..."
    
    # Lazy load the model on first request
    if model is None:
        try:
            load_model()
        except Exception as e:
            yield f"Error loading model:\n{str(e)}\n\n(Note: This model requires at least 16-24GB of VRAM to run successfully. It will fail on free CPU spaces.)"
            return

    yield "Step 2/3: Processing image and text inputs..."
    
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
        
        yield "Step 3/3: Generating clinical interpretation (this might take a moment depending on the GPU)..."
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512)
            
        generated_text = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        yield generated_text
    except Exception as e:
        error_trace = traceback.format_exc()
        yield f"Error during inference:\n{str(e)}\n\nDetails:\n{error_trace}"

# Build the Gradio UI
with gr.Blocks() as demo:
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
            
            gr.Examples(
                examples=[
                    ["examples/cad_stemi_ecg.jpg", "Analyze this ECG. Note the patient is a 42-year-old South Asian male with acute chest pain. Evaluate for early-onset STEMI."],
                    ["examples/cad_ischemia_ecg.jpg", "Evaluate for signs of Ischemic Heart Disease (e.g., ST depressions)."],
                    ["examples/lvh_ecg.jpg", "Assess for voltage criteria indicative of Left Ventricular Hypertrophy (LVH)."],
                    ["examples/afib_ecg.jpg", "Analyze the rhythm and rate. Look for Atrial Fibrillation."],
                    ["examples/hcm_dcm_ecg.jpg", "Patient has a family history of the MYBPC3 variant. Assess for signs of cardiomyopathy."],
                    ["examples/normal_variant_ecg.jpg", "Write a standard diagnostic report for this ECG image."]
                ],
                inputs=[input_image, input_prompt],
                label="Sample Clinical Cases (South Asian Phenotypes)"
            )
            
        with gr.Column(scale=1):
            output_report = gr.Textbox(label="AI-Generated Clinical Interpretation & Findings", lines=15)
            
    analyze_btn.click(fn=analyze_ecg, inputs=[input_image, input_prompt], outputs=output_report)

    gr.Markdown("---")
    gr.Markdown("""
### ⚠️ Disclaimer
**This application is an experimental research prototype intended solely for academic and demonstrative purposes.** 
It is **not** a certified medical device and should **not** be used for actual clinical diagnosis, triage, or patient management without the direct supervision of a qualified physician. The AI-generated findings may contain errors, omissions, or "hallucinations." Always consult a healthcare professional for medical advice.
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"))
