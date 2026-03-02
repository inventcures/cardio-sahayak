import os
import modal
import io

# Define the Modal App
app = modal.App("cardio-sahayak-streamlit")

# Define the Modal Image with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit",
        "transformers>=4.45.0",
        "peft>=0.7.0",
        "accelerate",
        "bitsandbytes",
        "torch",
        "Pillow",
        "huggingface_hub",
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Set up the volume to store cached model weights so we don't download 27B parameters every boot
model_volume = modal.Volume.from_name("cardio-sahayak-models", create_if_missing=True)

# Define the inference class to keep the model loaded in memory
@app.cls(
    image=image,
    gpu="A100", # Use A100 for fast inference of the 27B model
    secrets=[modal.Secret.from_name("huggingface-secret")], # Re-using the secret you created earlier
    volumes={"/root/models": model_volume},
    timeout=600, # 10 minute timeout
    keep_warm=0  # Don't keep any instances warm by default to save money
)
class CardioSahayakModel:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
        from peft import PeftModel
        from huggingface_hub import login

        print("Initializing Cardio-Sahayak VLM...")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        base_model_id = "google/medgemma-27b-it"
        adapter_model_id = "tp53/cardio-sahayak-vlm"

        self.processor = AutoProcessor.from_pretrained(base_model_id, cache_dir="/root/models")
        
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
            cache_dir="/root/models"
        )
        
        try:
            self.model = PeftModel.from_pretrained(base_model, adapter_model_id, cache_dir="/root/models")
            print("Successfully loaded fine-tuned VLM adapters.")
        except Exception as e:
            print(f"Could not load adapters, falling back to base model: {e}")
            self.model = base_model

    @modal.method()
    def generate_report(self, image_bytes, custom_prompt):
        import torch
        from PIL import Image
        import io
        import traceback

        try:
            image = Image.open(io.BytesIO(image_bytes))
            
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
            
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=input_text, images=[image], return_tensors="pt", padding=True).to(self.model.device)
            
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=512)
                
            generated_text = self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            return f"Error during inference: {str(e)}\\n\\n{traceback.format_exc()}"


# The Streamlit web server
@app.function(
    image=image,
    allow_concurrent_inputs=100
)
@modal.web_server(8000)
def ui():
    import subprocess
    import os
    
    # Write the actual streamlit code to a temporary file
    streamlit_script = """
import streamlit as st
import modal
import os
import io
from PIL import Image

st.set_page_config(page_title="Cardio-Sahayak India", page_icon="🫀", layout="wide")

st.title("🫀 Cardio-Sahayak India - ECG VLM Assistant (Modal Fallback)")
st.markdown("Upload a 12-lead ECG image. The fine-tuned **MedGemma-27B** VLM will analyze the waveforms and generate a comprehensive clinical report tailored for South Asian cardiology guidelines.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded ECG", use_container_width=True)
        
    custom_prompt = st.text_area("Custom Instructions (Optional)", placeholder="e.g., 'Focus specifically on the ST segments.'")
    
    analyze_button = st.button("Analyze ECG", type="primary", use_container_width=True)

with col2:
    st.subheader("AI-Generated Clinical Interpretation & Findings")
    
    if analyze_button:
        if uploaded_file is None:
            st.error("Please upload an ECG image first.")
        else:
            with st.spinner("Initializing A100 GPU and loading 27B model (this may take 1-2 minutes on cold start)..."):
                try:
                    # Connect to the Modal Class defined in the outer scope
                    ModelCls = modal.Cls.lookup("cardio-sahayak-streamlit", "CardioSahayakModel")
                    model_instance = ModelCls()
                    
                    # Convert uploaded file to bytes to pass over the network
                    image_bytes = uploaded_file.getvalue()
                    
                    st.info("Running inference on A100 GPU...")
                    result = model_instance.generate_report.remote(image_bytes, custom_prompt)
                    
                    st.success("Analysis Complete!")
                    st.write(result)
                except Exception as e:
                    st.error(f"Failed to communicate with GPU backend: {e}")
    else:
        st.info("Awaiting input...")

st.markdown("---")
st.markdown(\"""
### ⚠️ Disclaimer
**This application is an experimental research prototype intended solely for academic and demonstrative purposes.** 
It is **not** a certified medical device and should **not** be used for actual clinical diagnosis, triage, or patient management without the direct supervision of a qualified physician. 
\""")
"""
    with open("/tmp/streamlit_app.py", "w") as f:
        f.write(streamlit_script)
        
    # Start the streamlit server, listening on port 8000 inside the container
    subprocess.Popen([
        "streamlit", "run", "/tmp/streamlit_app.py",
        "--server.port", "8000",
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])
