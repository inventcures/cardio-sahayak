import os
import modal

# 1. Define the Modal App
app = modal.App("cardio-sahayak-gguf-converter")

# 2. Define the Image with all dependencies for conversion
# Requires build-essential and cmake for llama.cpp
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake")
    .pip_install(
        "transformers>=4.45.0",
        "peft>=0.7.0",
        "torch>=2.0.0",
        "accelerate>=0.24.0",
        "huggingface_hub>=0.20.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "numpy",
        "gguf",
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# 3. Define the Conversion Function
# Using A100-80GB to ensure enough VRAM and RAM to merge 27B model
@app.function(
    image=image,
    gpu="A100-80GB", 
    timeout=86400, # Can take a while to clone, build, merge, and convert
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def convert_to_gguf():
    import sys
    import torch
    import subprocess
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi, login

    def run_cmd(cmd, desc):
        print(f"[{desc}] Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {e.stderr}")
            return False

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN secret not found.")
    login(token=hf_token)

    BASE_MODEL = "google/medgemma-27b-it"
    ADAPTER_MODEL = "tp53/cardio-sahayak"
    OUTPUT_REPO = "tp53/cardio-sahayak-gguf"
    
    merged_dir = "/tmp/merged_model"
    os.makedirs(merged_dir, exist_ok=True)

    print("Step 1: Loading base model and merging adapters...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16, # Use float16 for merging
        device_map="auto",
        token=hf_token
    )
    
    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, token=hf_token)
    
    print("Merging and unloading...")
    merged_model = model.merge_and_unload()
    
    print("Saving merged model temporarily...")
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    tokenizer.save_pretrained(merged_dir)
    
    # Free memory
    del model, base_model, merged_model
    torch.cuda.empty_cache()

    print("\nStep 2: Setting up llama.cpp")
    run_cmd(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"], "Clone llama.cpp")
    
    print("\nStep 3: Converting to FP16 GGUF")
    gguf_output_dir = "/tmp/gguf_output"
    os.makedirs(gguf_output_dir, exist_ok=True)
    
    model_name = "cardio-sahayak"
    f16_gguf = f"{gguf_output_dir}/{model_name}-f16.gguf"
    
    convert_script = "/tmp/llama.cpp/convert_hf_to_gguf.py"
    run_cmd([sys.executable, convert_script, merged_dir, "--outfile", f16_gguf, "--outtype", "f16"], "Convert to F16 GGUF")

    print("\nStep 4: Building quantize tool")
    run_cmd(["cmake", "-B", "/tmp/llama.cpp/build", "-S", "/tmp/llama.cpp", "-DGGML_CUDA=OFF"], "CMake Config")
    run_cmd(["cmake", "--build", "/tmp/llama.cpp/build", "--target", "llama-quantize", "-j", "4"], "Build quantize")

    quantize_bin = "/tmp/llama.cpp/build/bin/llama-quantize"

    print("\nStep 5: Quantizing to Q4_K_M (Recommended for 27B on consumer hardware)")
    q4_gguf = f"{gguf_output_dir}/{model_name}-q4_k_m.gguf"
    run_cmd([quantize_bin, f16_gguf, q4_gguf, "Q4_K_M"], "Quantize Q4_K_M")

    print("\nStep 6: Uploading to Hugging Face Hub")
    api = HfApi()
    
    try:
        api.create_repo(repo_id=OUTPUT_REPO, repo_type="model", exist_ok=True, private=True)
    except Exception as e:
        print(f"Repo creation notice: {e}")

    print("Uploading Q4_K_M GGUF...")
    api.upload_file(
        path_or_fileobj=q4_gguf,
        path_in_repo=f"{model_name}-q4_k_m.gguf",
        repo_id=OUTPUT_REPO,
        token=hf_token
    )
    
    print("Uploading FP16 GGUF...")
    api.upload_file(
        path_or_fileobj=f16_gguf,
        path_in_repo=f"{model_name}-f16.gguf",
        repo_id=OUTPUT_REPO,
        token=hf_token
    )

    print("✅ GGUF Conversion and Upload Complete!")

@app.local_entrypoint()
def main():
    convert_to_gguf.remote()
