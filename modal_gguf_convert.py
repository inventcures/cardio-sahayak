import os
import modal

# 1. Define the Modal App
app = modal.App("cardio-sahayak-gguf-converter")

# 2. Define the Image with all dependencies for conversion
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "sed")
    .pip_install(
        "transformers>=4.48.0",
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
@app.function(
    image=image,
    gpu="A100-80GB", 
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def convert_to_gguf():
    import sys
    import torch
    import subprocess
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True
    )
    
    print("Merging and unloading...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, token=hf_token)
    merged_model = model.merge_and_unload()
    
    # Extract language_model for Gemma3
    if hasattr(merged_model, "language_model"):
        print("Extracting language_model...")
        text_model = merged_model.language_model
        text_model.save_pretrained(merged_dir, safe_serialization=True)
        
        text_config = merged_model.config.text_config
        config_dict = text_config.to_dict()
        config_dict["architectures"] = ["Gemma2ForCausalLM"]
        config_dict["model_type"] = "gemma2"
        
        config_path = os.path.join(merged_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    else:
        merged_model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    tokenizer.save_pretrained(merged_dir)
    
    del model, base_model, merged_model
    torch.cuda.empty_cache()

    print("\nStep 2: Setting up llama.cpp")
    run_cmd(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"], "Clone llama.cpp")
    
    print("\nStep 3: Patching llama.cpp for Gemma3 pre-tokenizer...")
    # This sed command replaces the NotImplementedError with a return "gemma"
    patch_script = """
sed -i 's/raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()") /return "gemma" # /' /tmp/llama.cpp/convert_hf_to_gguf.py
"""
    # Note: We use a more flexible regex to catch the line even with slight whitespace variations
    run_cmd(["sed", "-i", 's/raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")/return "gemma"/g', "/tmp/llama.cpp/convert_hf_to_gguf.py"], "Patch llama.cpp vocab")

    print("\nStep 4: Converting to FP16 GGUF")
    gguf_output_dir = "/tmp/gguf_output"
    os.makedirs(gguf_output_dir, exist_ok=True)
    
    model_name = "cardio-sahayak"
    f16_gguf = f"{gguf_output_dir}/{model_name}-f16.gguf"
    
    convert_script = "/tmp/llama.cpp/convert_hf_to_gguf.py"
    run_cmd([sys.executable, convert_script, merged_dir, "--outfile", f16_gguf, "--outtype", "f16"], "Convert to F16 GGUF")

    print("\nStep 5: Building quantize tool")
    run_cmd(["cmake", "-B", "/tmp/llama.cpp/build", "-S", "/tmp/llama.cpp", "-DGGML_CUDA=OFF"], "CMake Config")
    run_cmd(["cmake", "--build", "/tmp/llama.cpp/build", "--target", "llama-quantize", "-j", "4"], "Build quantize")

    quantize_bin = "/tmp/llama.cpp/build/bin/llama-quantize"

    print("\nStep 6: Quantizing to Q4_K_M")
    q4_gguf = f"{gguf_output_dir}/{model_name}-q4_k_m.gguf"
    run_cmd([quantize_bin, f16_gguf, q4_gguf, "Q4_K_M"], "Quantize Q4_K_M")

    print("\nStep 7: Uploading to Hugging Face Hub")
    api = HfApi()
    api.create_repo(repo_id=OUTPUT_REPO, repo_type="model", exist_ok=True, private=False)

    print("Uploading Q4_K_M GGUF...")
    api.upload_file(path_or_fileobj=q4_gguf, path_in_repo=f"{model_name}-q4_k_m.gguf", repo_id=OUTPUT_REPO, token=hf_token)
    
    print("Uploading FP16 GGUF...")
    api.upload_file(path_or_fileobj=f16_gguf, path_in_repo=f"{model_name}-f16.gguf", repo_id=OUTPUT_REPO, token=hf_token)

    print("✅ GGUF Conversion and Upload Complete!")

@app.local_entrypoint()
def main():
    convert_to_gguf.remote()
