import os
import modal

# 1. Define the Modal App
app = modal.App("cardio-sahayak-eval")

# 2. Define the Image with all dependencies for evaluation
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "torch",
        "huggingface_hub",
        "lm-eval[api]", # EleutherAI LM Evaluation Harness
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# 3. Define the Evaluation Function
@app.function(
    image=image,
    gpu="A100-80GB", # Need significant VRAM for 27B model
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate_model():
    import torch
    import json
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from huggingface_hub import login
    import os

    # Ensure HF Token is set
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN secret not found.")
    login(token=hf_token)

    base_model_id = "google/medgemma-27b-it"
    adapter_model_id = "tp53/cardio-sahayak"

    print(f"Loading model for evaluation: {base_model_id} + {adapter_model_id}")
    
    # Initialize the HFLM model wrapper for lm-eval
    # We load the base model and specify the PEFT adapter
    lm = HFLM(
        pretrained=base_model_id,
        peft=adapter_model_id,
        device="cuda",
        dtype="bfloat16",
        # Using 4-bit quantization to fit the 27B model on a single 80GB GPU during inference
        model_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"}
    )

    # We will evaluate on MedQA (USMLE style questions) and PubMedQA
    # Note: For MedQA, the task name in lm_eval is 'medqa_4options'
    # For PubMedQA, it's 'pubmedqa'
    tasks = ["medqa_4options", "pubmedqa"]

    print(f"Starting evaluation on tasks: {tasks}")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size="auto",
        device="cuda"
    )

    print("
" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Extract and print key metrics
    for task_name in tasks:
        if task_name in results["results"]:
            task_results = results["results"][task_name]
            acc = task_results.get("acc,none", "N/A")
            acc_norm = task_results.get("acc_norm,none", "N/A")
            
            print(f"
Task: {task_name.upper()}")
            print(f"Accuracy: {acc}")
            print(f"Normalized Accuracy: {acc_norm}")
    
    print("="*50)
    
    # Save full results to a file
    with open("eval_results.json", "w") as f:
        json.dump(results["results"], f, indent=4)
        
    print("
Full results saved to eval_results.json")

@app.local_entrypoint()
def main():
    evaluate_model.remote()
