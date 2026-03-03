import os
from datasets import load_dataset
from huggingface_hub import login

def push_to_hub():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set.")
        return
        
    login(token=hf_token)
    
    dataset_path = "data/processed_datasets/cardio_sahayak_india_instruct_v2.jsonl"
    repo_id = "tp53/cardio-sahayak-india-instruct-v2"
    
    print(f"Loading local dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    print(f"Pushing to Hugging Face Hub: {repo_id}...")
    dataset.push_to_hub(repo_id, private=False)
    print("✅ Dataset successfully pushed to Hugging Face Hub!")

if __name__ == "__main__":
    push_to_hub()
