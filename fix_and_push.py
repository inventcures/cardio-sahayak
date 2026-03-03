import os
import json
from datasets import Dataset
from huggingface_hub import login

def fix_and_push():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set.")
        return
        
    login(token=hf_token)
    
    input_path = "data/processed_datasets/cardio_sahayak_india_instruct_v2.jsonl"
    repo_id = "tp53/cardio-sahayak-india-instruct-v2"
    
    print(f"Reading and cleaning data from {input_path}...")
    valid_records = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # The file might have multiple JSON objects concatenated or separated by newlines
    # Let's use the robust parsing method
    parts = content.split('}\n{')
    if len(parts) == 1:
        parts = content.split('}{')
        
    for i, part in enumerate(parts):
        if not part.strip(): continue
        if i > 0 and not part.startswith('{'): part = '{' + part
        if i < len(parts) - 1 and not part.endswith('}'): part = part + '}'
        
        try:
            record = json.loads(part)
            # Ensure output is a string, not missing
            if 'output' not in record:
                record['output'] = ""
            valid_records.append(record)
        except json.JSONDecodeError:
            pass
            
    print(f"Recovered {len(valid_records)} valid records.")
    
    # Create huggingface dataset directly from list of dicts to avoid file parsing issues
    print("Creating HF Dataset object...")
    dataset = Dataset.from_list(valid_records)
    
    print(f"Pushing to Hugging Face Hub: {repo_id}...")
    dataset.push_to_hub(repo_id, private=False)
    print("✅ Dataset successfully pushed to Hugging Face Hub!")

if __name__ == "__main__":
    fix_and_push()
