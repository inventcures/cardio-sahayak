import time
import os
from datetime import datetime
from huggingface_hub import HfApi

REPO_ID = "tp53/cardio-sahayak-gguf"

def get_status():
    api = HfApi()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        files = api.list_repo_tree(REPO_ID)
        file_list = [f for f in files if f.path.endswith(".gguf")]
        
        if not file_list:
            status = "Merging/Quantizing in progress (No GGUF files yet)."
        else:
            status = " | ".join([f"{f.path}: {f.size/(1024**3):.2f} GB" for f in file_list])
            
        print(f"[{timestamp}] Status for {REPO_ID}: {status}", flush=True)
        
        # Check if both expected files are there and of reasonable size
        if any(f.path.endswith("-q4_k_m.gguf") for f in file_list) and any(f.path.endswith("-f16.gguf") for f in file_list):
            print(f"[{timestamp}] Both GGUF files are uploaded. Script exiting.", flush=True)
            return True
            
    except Exception as e:
        print(f"[{timestamp}] Error polling status: {e}", flush=True)
    return False

if __name__ == "__main__":
    REPO_ID = "tp53/cardio-sahayak-gguf"
    print(f"Starting GGUF poll status script for {REPO_ID}...", flush=True)
    while True:
        if get_status():
            break
        time.sleep(900) # 15 minutes
