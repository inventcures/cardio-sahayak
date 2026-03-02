from huggingface_hub import HfApi
import os

api = HfApi()
space_id = "tp53/cardio-sahayak-demo"
hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    print("HF_TOKEN environment variable is not set!")
else:
    try:
        api.add_space_secret(repo_id=space_id, key="HF_TOKEN", value=hf_token)
        print("Successfully added HF_TOKEN secret to the space.")
        
        # It's also usually necessary to restart the space so it picks up the new environment variable
        api.restart_space(repo_id=space_id)
        print("Restarted the space to apply the secret.")
    except Exception as e:
        print(f"Failed to add secret or restart space: {e}")
