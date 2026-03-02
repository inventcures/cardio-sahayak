from huggingface_hub import HfApi
import os

api = HfApi()

space_id = "tp53/cardio-sahayak-demo"

# Create the space if it doesn't exist
try:
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False
    )
    print(f"Space {space_id} created or already exists.")
except Exception as e:
    print(f"Failed to create space: {e}")

# Upload the files
try:
    print("Uploading app.py...")
    api.upload_file(
        path_or_fileobj="app.py",
        path_in_repo="app.py",
        repo_id=space_id,
        repo_type="space"
    )
    
    print("Uploading requirements.txt...")
    api.upload_file(
        path_or_fileobj="requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=space_id,
        repo_type="space"
    )
    
    print("Uploading examples...")
    api.upload_folder(
        folder_path="examples",
        path_in_repo="examples",
        repo_id=space_id,
        repo_type="space"
    )
    
    print(f"Successfully deployed to https://huggingface.co/spaces/{space_id}")
except Exception as e:
    print(f"Failed to upload files: {e}")
