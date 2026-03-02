from huggingface_hub import HfApi, ModelCard, ModelCardData
import os

api = HfApi()

repos = [
    "tp53/cardio-sahayak",
    "tp53/cardio-sahayak-vlm",
    "tp53/cardio-sahayak-gguf"
]

for repo_id in repos:
    try:
        # Create repo if it doesn't exist (just in case)
        try:
            api.create_repo(repo_id, exist_ok=True)
        except Exception:
            pass

        # Update visibility to public
        api.update_repo_settings(repo_id=repo_id, private=False)
        print(f"Set {repo_id} to public.")
        
        # Update Model Card for license
        try:
            card = ModelCard.load(repo_id)
        except Exception:
            card = ModelCard("")
            
        # Ensure cc-by-4.0 is set in metadata
        if card.data is None:
            card.data = ModelCardData()
            
        card.data.license = "cc-by-4.0"
        
        # Add explicit attribution clause if not present
        attribution_text = """

## License & Attribution
This model is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

**You must provide appropriate credit** to the `inventcures/cardio-sahayak` project and this Hugging Face repository for any kind of academic, non-commercial, or commercial use.
"""
        
        if "## License" not in card.text:
            card.text += attribution_text
            
        card.push_to_hub(repo_id)
        print(f"Updated license and README for {repo_id}.")
        
    except Exception as e:
        print(f"Failed to update {repo_id}: {e}")
