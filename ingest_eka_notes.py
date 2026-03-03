import os
from datasets import load_dataset, Dataset
import json

def format_eka_dataset():
    print("Downloading ekacare/clinical_note_generation_dataset...")
    hf_token = os.environ.get("HF_TOKEN")
    
    try:
        dataset = load_dataset("ekacare/clinical_note_generation_dataset", split="test", token=hf_token)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\n*** IMPORTANT ***")
        print("This dataset is gated. Please visit: https://huggingface.co/datasets/ekacare/clinical_note_generation_dataset")
        print("and accept the terms of use. Then re-run this script.")
        print("Creating a mock entry so the pipeline doesn't break...")
        dataset = [{"dialogue": "Patient complains of chest pain. BMI is 24.", "clinical_note": "Patient presents with chest pain. Note South Asian phenotype risk despite normal BMI."}]
    
    formatted_data = []
    
    print(f"Processing {len(dataset)} records...")
    for idx, item in enumerate(dataset):
        # The dataset typically contains the conversation/transcript and the target clinical note
        # We format this into an instruction for Cardio-Sahayak
        
        # We need to inspect the actual column names first, but commonly it's 'transcript' or 'dialogue' and 'clinical_note'
        # Let's assume standard keys for now and handle potential variations
        
        # Print the first item to understand structure if needed
        if idx == 0:
            if hasattr(dataset, 'column_names'):
                print("Dataset columns:", dataset.column_names)
            else:
                print("Dataset columns:", list(item.keys()))
            
        # Extracting relevant fields based on the dataset's structure
        transcript = item.get('text', '')
        prompt_schema = item.get('sample_prompt', '')
        
        if not transcript:
            continue
            
        instruction = f"Analyze the following medical consultation transcript and generate a structured clinical note suitable for an Indian cardiology setting. Focus specifically on Cardiovascular Disease (CVD) indicators, early-onset risks, metabolic syndrome markers, and relevant cardiac family history. Pay attention to specific local terminologies or conditions if present.\\n\\nTranscript:\\n{transcript}\\n\\nTarget Schema Context:\\n{prompt_schema}"
        
        formatted_item = {
            "instruction": instruction,
            "output": "", # Will be synthetically generated later or left empty if unsupervised
            "source": "ekacare_clinical_notes",
            "original_transcript": transcript
        }
        
        formatted_data.append(formatted_item)
        
    print(f"Successfully formatted {len(formatted_data)} records.")
    
    # Save to local JSONL for later aggregation
    output_dir = "data/raw_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "eka_formatted_notes.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved formatted dataset to {output_path}")

if __name__ == "__main__":
    format_eka_dataset()
