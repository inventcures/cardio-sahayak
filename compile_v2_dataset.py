import os
import glob
import json
import time

def compile_v2():
    print("Initiating V2 Dataset Compilation...")
    
    output_dir = "data/processed_datasets"
    os.makedirs(output_dir, exist_ok=True)
    v2_output_path = os.path.join(output_dir, "cardio_sahayak_india_instruct_v2.jsonl")
    
    compiled_data = []
    
    def load_jsonl_robust(filepath):
        results = []
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            # Replace literal \n with actual newline for splitting
            content = content.replace('\\n', '\n')
            # Split by common JSON object boundaries
            parts = content.split('}\n{')
            if len(parts) == 1:
                parts = content.split('}{')
            
            for i, part in enumerate(parts):
                if not part.strip(): continue
                # Reconstruct the valid JSON string
                if i > 0: part = '{' + part
                if i < len(parts) - 1: part = part + '}'
                
                try:
                    results.append(json.loads(part))
                except json.JSONDecodeError as e:
                    pass
        return results

    # 1. Ingest Eka Clinical Notes
    eka_path = "data/raw_datasets/eka_formatted_notes.jsonl"
    if os.path.exists(eka_path):
        print("Merging Eka Structured Clinical Notes...")
        eka_data = load_jsonl_robust(eka_path)
        compiled_data.extend(eka_data)
                    
    # 2. Ingest Synthetic Indian Phenotype Vignettes
    synthetic_path = "data/processed_datasets/synthetic_indian_vignettes.jsonl"
    if os.path.exists(synthetic_path):
        print("Merging Synthesized South Asian Vignettes (Gemini)...")
        synth_data = load_jsonl_robust(synthetic_path)
        for item in synth_data:
            if 'shifted' in item:
                instruction = f"Analyze the following patient profile and identify key Cardiovascular Disease (CVD) risks specific to the South Asian phenotype:\n\n{item['shifted']}"
                compiled_data.append({
                    "instruction": instruction,
                    "output": "", # Target output to be generated or trained as unsupervised SFT
                    "source": "synthetic_phenotype_shift"
                })
                    
    # 3. Reference Multimodal Sources (IIIT, ScienceOpen, MIMIC-IV)
    print("Integrating references to Multimodal ECG datasets...")
    multimodal_sources = [
        "data/raw_datasets/iiit_ecg_mock/metadata.jsonl",
        "data/raw_datasets/scienceopen_ecg_mock/metadata.jsonl",
        "data/raw_datasets/mimic_iv_ecg_sample/metadata.jsonl",
        "data/raw_datasets/meeti_mock/metadata.jsonl"
    ]
    
    for mm_path in multimodal_sources:
        if os.path.exists(mm_path):
            print(f"Linking multimodal source: {mm_path}")
            mm_data = load_jsonl_robust(mm_path)
            for record in mm_data:
                desc = record.get("diagnosis") or record.get("desc") or record.get("text_interpretation") or "ECG Analysis"
                compiled_data.append({
                    "instruction": f"Generate a detailed clinical report based on the following preliminary findings: {desc}",
                    "output": "", 
                    "source": f"multimodal_{os.path.basename(os.path.dirname(mm_path))}",
                    "reference_image": record.get("url") or record.get("image") or record.get("waveform_path")
                })
                        
    # Write Final Dataset
    print(f"Writing {len(compiled_data)} total records to V2 Dataset...")
    with open(v2_output_path, "w", encoding="utf-8") as f:
        for item in compiled_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Compilation Complete! V2 Dataset ready at: {v2_output_path}")
    print("\nNext Steps:")
    print("1. Push this dataset to Hugging Face: tp53/cardio-sahayak-india-instruct-v2")
    print("2. Re-run modal_train_vlm_cardio_sahayak.py referencing the new dataset.")

if __name__ == "__main__":
    compile_v2()
