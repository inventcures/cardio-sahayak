import os
import json
import time

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Please install google-genai: pip install google-genai")
    exit(1)

def shift_phenotype(input_file="data/raw_datasets/western_vignettes_sample.jsonl", output_file="data/processed_datasets/synthetic_indian_vignettes.jsonl"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set it before running this script.")
        # Create a mock output for demonstration
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"original": "50yo white male, BMI 28, chest pain.", "shifted": "42yo Indian male, BMI 24 (central obesity noted), chest pain. Consider MYBPC3 delta25bp screening based on family history."}) + "\\n")
        return

    client = genai.Client(api_key=api_key)
    
    # Prompt instructing the model to perform the phenotype shift
    system_instruction = """
    You are an expert cardiologist specializing in Cardiovascular Disease (CVD) and South Asian demographics. 
    Your task is to take a general or Western clinical vignette and 'shift' the phenotype to match a South Asian (specifically Indian) cardiology patient profile.
    
    Key cardiology adjustments to make:
    1. Decrease the age of onset for ischemic events (MI, CAD) by 5-10 years to reflect early-onset risks.
    2. Adjust standard BMI down by 3-4 points but explicitly note central adiposity, metabolic syndrome, or insulin resistance indicators as severe CVD risk factors.
    3. Inject relevant cardiac family history, specifically mentioning the MYBPC3 Δ25bp variant if hypertrophic cardiomyopathy (HCM) or heart failure is suspected.
    4. Mention elevated Lipoprotein(a) [Lp(a)] levels if lipid profiles are discussed.
    5. Ensure the clinical notes reflect Indian National Consensus guidelines on CVD management where applicable.
    
    Return ONLY the shifted cardiology clinical vignette.
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check if input file exists, if not create a mock one for testing
    if not os.path.exists(input_file):
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        with open(input_file, "w") as f:
            f.write(json.dumps({"text": "Patient is a 55-year-old Caucasian male presenting with acute chest pain. BMI is 29. No significant family history of early heart disease."}) + "\\n")
            f.write(json.dumps({"text": "60-year-old female with suspected hypertrophic cardiomyopathy. Echo shows septal hypertrophy. BMI 26."}) + "\\n")
            
    print(f"Reading from {input_file}...")
    
    with open(input_file, "r") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
                
            data = json.loads(line)
            original_text = data.get("text", "")
            
            if not original_text:
                continue
                
            print(f"Processing vignette...")
            
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=original_text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.4,
                    )
                )
                
                shifted_text = response.text.strip()
                
                output_data = {
                    "original": original_text,
                    "shifted": shifted_text,
                    "source": "synthetic_phenotype_shift"
                }
                
                outfile.write(json.dumps(output_data) + "\\n")
                # Sleep briefly to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Failed to generate content: {e}")

    print(f"Finished generating synthetic vignettes. Saved to {output_file}")

if __name__ == "__main__":
    shift_phenotype()
