# /// script
# dependencies = ["datasets", "pandas"]
# ///

from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd

def prep_data():
    print("Loading base cardiology dataset in streaming mode...")
    # Using streaming=True to prevent disk overflow
    base_ds_stream = load_dataset("eyone/cardiology_dataset", split="train", streaming=True)
    
    # Take only the first 1,000 samples to stay within memory/time limits
    print("Taking 1,000 samples...")
    base_samples = list(base_ds_stream.take(1000))
    base_ds = Dataset.from_list(base_samples)

    print("Adding Indian-specific Knowledge...")
    indian_knowledge = [
        {
            "messages": [
                {"role": "user", "content": "What are the specific risk factors for Myocardial Infarction (MI) in the Indian population compared to Western populations?"},
                {"role": "assistant", "content": "In the Indian population, MI often occurs at a younger age (5-10 years earlier than in Western populations). Key Indian-specific risk factors include: 1. Lower BMI thresholds (increased risk even at BMI < 25), 2. Higher prevalence of central obesity (increased waist-to-hip ratio), 3. Higher prevalence of metabolic syndrome and Type 2 Diabetes, 4. Genetic predisposition such as the MYBPC3 Δ25bp variant (associated with cardiomyopathy), and 5. Elevated Lipoprotein(a) [Lp(a)] levels which are highly prevalent in South Asians."}
            ]
        },
        {
            "role": "user", "content": "How should a clinician evaluate a 40-year-old Indian male presenting with stable chest pain?",
            "messages": [
                {"role": "user", "content": "How should a clinician evaluate a 40-year-old Indian male presenting with stable chest pain?"},
                {"role": "assistant", "content": "Evaluation should follow the National Consensus on Cardiology guidelines: 1. Conduct a detailed history focusing on early-onset family history of CAD. 2. Perform a 12-lead ECG (monitoring for ST-segment abnormalities). 3. Use lower thresholds for screening; even if BMI is 'normal' by Western standards, check for abdominal obesity. 4. Investigate for 'South Asian Phenotype' markers: low HDL, high triglycerides, and insulin resistance. 5. Consider stress testing or coronary calcium scoring earlier given the aggressive nature of CAD in South Asians."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Discuss the significance of the MYBPC3 Δ25bp variant in Indian cardiology."},
                {"role": "assistant", "content": "The MYBPC3 Δ25bp (a 25-base-pair deletion in the Myosin Binding Protein C3 gene) is a major genetic risk factor uniquely prevalent in about 4% of the Indian population (estimated ~45 million people). This variant is associated with an increased risk of developing heart failure and hypertrophic cardiomyopathy (HCM). Research indicates that while many carriers remain asymptomatic until later in life, environmental or other genetic stressors can trigger severe cardiac dysfunction, necessitating early screening and long-term monitoring for carriers in India."}
            ]
        }
    ]

    custom_ds = Dataset.from_list(indian_knowledge)

    def format_base(example):
        q = example.get('question', example.get('instruction', 'Describe cardiology best practices.'))
        a = example.get('answer', example.get('output', 'Follow standard clinical protocols.'))
        return {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        }

    print("Formatting and Combining...")
    formatted_base = base_ds.map(format_base, remove_columns=base_ds.column_names)
    final_ds = concatenate_datasets([formatted_base, custom_ds])
    
    final_ds = final_ds.shuffle(seed=42)
    final_ds = final_ds.train_test_split(test_size=0.1)
    
    print(f"Final dataset: {len(final_ds['train'])} train, {len(final_ds['test'])} test.")
    return final_ds

if __name__ == "__main__":
    ds = prep_data()
    # Using the token for push
    ds.push_to_hub("tp53/cardio-sahayak-india-instruct-v0")
    print("Dataset pushed successfully!")
