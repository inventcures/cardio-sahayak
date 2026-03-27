"""
Synthetic Indian clinical vignette generator.

Generates structured clinical scenarios covering all Diamond Approach
comorbidity combinations with South Asian phenotype adjustments.
Uses LLM (Gemini/Claude) for natural language generation with
structured seed templates.
"""
import json
import os
import random
from pathlib import Path
from dataclasses import dataclass, field


SEED_CONDITIONS = [
    "Acute Coronary Syndrome (STEMI)",
    "Acute Coronary Syndrome (NSTEMI)",
    "Chronic Stable Angina",
    "Hypertrophic Cardiomyopathy (HCM)",
    "Dilated Cardiomyopathy (DCM)",
    "Atrial Fibrillation",
    "Heart Failure with Reduced EF (HFrEF)",
    "Heart Failure with Preserved EF (HFpEF)",
    "Valvular Heart Disease - Mitral Regurgitation",
    "Valvular Heart Disease - Aortic Stenosis",
    "Pulmonary Hypertension",
    "Acute Myocarditis",
    "Pericardial Effusion / Tamponade",
    "Microvascular Angina",
]

COMORBIDITY_COMBINATIONS = [
    ["diabetes_mellitus", "hypertension"],
    ["diabetes_mellitus", "hypertension", "dyslipidemia"],
    ["diabetes_mellitus", "ckd"],
    ["hypertension", "atrial_fibrillation"],
    ["diabetes_mellitus", "heart_failure"],
    ["hypertension", "lv_dysfunction"],
    ["smoking", "family_history_premature_cad"],
    ["diabetes_mellitus", "hypertension", "ckd", "dyslipidemia"],
    ["obesity_central", "diabetes_mellitus"],
    ["tobacco_gutka", "hypertension"],
]

SOUTH_ASIAN_DEMOGRAPHIC_TEMPLATES = [
    {"age_range": (35, 50), "gender": "male", "bmi_range": (23.5, 28.0), "setting": "urban"},
    {"age_range": (40, 55), "gender": "male", "bmi_range": (22.0, 26.0), "setting": "rural"},
    {"age_range": (45, 60), "gender": "female", "bmi_range": (24.0, 30.0), "setting": "urban"},
    {"age_range": (50, 65), "gender": "male", "bmi_range": (21.0, 25.0), "setting": "semi-urban"},
    {"age_range": (30, 45), "gender": "male", "bmi_range": (25.0, 30.0), "setting": "urban"},
    {"age_range": (55, 70), "gender": "female", "bmi_range": (23.0, 27.0), "setting": "rural"},
]


@dataclass
class VignetteSeed:
    condition: str
    comorbidities: list[str]
    age: int
    gender: str
    bmi: float
    setting: str
    south_asian_flags: list[str] = field(default_factory=list)


@dataclass
class GeneratedVignette:
    seed: VignetteSeed
    clinical_text: str
    instruction: str
    expected_output: str
    source: str = "synthetic_indian_vignette"


def create_seed(condition: str, comorbidities: list[str]) -> VignetteSeed:
    template = random.choice(SOUTH_ASIAN_DEMOGRAPHIC_TEMPLATES)
    age = random.randint(*template["age_range"])
    bmi = round(random.uniform(*template["bmi_range"]), 1)

    flags = []
    if bmi >= 23.0:
        flags.append(f"BMI {bmi} exceeds South Asian threshold (>=23 vs >=25 Western)")
    if age < 55 and template["gender"] == "male":
        flags.append("Premature CAD risk window for South Asian male")
    if "diabetes_mellitus" in comorbidities:
        flags.append("50-60% CAD-DM overlap in India")
    if random.random() < 0.15:
        flags.append("MYBPC3 Delta-25bp carrier (4% South Asian prevalence)")
    if random.random() < 0.3:
        flags.append("Elevated Lp(a) - independent CV risk factor in South Asians")

    return VignetteSeed(
        condition=condition,
        comorbidities=comorbidities,
        age=age,
        gender=template["gender"],
        bmi=bmi,
        setting=template["setting"],
        south_asian_flags=flags,
    )


def build_generation_prompt(seed: VignetteSeed) -> str:
    return f"""Generate a detailed Indian clinical vignette for a cardiology case:

Patient: {seed.age}-year-old {seed.gender}, BMI {seed.bmi} kg/m2
Setting: {seed.setting} India
Primary condition: {seed.condition}
Comorbidities: {', '.join(seed.comorbidities)}
South Asian phenotype flags: {', '.join(seed.south_asian_flags) if seed.south_asian_flags else 'None'}

Generate:
1. A presenting complaint and clinical history (3-4 sentences)
2. Relevant vitals and examination findings
3. Key laboratory values (include lipid panel, HbA1c if diabetic, troponin if ACS, BNP if HF)
4. ECG findings (if relevant)
5. A clinical question asking for diagnosis, risk stratification, and management plan

Use Indian drug names and formulations (e.g., atorvastatin, ramipril, metoprolol).
Reference Indian Consensus guidelines where appropriate.
Apply South Asian BMI thresholds (>=23 overweight, >=27.5 obese).
Mention MYBPC3 screening if HCM or family history of cardiomyopathy.

Output the vignette as JSON with keys: "clinical_history", "vitals", "labs", "ecg_findings", "question"
"""


def generate_all_seeds(
    max_per_combination: int = 3,
) -> list[VignetteSeed]:
    seeds = []
    for condition in SEED_CONDITIONS:
        for combos in COMORBIDITY_COMBINATIONS:
            for _ in range(max_per_combination):
                seeds.append(create_seed(condition, combos))
    return seeds


def save_seeds(seeds: list[VignetteSeed], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for seed in seeds:
            f.write(json.dumps({
                "condition": seed.condition,
                "comorbidities": seed.comorbidities,
                "age": seed.age,
                "gender": seed.gender,
                "bmi": seed.bmi,
                "setting": seed.setting,
                "south_asian_flags": seed.south_asian_flags,
            }) + "\n")
    print(f"Saved {len(seeds)} seeds to {output_path}")


def save_vignettes(vignettes: list[GeneratedVignette], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for v in vignettes:
            f.write(json.dumps({
                "instruction": v.instruction,
                "output": v.expected_output,
                "clinical_text": v.clinical_text,
                "condition": v.seed.condition,
                "comorbidities": v.seed.comorbidities,
                "age": v.seed.age,
                "gender": v.seed.gender,
                "bmi": v.seed.bmi,
                "south_asian_flags": v.seed.south_asian_flags,
                "source": v.source,
            }) + "\n")
    print(f"Saved {len(vignettes)} vignettes to {output_path}")
