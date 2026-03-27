import json
import random
from pathlib import Path


QUESTION_TEMPLATES = {
    "acute_presentation": [
        "A {age}-year-old {gender} Indian patient presents to the emergency department with {symptom} for {duration}. {vitals}. ECG shows {ecg_finding}. {labs}. What is the diagnosis, risk stratification per Indian Consensus guidelines, and immediate management plan?",
        "A {age}-year-old {gender} from {setting} India presents with {symptom}. Past history: {history}. {vitals}. What are the differential diagnoses and urgent investigations needed?",
    ],
    "chronic_management": [
        "A {age}-year-old {gender} Indian patient with known {condition} on {medications} presents for routine follow-up. {vitals}. {labs}. Using the Diamond Approach (IJAM 2023), assess current therapy and recommend adjustments.",
        "A {age}-year-old {gender} with {condition} and {comorbidity} reports {symptom}. Current medications: {medications}. Evaluate treatment adequacy per Indian Consensus guidelines and suggest modifications.",
    ],
    "risk_stratification": [
        "A {age}-year-old {gender} Indian patient with {risk_factors} is screened at a PHC. {vitals}. No symptoms currently. Calculate the comorbidity risk score (IJAM Table 2) and recommend appropriate next steps including referral decision.",
        "An ASHA worker screens a {age}-year-old {gender} in rural India. Findings: {screening_findings}. Using the CHW screening protocol, determine risk level (RED/YELLOW/GREEN) and referral action.",
    ],
    "diabetic_cardiac": [
        "A {age}-year-old {gender} Indian diabetic (HbA1c {hba1c}%) with {cardiac_condition} presents with {symptom}. Current: {medications}. {labs}. Provide integrated cardio-metabolic management per Indian Consensus. Address SGLT2i/GLP-1RA indication.",
        "A {age}-year-old {gender} with type 2 diabetes and newly diagnosed {cardiac_condition}. {vitals}. LDL {ldl} mg/dl. What are the treatment targets per Indian Consensus and what medications should be initiated?",
    ],
    "women_cardiac": [
        "A {age}-year-old Indian woman presents with {atypical_symptom}. {risk_factors}. Her stress ECG is equivocal. Per Indian Consensus, how should women with CCS be evaluated differently? What investigation is preferred?",
    ],
    "patient_education": [
        "A {age}-year-old Indian patient diagnosed with {condition} asks: '{patient_question}'. Provide an explanation suitable for a patient with limited health literacy. Include lifestyle guidance specific to Indian context (diet, exercise, tobacco).",
    ],
    "resource_limited": [
        "A {age}-year-old {gender} presents to a rural PHC with suspected {condition}. No echo, no cath lab, no cardiologist available. Nearest district hospital is {distance} km. What can be done at PHC level and when should transfer be arranged?",
    ],
    "hcm_cardiomyopathy": [
        "A {age}-year-old Indian {gender} with family history of sudden cardiac death is found to have {finding}. Should MYBPC3 Delta-25bp screening be recommended? What is the significance in South Asian populations? Outline management and family cascade screening.",
    ],
    "heart_failure": [
        "A {age}-year-old {gender} with LVEF {lvef}%, {symptoms}, BNP {bnp} pg/ml. {comorbidities}. Using the Diamond Approach, select appropriate antianginal and heart failure therapy. Consider which drugs are contraindicated.",
    ],
    "atrial_fibrillation": [
        "A {age}-year-old {gender} Indian patient with {af_type} atrial fibrillation and {comorbidity}. Heart rate {hr} bpm. Per Diamond Approach, which rate control agents are preferred and which are contraindicated?",
    ],
}

DEMOGRAPHICS = {
    "ages": list(range(35, 75)),
    "genders": ["male", "female"],
    "settings": ["rural", "semi-urban", "urban"],
    "symptoms": [
        "crushing chest pain radiating to left arm",
        "exertional chest discomfort for 3 months",
        "progressive breathlessness on exertion",
        "sudden onset palpitations",
        "syncope while climbing stairs",
        "bilateral pedal edema worsening over 2 weeks",
        "episodic chest tightness relieved by rest",
    ],
    "atypical_symptoms": [
        "epigastric discomfort and fatigue",
        "jaw pain and nausea on exertion",
        "unexplained dyspnea and fatigue",
        "interscapular pain with lightheadedness",
    ],
    "ecg_findings": [
        "ST elevation in V1-V4",
        "ST depression in II, III, aVF",
        "new onset atrial fibrillation with rapid ventricular response",
        "left bundle branch block",
        "T-wave inversions in V4-V6",
        "normal sinus rhythm",
        "LVH by voltage criteria",
    ],
    "conditions": [
        "chronic stable angina",
        "hypertension",
        "dilated cardiomyopathy",
        "hypertrophic cardiomyopathy",
        "mitral regurgitation",
        "aortic stenosis",
        "heart failure with reduced EF",
    ],
    "comorbidities_text": [
        "diabetes mellitus and hypertension",
        "diabetes, CKD stage 3, and dyslipidemia",
        "hypertension and atrial fibrillation",
        "diabetes and obesity (BMI 28)",
        "COPD and hypertension",
    ],
    "medications_text": [
        "metoprolol 50mg BD, atorvastatin 40mg, aspirin 75mg",
        "ramipril 5mg, amlodipine 5mg, metformin 1g BD",
        "bisoprolol 5mg, telmisartan 40mg, rosuvastatin 20mg",
        "enalapril 10mg, furosemide 40mg, spironolactone 25mg",
    ],
    "patient_questions": [
        "What does my heart risk level mean and what should I do?",
        "Why do I need to take so many medicines? Can I stop some?",
        "What foods should I avoid for my heart condition?",
        "When should I go to the hospital immediately?",
        "Is my heart disease related to my diabetes?",
        "Can I do exercise with my heart condition?",
    ],
    "risk_factors_text": [
        "diabetes, hypertension, current bidi smoker",
        "family history of MI at age 48, dyslipidemia",
        "diabetes, central obesity (waist 98cm), sedentary",
        "hypertension, gutka user, family Hx premature CAD",
    ],
}


def _fill_template(template: str) -> str:
    d = DEMOGRAPHICS
    replacements = {
        "age": str(random.choice(d["ages"])),
        "gender": random.choice(d["genders"]),
        "setting": random.choice(d["settings"]),
        "symptom": random.choice(d["symptoms"]),
        "atypical_symptom": random.choice(d["atypical_symptoms"]),
        "duration": random.choice(["30 minutes", "2 hours", "6 hours", "since morning"]),
        "vitals": f"BP {random.randint(110, 180)}/{random.randint(70, 100)}, HR {random.randint(50, 120)}, SpO2 {random.randint(88, 99)}%",
        "ecg_finding": random.choice(d["ecg_findings"]),
        "labs": f"Troponin {'elevated' if random.random() > 0.5 else 'pending'}, LDL {random.randint(80, 200)} mg/dl",
        "condition": random.choice(d["conditions"]),
        "comorbidity": random.choice(d["comorbidities_text"]),
        "medications": random.choice(d["medications_text"]),
        "history": random.choice(d["comorbidities_text"]),
        "risk_factors": random.choice(d["risk_factors_text"]),
        "screening_findings": f"BP {random.randint(130, 180)}/{random.randint(80, 110)}, diabetic, tobacco user, BMI {random.uniform(23, 32):.1f}",
        "hba1c": f"{random.uniform(7.0, 11.0):.1f}",
        "cardiac_condition": random.choice(["CAD", "heart failure", "atrial fibrillation"]),
        "ldl": str(random.randint(100, 220)),
        "patient_question": random.choice(d["patient_questions"]),
        "distance": str(random.choice([30, 50, 80, 120])),
        "finding": random.choice(["asymmetric septal hypertrophy on echo", "LVH on ECG with family history of HCM"]),
        "lvef": str(random.choice([20, 25, 30, 35, 40])),
        "symptoms": random.choice(["NYHA class III dyspnea", "orthopnea and PND", "exertional breathlessness"]),
        "bnp": str(random.choice([500, 800, 1200, 2000, 3500])),
        "comorbidities": random.choice(["diabetes and CKD", "HTN and diabetes", "AF and diabetes"]),
        "af_type": random.choice(["paroxysmal", "persistent", "permanent"]),
        "hr": str(random.randint(90, 150)),
    }

    result = template
    for key, value in replacements.items():
        result = result.replace("{" + key + "}", value)
    return result


def generate_cardioqa_v2(
    output_path: str | Path,
    target_count: int = 200,
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    category_counts = {
        "acute_presentation": 30,
        "chronic_management": 25,
        "risk_stratification": 30,
        "diabetic_cardiac": 25,
        "women_cardiac": 10,
        "patient_education": 20,
        "resource_limited": 15,
        "hcm_cardiomyopathy": 10,
        "heart_failure": 20,
        "atrial_fibrillation": 15,
    }

    questions = []
    for category, count in category_counts.items():
        templates = QUESTION_TEMPLATES.get(category, [])
        if not templates:
            continue
        for i in range(count):
            template = random.choice(templates)
            question_text = _fill_template(template)
            questions.append({
                "id": f"cqa2_{category}_{i:03d}",
                "question": question_text,
                "category": category,
            })

    random.shuffle(questions)

    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    print(f"Generated {len(questions)} CardioQA-India v2 questions -> {output_path}")
    return len(questions)


if __name__ == "__main__":
    generate_cardioqa_v2("data/benchmarks/cardioqa_india_v2_questions.jsonl")
