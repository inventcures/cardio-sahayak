import json
import random
from pathlib import Path

from src.knowledge.schemas import (
    ClinicalCondition, DrugClass, DRUG_CLASS_NAMES,
    ChestPainPrecipitant,
)
from src.knowledge.diamond_approach import DIAMOND_TABLE


def generate_diamond_approach_mcqs() -> list[dict]:
    questions = []

    condition_names = {
        ClinicalCondition.HIGH_HR: "a patient with heart rate >=70 bpm",
        ClinicalCondition.BRADYCARDIA: "a patient with bradycardia",
        ClinicalCondition.HYPERTENSION: "a hypertensive patient",
        ClinicalCondition.HYPOTENSION: "a hypotensive patient",
        ClinicalCondition.LV_DYSFUNCTION: "a patient with LV dysfunction",
        ClinicalCondition.HEART_FAILURE: "a patient with heart failure",
        ClinicalCondition.ATRIAL_FIBRILLATION: "a patient with atrial fibrillation",
    }

    for condition, (preferred, _, contraindicated) in DIAMOND_TABLE.items():
        name = condition_names[condition]

        if preferred and contraindicated:
            correct_drug = random.choice(list(preferred))
            wrong_drugs = random.sample(
                list(contraindicated),
                min(3, len(contraindicated))
            )

            options = [DRUG_CLASS_NAMES[correct_drug]] + [
                DRUG_CLASS_NAMES[d] for d in wrong_drugs
            ]
            random.shuffle(options)
            correct_idx = options.index(DRUG_CLASS_NAMES[correct_drug])

            questions.append({
                "question": (
                    f"Per the Diamond Approach (IJAM 2023), which antianginal drug class "
                    f"is PREFERRED for {name} with chronic coronary syndrome?"
                ),
                "options": options,
                "correct_answer": chr(65 + correct_idx),
                "category": "diamond_approach",
                "difficulty": "medium",
                "guideline_source": "IJAM 2023 Table 3",
            })

            correct_ci = random.choice(list(contraindicated))
            safe_drugs = random.sample(
                list(preferred),
                min(3, len(preferred))
            )

            options2 = [DRUG_CLASS_NAMES[correct_ci]] + [
                DRUG_CLASS_NAMES[d] for d in safe_drugs
            ]
            random.shuffle(options2)
            correct_idx2 = options2.index(DRUG_CLASS_NAMES[correct_ci])

            questions.append({
                "question": (
                    f"Per the Diamond Approach (IJAM 2023), which drug class is "
                    f"CONTRAINDICATED for {name}?"
                ),
                "options": options2,
                "correct_answer": chr(65 + correct_idx2),
                "category": "diamond_approach",
                "difficulty": "medium",
                "guideline_source": "IJAM 2023 Table 3",
            })

    return questions


def generate_risk_stratification_mcqs() -> list[dict]:
    questions = [
        {
            "question": "Per ESC 2019 guidelines adapted for India, what is the annual CV mortality threshold for HIGH risk classification?",
            "options": [">3%", ">1%", ">5%", ">10%"],
            "correct_answer": "A",
            "category": "risk_stratification",
            "difficulty": "easy",
            "guideline_source": "ESC 2019 / IJAM 2023",
        },
        {
            "question": "A 48-year-old Indian male with LVEF 30% and diabetes. What is the risk category?",
            "options": ["HIGH", "INTERMEDIATE", "LOW", "Cannot determine"],
            "correct_answer": "A",
            "category": "risk_stratification",
            "difficulty": "easy",
            "guideline_source": "ESC 2019 / IJAM 2023",
        },
        {
            "question": "What is the LDL-C target for an Indian patient with both CAD and diabetes per Indian Consensus?",
            "options": ["<70 mg/dl", "<100 mg/dl", "<130 mg/dl", "<55 mg/dl"],
            "correct_answer": "A",
            "category": "treatment_targets",
            "difficulty": "medium",
            "guideline_source": "IJAM 2023",
        },
        {
            "question": "What is the BMI threshold for overweight classification in South Asians?",
            "options": [">=23 kg/m2", ">=25 kg/m2", ">=27.5 kg/m2", ">=30 kg/m2"],
            "correct_answer": "A",
            "category": "south_asian_phenotype",
            "difficulty": "easy",
            "guideline_source": "WHO South Asian guidelines",
        },
        {
            "question": "What percentage of CAD patients in India also have diabetes?",
            "options": ["50-60%", "20-30%", "10-15%", "70-80%"],
            "correct_answer": "A",
            "category": "indian_epidemiology",
            "difficulty": "medium",
            "guideline_source": "IJAM 2023",
        },
        {
            "question": "Which genetic variant should be screened in South Asian patients with HCM or unexplained heart failure?",
            "options": ["MYBPC3 Delta-25bp", "BRCA1", "CFTR", "HFE C282Y"],
            "correct_answer": "A",
            "category": "south_asian_phenotype",
            "difficulty": "hard",
            "guideline_source": "South Asian genetics literature",
        },
        {
            "question": "Per IJAM 2023 Chest Pain Scoring, exertional chest pain relieved by rest scores how many points?",
            "options": ["3 points", "1 point", "2 points", "0 points"],
            "correct_answer": "A",
            "category": "chest_pain_scoring",
            "difficulty": "medium",
            "guideline_source": "IJAM 2023 Table 1",
        },
        {
            "question": "For diabetic CCS patients, the Indian Consensus recommends which BP target?",
            "options": ["<=130/80 mmHg", "<=140/90 mmHg", "<=120/70 mmHg", "<=150/95 mmHg"],
            "correct_answer": "A",
            "category": "treatment_targets",
            "difficulty": "medium",
            "guideline_source": "IJAM 2023",
        },
        {
            "question": "Which drug class is recommended for diabetic patients with CVD per Indian Consensus (for cardiac + renal benefits)?",
            "options": ["SGLT2 inhibitors", "Sulfonylureas", "Thiazolidinediones", "DPP-4 inhibitors"],
            "correct_answer": "A",
            "category": "treatment_targets",
            "difficulty": "medium",
            "guideline_source": "IJAM 2023",
        },
        {
            "question": "What is the strongest predictor of long-term survival in chronic coronary syndromes?",
            "options": [
                "Left ventricular ejection fraction (LVEF)",
                "Serum cholesterol level",
                "Resting heart rate",
                "Body mass index",
            ],
            "correct_answer": "A",
            "category": "risk_stratification",
            "difficulty": "medium",
            "guideline_source": "IJAM 2023 / CASS study",
        },
    ]
    return questions


def generate_full_benchmark(output_path: str | Path) -> int:
    all_questions = []
    all_questions.extend(generate_diamond_approach_mcqs())
    all_questions.extend(generate_risk_stratification_mcqs())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")

    print(f"Generated {len(all_questions)} benchmark questions to {output_path}")
    return len(all_questions)


if __name__ == "__main__":
    generate_full_benchmark("data/benchmarks/cardioqa_india_v1.jsonl")
