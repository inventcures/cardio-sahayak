from src.knowledge.schemas import (
    PatientProfile,
    ChestPainInput,
    ClinicalCondition,
    RiskAssessment,
    DiamondResult,
    TreatmentTargets,
)
from src.knowledge.chest_pain_scoring import score_chest_pain
from src.knowledge.comorbidity_checklist import assess_comorbidity_risk
from src.knowledge.diamond_approach import select_antianginal_therapy
from src.knowledge.drug_contraindications import check_all_drug_safety, is_nlem_available
from src.knowledge.risk_stratification import assess_overall_risk
from src.knowledge.south_asian_phenotype import assess_south_asian_phenotype
from src.knowledge.treatment_targets import get_treatment_targets


def detect_clinical_conditions(patient: PatientProfile) -> list[ClinicalCondition]:
    conditions: list[ClinicalCondition] = []

    if patient.heart_rate_bpm is not None:
        if patient.heart_rate_bpm >= 70:
            conditions.append(ClinicalCondition.HIGH_HR)
        elif patient.heart_rate_bpm < 60:
            conditions.append(ClinicalCondition.BRADYCARDIA)

    if patient.bp_systolic is not None:
        if patient.bp_systolic >= 140 or (patient.bp_diastolic and patient.bp_diastolic >= 90):
            conditions.append(ClinicalCondition.HYPERTENSION)
        elif patient.bp_systolic < 90:
            conditions.append(ClinicalCondition.HYPOTENSION)
    elif patient.comorbidities.hypertension:
        conditions.append(ClinicalCondition.HYPERTENSION)

    if patient.lvef_percent is not None and patient.lvef_percent < 40:
        conditions.append(ClinicalCondition.LV_DYSFUNCTION)

    if patient.bnp_elevated:
        conditions.append(ClinicalCondition.HEART_FAILURE)

    return conditions


def run_full_assessment(
    patient: PatientProfile,
    chest_pain: ChestPainInput | None = None,
) -> dict:
    risk = assess_overall_risk(patient, chest_pain)
    conditions = detect_clinical_conditions(patient)
    diamond = select_antianginal_therapy(conditions)
    targets = get_treatment_targets(patient)
    phenotype = assess_south_asian_phenotype(patient)
    drug_safety = check_all_drug_safety(patient)

    chest_pain_result = None
    if chest_pain is not None:
        chest_pain_result = score_chest_pain(chest_pain)

    comorbidity_result = assess_comorbidity_risk(patient.comorbidities)

    return {
        "risk_assessment": risk,
        "clinical_conditions": conditions,
        "diamond_approach": diamond,
        "treatment_targets": targets,
        "south_asian_phenotype": phenotype,
        "drug_safety_warnings": drug_safety,
        "chest_pain_result": chest_pain_result,
        "comorbidity_result": comorbidity_result,
    }
