import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.knowledge.schemas import (
    PatientProfile,
    Gender,
    ComorbidityProfile,
    ChestPainInput,
    ChestPainPrecipitant,
    ChestPainLocation,
    ChestPainType,
    ChestPainDuration,
    ClinicalCondition,
    RiskCategory,
    BMICategory,
)
from src.knowledge.chest_pain_scoring import score_chest_pain
from src.knowledge.comorbidity_checklist import assess_comorbidity_risk
from src.knowledge.diamond_approach import select_antianginal_therapy
from src.knowledge.drug_contraindications import check_all_drug_safety, is_nlem_available
from src.knowledge.risk_stratification import assess_overall_risk, classify_by_lvef
from src.knowledge.treatment_targets import get_treatment_targets
from src.knowledge.south_asian_phenotype import (
    assess_south_asian_phenotype,
    classify_bmi_south_asian,
)
from src.knowledge.indian_guidelines import run_full_assessment, detect_clinical_conditions
from src.knowledge.schemas import DrugClass


def test_chest_pain_high_score():
    pain = ChestPainInput(
        precipitant=ChestPainPrecipitant.EXERTION_RELIEVED_BY_REST,
        location=ChestPainLocation.RETROSTERNAL_NECK_SHOULDER_JAW_ARM_EPIGASTRIC,
        pain_type=ChestPainType.CONSTRICTING_CRAMPING_HEAVY_TIGHT_BURNING_DULL,
        duration=ChestPainDuration.LESS_THAN_15_MIN,
    )
    result = score_chest_pain(pain)
    assert result.score == 6  # 3+1+1+1
    assert result.probability == "HIGH"


def test_chest_pain_low_score():
    pain = ChestPainInput(
        precipitant=ChestPainPrecipitant.BREATHING,
        location=ChestPainLocation.RIGHT_SIDE_SUBMAMMARY_LOCALIZED,
        pain_type=ChestPainType.REPRODUCIBLE_BY_PALPATION,
        duration=ChestPainDuration.MORE_THAN_15_MIN,
    )
    result = score_chest_pain(pain)
    assert result.score == -3  # -1+0+(-1)+(-1)
    assert result.probability == "LOW"


def test_chest_pain_intermediate():
    pain = ChestPainInput(
        precipitant=ChestPainPrecipitant.EMOTIONAL_COLD_MEAL,
        location=ChestPainLocation.RETROSTERNAL_NECK_SHOULDER_JAW_ARM_EPIGASTRIC,
        pain_type=ChestPainType.STABBING_SHARP,
        duration=ChestPainDuration.FEW_SECONDS,
    )
    result = score_chest_pain(pain)
    assert result.score == 2  # 1+1+0+0
    assert result.probability == "INTERMEDIATE"


def test_comorbidity_high_risk():
    profile = ComorbidityProfile(
        diabetes_mellitus=True,
        hypertension=True,
        current_smoker=True,
    )
    result = assess_comorbidity_risk(profile)
    assert result.score == 3
    assert result.risk_level == "HIGH"


def test_comorbidity_past_ihd_forces_high():
    profile = ComorbidityProfile(past_ihd=True)
    result = assess_comorbidity_risk(profile)
    assert result.risk_level == "HIGH"
    assert result.requires_cardiology_referral is True


def test_comorbidity_low_risk():
    profile = ComorbidityProfile()
    result = assess_comorbidity_risk(profile)
    assert result.score == 0
    assert result.risk_level == "LOW"


def test_diamond_htn_plus_lv_dysfunction():
    result = select_antianginal_therapy([
        ClinicalCondition.HYPERTENSION,
        ClinicalCondition.LV_DYSFUNCTION,
    ])
    assert DrugClass.BB in result.preferred
    assert DrugClass.DHP in result.contraindicated
    assert DrugClass.VER_DILT in result.contraindicated


def test_diamond_afib():
    result = select_antianginal_therapy([ClinicalCondition.ATRIAL_FIBRILLATION])
    assert DrugClass.BB in result.preferred
    assert DrugClass.VER_DILT in result.preferred
    assert DrugClass.IVAB in result.contraindicated


def test_diamond_no_conditions():
    result = select_antianginal_therapy([])
    assert len(result.preferred) > 0
    assert len(result.contraindicated) == 0


def test_lvef_risk_classification():
    assert classify_by_lvef(25) == RiskCategory.HIGH
    assert classify_by_lvef(42) == RiskCategory.INTERMEDIATE
    assert classify_by_lvef(60) == RiskCategory.LOW
    assert classify_by_lvef(None) == RiskCategory.INTERMEDIATE


def test_overall_risk_high():
    patient = PatientProfile(
        age=52,
        gender=Gender.MALE,
        lvef_percent=30,
        troponin_elevated=True,
        comorbidities=ComorbidityProfile(
            diabetes_mellitus=True,
            hypertension=True,
            current_smoker=True,
        ),
        has_diabetes=True,
    )
    result = assess_overall_risk(patient)
    assert result.risk_category == RiskCategory.HIGH
    assert result.annual_cv_mortality_estimate == ">3%"


def test_overall_risk_low():
    patient = PatientProfile(
        age=35,
        gender=Gender.FEMALE,
        lvef_percent=65,
    )
    result = assess_overall_risk(patient)
    assert result.risk_category == RiskCategory.LOW


def test_treatment_targets_diabetic():
    patient = PatientProfile(
        age=55,
        gender=Gender.MALE,
        has_diabetes=True,
        ldl_mg_dl=142,
        hba1c=8.2,
    )
    targets = get_treatment_targets(patient)
    assert targets.ldl_target_mg_dl == 70.0
    assert targets.hba1c_target == 7.0
    assert targets.bp_systolic_target == 130
    assert any("SGLT2i" in r for r in targets.specific_recommendations)
    assert any("LDL" in r for r in targets.specific_recommendations)


def test_treatment_targets_secondary_prevention():
    patient = PatientProfile(
        age=60,
        gender=Gender.MALE,
        comorbidities=ComorbidityProfile(past_ihd=True),
    )
    targets = get_treatment_targets(patient)
    assert targets.ldl_target_mg_dl == 70.0
    assert any("secondary prevention" in r for r in targets.specific_recommendations)


def test_south_asian_bmi():
    assert classify_bmi_south_asian(22) == BMICategory.NORMAL
    assert classify_bmi_south_asian(24) == BMICategory.OVERWEIGHT
    assert classify_bmi_south_asian(28) == BMICategory.OBESE


def test_south_asian_phenotype_flags():
    patient = PatientProfile(
        age=48,
        gender=Gender.MALE,
        bmi=24.5,
        waist_circumference_cm=95,
        has_diabetes=True,
        lvef_percent=45,
    )
    result = assess_south_asian_phenotype(patient)
    assert result.bmi_category == BMICategory.OVERWEIGHT
    assert any("BMI" in f for f in result.flags)
    assert any("Central adiposity" in f for f in result.flags)
    assert any("MYBPC3" in f for f in result.flags)
    assert any("Lp(a)" in f for f in result.flags)
    assert any("diabetes" in f.lower() for f in result.flags)


def test_nlem_availability():
    assert is_nlem_available(DrugClass.BB) is True
    assert is_nlem_available(DrugClass.IVAB) is False
    assert is_nlem_available(DrugClass.SGLT2I) is True


def test_drug_safety_sglt2i_low_egfr():
    patient = PatientProfile(
        age=70,
        gender=Gender.MALE,
        egfr=15,
        has_diabetes=True,
    )
    checks = check_all_drug_safety(patient)
    sglt2i_checks = [c for c in checks if c.drug_class == DrugClass.SGLT2I]
    assert len(sglt2i_checks) == 1
    assert sglt2i_checks[0].safe is False


def test_detect_clinical_conditions():
    patient = PatientProfile(
        age=55,
        gender=Gender.MALE,
        heart_rate_bpm=85,
        bp_systolic=160,
        bp_diastolic=95,
        lvef_percent=35,
        bnp_elevated=True,
    )
    conditions = detect_clinical_conditions(patient)
    assert ClinicalCondition.HIGH_HR in conditions
    assert ClinicalCondition.HYPERTENSION in conditions
    assert ClinicalCondition.LV_DYSFUNCTION in conditions
    assert ClinicalCondition.HEART_FAILURE in conditions


def test_full_assessment():
    patient = PatientProfile(
        age=52,
        gender=Gender.MALE,
        bmi=25.5,
        waist_circumference_cm=96,
        heart_rate_bpm=82,
        bp_systolic=148,
        bp_diastolic=92,
        lvef_percent=38,
        has_diabetes=True,
        ldl_mg_dl=155,
        hba1c=8.5,
        troponin_elevated=False,
        bnp_elevated=True,
        comorbidities=ComorbidityProfile(
            diabetes_mellitus=True,
            hypertension=True,
            cholesterol_gt_250=False,
            current_smoker=True,
        ),
    )
    chest_pain = ChestPainInput(
        precipitant=ChestPainPrecipitant.EXERTION_RELIEVED_BY_REST,
        location=ChestPainLocation.RETROSTERNAL_NECK_SHOULDER_JAW_ARM_EPIGASTRIC,
        pain_type=ChestPainType.CONSTRICTING_CRAMPING_HEAVY_TIGHT_BURNING_DULL,
        duration=ChestPainDuration.LESS_THAN_15_MIN,
    )

    result = run_full_assessment(patient, chest_pain)

    assert result["risk_assessment"].risk_category == RiskCategory.HIGH
    assert result["chest_pain_result"].probability == "HIGH"
    assert result["comorbidity_result"].risk_level == "HIGH"
    assert result["treatment_targets"].ldl_target_mg_dl == 70.0
    assert result["treatment_targets"].hba1c_target == 7.0
    assert len(result["drug_safety_warnings"]) > 0
    assert result["south_asian_phenotype"].bmi_category == BMICategory.OVERWEIGHT
    assert ClinicalCondition.HYPERTENSION in result["clinical_conditions"]
    assert ClinicalCondition.LV_DYSFUNCTION in result["clinical_conditions"]


if __name__ == "__main__":
    test_funcs = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for func in test_funcs:
        try:
            func()
            passed += 1
            print(f"  PASS: {func.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {func.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
