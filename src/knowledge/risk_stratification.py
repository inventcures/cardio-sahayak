from src.knowledge.schemas import (
    PatientProfile,
    RiskCategory,
    RiskAssessment,
    ChestPainInput,
)
from src.knowledge.chest_pain_scoring import score_chest_pain
from src.knowledge.comorbidity_checklist import assess_comorbidity_risk


def classify_by_lvef(lvef_percent: float | None) -> RiskCategory:
    if lvef_percent is None:
        return RiskCategory.INTERMEDIATE
    if lvef_percent < 35:
        return RiskCategory.HIGH
    if lvef_percent < 50:
        return RiskCategory.INTERMEDIATE
    return RiskCategory.LOW


RISK_PRIORITY = {
    RiskCategory.LOW: 0,
    RiskCategory.INTERMEDIATE: 1,
    RiskCategory.HIGH: 2,
}

RECOMMENDATIONS = {
    RiskCategory.HIGH: (
        "Early revascularization + optimal medical therapy. "
        "Urgent cardiology referral."
    ),
    RiskCategory.INTERMEDIATE: (
        "Aggressive OMT. Consider CCTA or stress imaging. "
        "Cardiology consultation within 2 weeks."
    ),
    RiskCategory.LOW: (
        "Lifestyle modification. Primary prevention pharmacotherapy. "
        "Routine follow-up."
    ),
}

MORTALITY_ESTIMATES = {
    RiskCategory.HIGH: ">3%",
    RiskCategory.INTERMEDIATE: "1-3%",
    RiskCategory.LOW: "<1%",
}


def assess_overall_risk(
    patient: PatientProfile,
    chest_pain: ChestPainInput | None = None,
) -> RiskAssessment:
    factors: list[str] = []
    risk_levels: list[RiskCategory] = []

    # LVEF-based risk
    lvef_risk = classify_by_lvef(patient.lvef_percent)
    risk_levels.append(lvef_risk)
    if patient.lvef_percent is not None:
        factors.append(f"LVEF {patient.lvef_percent}% -> {lvef_risk.value} risk")

    # Comorbidity risk
    comorbidity_result = assess_comorbidity_risk(patient.comorbidities)
    factors.extend(comorbidity_result.factors_present)
    if comorbidity_result.risk_level == "HIGH":
        risk_levels.append(RiskCategory.HIGH)
    elif comorbidity_result.risk_level == "MODERATE":
        risk_levels.append(RiskCategory.INTERMEDIATE)

    # Chest pain score
    chest_pain_score = None
    if chest_pain is not None:
        cp_result = score_chest_pain(chest_pain)
        chest_pain_score = cp_result.score
        factors.append(f"Chest pain score: {cp_result.score}/6 ({cp_result.probability})")
        if cp_result.score >= 3:
            risk_levels.append(RiskCategory.HIGH)

    # Troponin elevation
    if patient.troponin_elevated:
        factors.append("Elevated troponin")
        risk_levels.append(RiskCategory.HIGH)

    # Diabetes (50-60% CAD-DM overlap in India)
    if patient.has_diabetes:
        factors.append("Diabetes mellitus (2x CAD risk; 50-60% CAD-DM overlap in India)")

    overall = max(risk_levels, key=lambda r: RISK_PRIORITY[r])

    return RiskAssessment(
        risk_category=overall,
        chest_pain_score=chest_pain_score,
        comorbidity_score=comorbidity_result.score,
        annual_cv_mortality_estimate=MORTALITY_ESTIMATES[overall],
        factors=factors,
        recommendation=RECOMMENDATIONS[overall],
    )
