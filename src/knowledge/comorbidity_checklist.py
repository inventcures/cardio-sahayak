from src.knowledge.schemas import ComorbidityProfile, ComorbidityResult

FACTOR_LABELS = {
    "diabetes_mellitus": "Diabetes mellitus",
    "cholesterol_gt_250": "Cholesterol >250 mg/dl (>6.47 mmol/l)",
    "current_smoker": "Current smoker or recent ex-smoker",
    "family_history_cad_lt_60": "Family history of CAD in first-degree relative <60 years",
    "hypertension": "Hypertension",
    "past_ihd": "Past history of IHD/PCI/CABG",
}


def assess_comorbidity_risk(profile: ComorbidityProfile) -> ComorbidityResult:
    factors_present = [
        label
        for field_name, label in FACTOR_LABELS.items()
        if getattr(profile, field_name)
    ]

    score = len(factors_present)
    requires_cardiology_referral = profile.past_ihd

    if score >= 3 or requires_cardiology_referral:
        risk_level = "HIGH"
    elif score >= 1:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return ComorbidityResult(
        score=score,
        risk_level=risk_level,
        requires_cardiology_referral=requires_cardiology_referral,
        factors_present=factors_present,
    )
