from dataclasses import dataclass, field

from src.knowledge.schemas import PatientProfile, BMICategory

SOUTH_ASIAN_BMI_NORMAL_UPPER = 23.0
SOUTH_ASIAN_BMI_OVERWEIGHT_UPPER = 27.5

SOUTH_ASIAN_WAIST_MALE = 90
SOUTH_ASIAN_WAIST_FEMALE = 80

MYBPC3_PREVALENCE = 0.04


@dataclass
class PhenotypeAssessment:
    bmi_category: BMICategory | None
    flags: list[str] = field(default_factory=list)
    age_adjusted_risk: str = ""


def classify_bmi_south_asian(bmi: float) -> BMICategory:
    if bmi < SOUTH_ASIAN_BMI_NORMAL_UPPER:
        return BMICategory.NORMAL
    if bmi < SOUTH_ASIAN_BMI_OVERWEIGHT_UPPER:
        return BMICategory.OVERWEIGHT
    return BMICategory.OBESE


def assess_south_asian_phenotype(patient: PatientProfile) -> PhenotypeAssessment:
    flags: list[str] = []
    bmi_category = None

    if patient.bmi is not None:
        bmi_category = classify_bmi_south_asian(patient.bmi)
        if bmi_category != BMICategory.NORMAL:
            flags.append(
                f"BMI {patient.bmi} kg/m2 exceeds South Asian threshold "
                f"(>={SOUTH_ASIAN_BMI_NORMAL_UPPER} vs >=25 Western). "
                f"Category: {bmi_category.value}"
            )

    if patient.waist_circumference_cm is not None:
        threshold = (
            SOUTH_ASIAN_WAIST_MALE
            if patient.gender.value == "male"
            else SOUTH_ASIAN_WAIST_FEMALE
        )
        if patient.waist_circumference_cm > threshold:
            flags.append(
                f"Central adiposity: waist {patient.waist_circumference_cm} cm "
                f"exceeds South Asian threshold ({threshold} cm for {patient.gender.value}). "
                f"Central adiposity more predictive than BMI for CV risk in South Asians."
            )

    # Premature CAD risk window (5-10 years earlier than Western)
    if patient.gender.value == "male" and patient.age < 55:
        age_risk = (
            f"South Asian male aged {patient.age}: within premature CAD risk window "
            f"(onset 5-10 years earlier than Western populations)"
        )
        if patient.age < 50:
            flags.append(age_risk)
    elif patient.gender.value == "female" and patient.age < 65:
        age_risk = f"South Asian female aged {patient.age}: within premature CAD risk window"
    else:
        age_risk = f"Age {patient.age}: standard age-related CV risk"

    if patient.lvef_percent is not None and patient.lvef_percent < 50:
        flags.append(
            "Consider MYBPC3 Delta-25bp screening: 4% prevalence in South Asians, "
            "associated with HCM and heart failure. Important for family cascade screening."
        )

    flags.append(
        "Elevated Lp(a) is an independent CV risk factor in South Asians. "
        "Consider measurement if family history of premature CAD."
    )

    if patient.has_diabetes:
        flags.append(
            "India: 50-60% of CAD patients also have diabetes. "
            "Prioritize cardio-renal-metabolic integration."
        )

    return PhenotypeAssessment(
        bmi_category=bmi_category,
        flags=flags,
        age_adjusted_risk=age_risk,
    )
