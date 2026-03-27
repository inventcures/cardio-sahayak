from dataclasses import dataclass, field

from src.knowledge.schemas import PatientProfile, DiamondResult, DrugClass, DRUG_CLASS_NAMES
from src.knowledge.drug_contraindications import is_nlem_available


@dataclass
class CrossCheckResult:
    contradictions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: bool = True


def cross_check_diamond_vs_report(
    diamond: DiamondResult,
    recommended_drugs: list[str],
) -> CrossCheckResult:
    result = CrossCheckResult()

    contraindicated_names = set()
    for dc in diamond.contraindicated:
        name = DRUG_CLASS_NAMES.get(dc, dc.value).lower()
        contraindicated_names.add(name)
        for word in name.split():
            if len(word) > 4:
                contraindicated_names.add(word)

    for drug in recommended_drugs:
        drug_lower = drug.lower()
        for ci_name in contraindicated_names:
            if ci_name in drug_lower or drug_lower in ci_name:
                result.contradictions.append(
                    f"Drug '{drug}' appears contraindicated by Diamond Approach "
                    f"for conditions: {[c.value for c in diamond.conditions]}"
                )
                result.passed = False

    return result


def cross_check_risk_vs_referral(
    risk_category: str,
    referral_urgency: str,
) -> CrossCheckResult:
    result = CrossCheckResult()

    if risk_category == "HIGH" and referral_urgency in ("MANAGE_AT_PHC", "ROUTINE"):
        result.contradictions.append(
            f"HIGH risk patient assigned {referral_urgency} referral. "
            f"Should be URGENT or EMERGENCY."
        )
        result.passed = False

    if risk_category == "LOW" and referral_urgency == "EMERGENCY":
        result.warnings.append(
            f"LOW risk patient assigned EMERGENCY referral. Verify clinical justification."
        )

    return result


def cross_check_nlem_availability(
    recommended_drug_classes: list[DrugClass],
) -> CrossCheckResult:
    result = CrossCheckResult()

    for dc in recommended_drug_classes:
        if not is_nlem_available(dc):
            result.warnings.append(
                f"{DRUG_CLASS_NAMES.get(dc, dc.value)} not in Indian NLEM 2022. "
                f"May have limited availability in rural settings."
            )

    return result


def run_all_cross_checks(
    patient: PatientProfile,
    diamond: DiamondResult,
    recommended_drugs: list[str],
    risk_category: str,
    referral_urgency: str,
    recommended_drug_classes: list[DrugClass] | None = None,
) -> CrossCheckResult:
    combined = CrossCheckResult()

    diamond_check = cross_check_diamond_vs_report(diamond, recommended_drugs)
    combined.contradictions.extend(diamond_check.contradictions)
    combined.warnings.extend(diamond_check.warnings)

    referral_check = cross_check_risk_vs_referral(risk_category, referral_urgency)
    combined.contradictions.extend(referral_check.contradictions)
    combined.warnings.extend(referral_check.warnings)

    if recommended_drug_classes:
        nlem_check = cross_check_nlem_availability(recommended_drug_classes)
        combined.warnings.extend(nlem_check.warnings)

    combined.passed = len(combined.contradictions) == 0
    return combined
