from dataclasses import dataclass

from src.knowledge.schemas import PatientProfile, DrugClass


@dataclass
class DrugCheck:
    drug_class: DrugClass
    safe: bool
    warning: str = ""


NLEM_AVAILABLE = {
    DrugClass.BB: True,
    DrugClass.DHP: True,
    DrugClass.VER_DILT: True,
    DrugClass.IVAB: False,
    DrugClass.NIC: False,
    DrugClass.NITR: True,
    DrugClass.RAN: False,
    DrugClass.TRIM: False,
    DrugClass.ACEI: True,
    DrugClass.ARB: True,
    DrugClass.STATIN: True,
    DrugClass.SGLT2I: True,
    DrugClass.GLP1RA: False,
    DrugClass.ANTIPLATELET: True,
}


def check_sglt2i_safety(patient: PatientProfile) -> DrugCheck:
    if patient.egfr is not None and patient.egfr < 20:
        return DrugCheck(
            DrugClass.SGLT2I, safe=False,
            warning="SGLT2i contraindicated: eGFR <20 ml/min/1.73m2",
        )
    if patient.egfr is not None and patient.egfr < 45:
        return DrugCheck(
            DrugClass.SGLT2I, safe=True,
            warning="SGLT2i: reduced efficacy for glycemic control at eGFR 20-45, but cardio-renal benefits persist",
        )
    return DrugCheck(DrugClass.SGLT2I, safe=True)


def check_ranolazine_safety(patient: PatientProfile) -> DrugCheck:
    return DrugCheck(
        DrugClass.RAN, safe=True,
        warning="Ranolazine: monitor QTc. Caution with other QT-prolonging drugs. Benefits more prominent with higher HbA1c.",
    )


def check_acei_safety(patient: PatientProfile) -> DrugCheck:
    if patient.gender.value == "female" and patient.age < 50:
        return DrugCheck(
            DrugClass.ACEI, safe=True,
            warning="ACEi: confirm not pregnant or planning pregnancy (teratogenic)",
        )
    return DrugCheck(DrugClass.ACEI, safe=True)


def check_statin_recommendation(patient: PatientProfile) -> DrugCheck:
    if patient.ldl_mg_dl is not None and patient.ldl_mg_dl > 100:
        if patient.has_diabetes or patient.comorbidities.past_ihd:
            return DrugCheck(
                DrugClass.STATIN, safe=True,
                warning=f"High-intensity statin indicated: LDL {patient.ldl_mg_dl} mg/dl, target <70 mg/dl. Recommend atorvastatin 40-80mg.",
            )
        return DrugCheck(
            DrugClass.STATIN, safe=True,
            warning=f"Statin indicated even with mild LDL elevation ({patient.ldl_mg_dl} mg/dl >100). Per Indian Consensus: do not defer.",
        )
    return DrugCheck(DrugClass.STATIN, safe=True)


def check_all_drug_safety(patient: PatientProfile) -> list[DrugCheck]:
    checks = [
        check_sglt2i_safety(patient),
        check_ranolazine_safety(patient),
        check_acei_safety(patient),
        check_statin_recommendation(patient),
    ]
    return [c for c in checks if c.warning]


def is_nlem_available(drug_class: DrugClass) -> bool:
    return NLEM_AVAILABLE.get(drug_class, False)
