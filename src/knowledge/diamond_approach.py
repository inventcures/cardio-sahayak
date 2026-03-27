from src.knowledge.schemas import (
    ClinicalCondition, DrugClass, DiamondResult, DRUG_CLASS_NAMES,
)

DIAMOND_TABLE: dict[ClinicalCondition, tuple[set[DrugClass], set[DrugClass], set[DrugClass]]] = {
    ClinicalCondition.HIGH_HR: (
        {DrugClass.BB, DrugClass.VER_DILT, DrugClass.IVAB},
        {DrugClass.TRIM, DrugClass.RAN},
        {DrugClass.DHP, DrugClass.NITR, DrugClass.NIC},
    ),
    ClinicalCondition.BRADYCARDIA: (
        {DrugClass.DHP, DrugClass.NIC, DrugClass.NITR, DrugClass.TRIM, DrugClass.RAN},
        set(),
        {DrugClass.BB, DrugClass.VER_DILT, DrugClass.IVAB},
    ),
    ClinicalCondition.HYPERTENSION: (
        {DrugClass.BB, DrugClass.DHP, DrugClass.VER_DILT, DrugClass.NITR, DrugClass.NIC},
        {DrugClass.TRIM, DrugClass.RAN, DrugClass.IVAB},
        set(),
    ),
    ClinicalCondition.HYPOTENSION: (
        {DrugClass.TRIM, DrugClass.RAN, DrugClass.IVAB},
        set(),
        {DrugClass.BB, DrugClass.VER_DILT, DrugClass.DHP, DrugClass.NITR, DrugClass.NIC},
    ),
    ClinicalCondition.LV_DYSFUNCTION: (
        {DrugClass.BB},
        {DrugClass.TRIM, DrugClass.IVAB, DrugClass.RAN, DrugClass.NITR},
        {DrugClass.DHP, DrugClass.VER_DILT, DrugClass.NIC},
    ),
    ClinicalCondition.HEART_FAILURE: (
        {DrugClass.BB, DrugClass.IVAB},
        {DrugClass.TRIM, DrugClass.NITR, DrugClass.RAN},
        {DrugClass.DHP, DrugClass.VER_DILT, DrugClass.NIC},
    ),
    ClinicalCondition.ATRIAL_FIBRILLATION: (
        {DrugClass.BB, DrugClass.VER_DILT},
        {DrugClass.TRIM, DrugClass.RAN},
        {DrugClass.DHP, DrugClass.NITR, DrugClass.NIC, DrugClass.IVAB},
    ),
}


def select_antianginal_therapy(conditions: list[ClinicalCondition]) -> DiamondResult:
    if not conditions:
        return DiamondResult(
            conditions=[],
            preferred=list(DrugClass),
            acceptable=[],
            contraindicated=[],
            rationale=["No specific conditions identified. All antianginal classes may be considered."],
        )

    all_preferred: set[DrugClass] | None = None
    all_acceptable: set[DrugClass] = set()
    all_contraindicated: set[DrugClass] = set()
    rationale: list[str] = []

    for condition in conditions:
        if condition not in DIAMOND_TABLE:
            continue

        preferred, acceptable, contraindicated = DIAMOND_TABLE[condition]

        if all_preferred is None:
            all_preferred = set(preferred)
        else:
            all_preferred &= preferred

        all_acceptable |= acceptable
        all_contraindicated |= contraindicated

        cond_name = condition.value.replace("_", " ").title()
        preferred_names = ", ".join(
            DRUG_CLASS_NAMES.get(d, d.value)
            for d in sorted(preferred, key=lambda x: x.value)
        )
        rationale.append(f"{cond_name}: preferred [{preferred_names}]")

    if all_preferred is None:
        all_preferred = set()

    all_preferred -= all_contraindicated
    all_acceptable -= all_contraindicated
    all_acceptable -= all_preferred

    return DiamondResult(
        conditions=list(conditions),
        preferred=sorted(all_preferred, key=lambda x: x.value),
        acceptable=sorted(all_acceptable, key=lambda x: x.value),
        contraindicated=sorted(all_contraindicated, key=lambda x: x.value),
        rationale=rationale,
    )
