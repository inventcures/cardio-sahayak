from src.knowledge.schemas import PatientProfile, TreatmentTargets


def get_treatment_targets(patient: PatientProfile) -> TreatmentTargets:
    recommendations: list[str] = []
    ldl_target = 100.0
    bp_sys = 140
    bp_dia = 90
    hba1c_target = None

    if patient.has_diabetes:
        ldl_target = 70.0
        bp_sys = 130
        bp_dia = 80
        hba1c_target = 7.0
        recommendations.append(
            "SGLT2i recommended: empagliflozin 10-25mg or dapagliflozin 10mg "
            "(cardiac + renal benefits)"
        )
        recommendations.append(
            "Consider GLP-1RA: liraglutide 1.2-1.8mg or semaglutide 0.5-1mg "
            "if HbA1c above target"
        )
        recommendations.append("ACE inhibitor for event prevention in diabetic CCS patients")

        if patient.ldl_mg_dl is not None and patient.ldl_mg_dl > 70:
            recommendations.append(
                f"LDL {patient.ldl_mg_dl} mg/dl exceeds target of <70 mg/dl. "
                f"High-intensity statin: atorvastatin 40-80mg or rosuvastatin 20-40mg"
            )
        if patient.hba1c is not None and patient.hba1c > 7.0:
            recommendations.append(
                f"HbA1c {patient.hba1c}% above target 7%. Intensify glycemic management."
            )

    elif patient.comorbidities.past_ihd:
        ldl_target = 70.0
        bp_sys = 130
        bp_dia = 80
        recommendations.append("High-intensity statin mandatory for secondary prevention")
        recommendations.append("Dual antiplatelet therapy if within 12 months of ACS/PCI")

    elif patient.comorbidities.hypertension:
        ldl_target = 100.0
        bp_sys = 130
        bp_dia = 80
        recommendations.append("First-line: ACEi/ARB + CCB or thiazide combination")

    if (
        patient.ldl_mg_dl is not None
        and patient.ldl_mg_dl > 100
        and ldl_target > 70
    ):
        recommendations.append(
            f"Statin indicated: LDL {patient.ldl_mg_dl} mg/dl >100 mg/dl. "
            f"Per Indian Consensus: do not defer even with mild elevation."
        )

    if patient.has_ckd:
        recommendations.append(
            "CKD: adjust renally excreted drug doses. Minimize iodinated contrast."
        )
        if patient.egfr is not None and patient.egfr < 30:
            recommendations.append(
                "Severe CKD: avoid metformin. SGLT2i cardio-renal benefits may persist."
            )

    if patient.gender.value == "female":
        recommendations.append(
            "Women with CCS: treat risk factors more aggressively. "
            "Stress echo preferred over stress ECG."
        )

    return TreatmentTargets(
        ldl_target_mg_dl=ldl_target,
        bp_systolic_target=bp_sys,
        bp_diastolic_target=bp_dia,
        hba1c_target=hba1c_target,
        specific_recommendations=recommendations,
    )
