import gradio as gr
from src.knowledge.schemas import (
    PatientProfile, Gender, ComorbidityProfile, ChestPainInput,
    ChestPainPrecipitant, ChestPainLocation, ChestPainType, ChestPainDuration,
    ClinicalCondition, DRUG_CLASS_NAMES,
)
from src.knowledge.indian_guidelines import run_full_assessment, detect_clinical_conditions
from src.knowledge.south_asian_phenotype import assess_south_asian_phenotype


PRECIPITANT_MAP = {
    "Exertion, relieved by rest": ChestPainPrecipitant.EXERTION_RELIEVED_BY_REST,
    "Emotional stress / cold / after meal": ChestPainPrecipitant.EMOTIONAL_COLD_MEAL,
    "Unpredictable": ChestPainPrecipitant.UNPREDICTABLE,
    "Breathing in/out": ChestPainPrecipitant.BREATHING,
}

LOCATION_MAP = {
    "Retrosternal / neck / shoulder / jaw / arm / epigastric": ChestPainLocation.RETROSTERNAL_NECK_SHOULDER_JAW_ARM_EPIGASTRIC,
    "Right side / sub-mammary / localized": ChestPainLocation.RIGHT_SIDE_SUBMAMMARY_LOCALIZED,
}

TYPE_MAP = {
    "Constricting / cramping / heavy / tight / burning / dull": ChestPainType.CONSTRICTING_CRAMPING_HEAVY_TIGHT_BURNING_DULL,
    "Stabbing / sharp": ChestPainType.STABBING_SHARP,
    "Reproducible by manual pressure": ChestPainType.REPRODUCIBLE_BY_PALPATION,
}

DURATION_MAP = {
    "Less than 15 minutes": ChestPainDuration.LESS_THAN_15_MIN,
    "Few seconds only": ChestPainDuration.FEW_SECONDS,
    "More than 15 minutes": ChestPainDuration.MORE_THAN_15_MIN,
}


def _build_patient_profile(
    age, gender, bmi, waist_cm, heart_rate, bp_sys, bp_dia,
    dm, htn, smoking, family_hx, past_ihd, high_cholesterol, has_ckd,
    ldl, hba1c, troponin_elevated, bnp_elevated, egfr,
):
    return PatientProfile(
        age=int(age),
        gender=Gender(gender.lower()),
        bmi=float(bmi) if bmi else None,
        waist_circumference_cm=float(waist_cm) if waist_cm else None,
        heart_rate_bpm=int(heart_rate) if heart_rate else None,
        bp_systolic=int(bp_sys) if bp_sys else None,
        bp_diastolic=int(bp_dia) if bp_dia else None,
        has_diabetes=dm,
        has_ckd=has_ckd,
        egfr=float(egfr) if egfr else None,
        hba1c=float(hba1c) if hba1c else None,
        ldl_mg_dl=float(ldl) if ldl else None,
        troponin_elevated=troponin_elevated,
        bnp_elevated=bnp_elevated,
        lvef_percent=None,
        comorbidities=ComorbidityProfile(
            diabetes_mellitus=dm,
            hypertension=htn,
            current_smoker=smoking,
            family_history_cad_lt_60=family_hx,
            past_ihd=past_ihd,
            cholesterol_gt_250=high_cholesterol,
        ),
    )


def _build_chest_pain_input(precipitant, location, pain_type, duration):
    if precipitant == "Not applicable":
        return None
    if (precipitant in PRECIPITANT_MAP and location in LOCATION_MAP
            and pain_type in TYPE_MAP and duration in DURATION_MAP):
        return ChestPainInput(
            precipitant=PRECIPITANT_MAP[precipitant],
            location=LOCATION_MAP[location],
            pain_type=TYPE_MAP[pain_type],
            duration=DURATION_MAP[duration],
        )
    return None


def _format_risk_section(risk):
    return f"""## Risk Assessment
**Category: {risk.risk_category.value}** (Annual CV mortality: {risk.annual_cv_mortality_estimate})

**Recommendation:** {risk.recommendation}

### Risk Factors Identified:
{chr(10).join(f'- {f}' for f in risk.factors)}
"""


def _format_chest_pain_section(chest_pain_result):
    if not chest_pain_result:
        return ""
    return f"""## Chest Pain Score
**Score: {chest_pain_result.score}** | Probability: **{chest_pain_result.probability}**
{chest_pain_result.recommendation}
"""


def _format_comorbidity_section(comorbidity):
    factors = ', '.join(comorbidity.factors_present) if comorbidity.factors_present else 'None identified'
    referral = 'Yes' if comorbidity.requires_cardiology_referral else 'No'
    return f"""## Comorbidity Assessment
**Score: {comorbidity.score}/6** | Risk Level: **{comorbidity.risk_level}**
Cardiology Referral Required: **{referral}**

Factors: {factors}
"""


def _format_diamond_section(diamond):
    preferred = "\n".join(
        f"- {DRUG_CLASS_NAMES.get(d, d.value)}" for d in diamond.preferred
    ) or "None specific"
    contraindicated = "\n".join(
        f"- {DRUG_CLASS_NAMES.get(d, d.value)}" for d in diamond.contraindicated
    ) or "None"
    conditions_display = ', '.join(
        c.value.replace('_', ' ').title() for c in diamond.conditions
    ) or 'None detected'

    return f"""## Diamond Approach Drug Selection
**Clinical Conditions:** {conditions_display}

### Preferred:
{preferred}

### Contraindicated:
{contraindicated}

### Rationale:
{chr(10).join(f'- {r}' for r in diamond.rationale)}
"""


def _format_targets_section(targets):
    hba1c_display = f'<{targets.hba1c_target}%' if targets.hba1c_target else 'N/A'
    return f"""## Treatment Targets
- **LDL:** <{targets.ldl_target_mg_dl} mg/dl
- **BP:** <={targets.bp_systolic_target}/{targets.bp_diastolic_target} mmHg
- **HbA1c:** {hba1c_display}

### Specific Recommendations:
{chr(10).join(f'- {r}' for r in targets.specific_recommendations)}
"""


def _format_phenotype_section(phenotype):
    bmi_display = phenotype.bmi_category.value if phenotype.bmi_category else 'N/A'
    return f"""## South Asian Phenotype Assessment
**BMI Category:** {bmi_display} (South Asian thresholds)
**Age Risk:** {phenotype.age_adjusted_risk}

### Flags:
{chr(10).join(f'- {f}' for f in phenotype.flags)}
"""


def _format_drug_safety_section(drug_safety):
    if not drug_safety:
        return ""
    return f"""## Drug Safety Warnings
{chr(10).join(f'- **{w.drug_class.value.upper()}**: {w.warning}' for w in drug_safety)}
"""


def run_assessment(
    age, gender, bmi, waist_cm, heart_rate, bp_sys, bp_dia,
    dm, htn, smoking, family_hx, past_ihd, high_cholesterol,
    chest_pain_precipitant, chest_pain_location, chest_pain_type, chest_pain_duration,
    ldl, hba1c, troponin_elevated, bnp_elevated, egfr,
    has_ckd,
):
    patient = _build_patient_profile(
        age, gender, bmi, waist_cm, heart_rate, bp_sys, bp_dia,
        dm, htn, smoking, family_hx, past_ihd, high_cholesterol, has_ckd,
        ldl, hba1c, troponin_elevated, bnp_elevated, egfr,
    )
    chest_pain = _build_chest_pain_input(
        chest_pain_precipitant, chest_pain_location,
        chest_pain_type, chest_pain_duration,
    )

    result = run_full_assessment(patient, chest_pain)
    phenotype = assess_south_asian_phenotype(patient)

    sections = [
        _format_risk_section(result["risk_assessment"]),
        _format_chest_pain_section(result["chest_pain_result"]),
        _format_comorbidity_section(result["comorbidity_result"]),
        _format_diamond_section(result["diamond_approach"]),
        _format_targets_section(result["treatment_targets"]),
        _format_phenotype_section(phenotype),
        _format_drug_safety_section(result["drug_safety_warnings"]),
    ]
    full_report = "\n".join(s for s in sections if s)

    return result["risk_assessment"].risk_category.value, full_report


def create_dashboard():
    with gr.Blocks(
        title="Cardio-Sahayak: Clinical Decision Support",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# Cardio-Sahayak: Clinical Decision Support for Indian Cardiovascular Medicine")
        gr.Markdown("*Powered by Indian Consensus Guidelines (IJAM 2023) + MARCUS-inspired Architecture*")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Patient Demographics")
                age = gr.Number(label="Age", value=52)
                gender = gr.Dropdown(
                    ["Male", "Female", "Other"], label="Gender", value="Male",
                )
                bmi = gr.Number(label="BMI (kg/m2)", value=25.5)
                waist_cm = gr.Number(label="Waist Circumference (cm)", value=None)
                heart_rate = gr.Number(label="Heart Rate (bpm)", value=82)
                with gr.Row():
                    bp_sys = gr.Number(label="Systolic BP", value=148)
                    bp_dia = gr.Number(label="Diastolic BP", value=92)

                gr.Markdown("### Comorbidities")
                dm = gr.Checkbox(label="Diabetes Mellitus", value=True)
                htn = gr.Checkbox(label="Hypertension", value=True)
                smoking = gr.Checkbox(label="Current Smoker", value=False)
                family_hx = gr.Checkbox(label="Family Hx CAD <60 years", value=False)
                past_ihd = gr.Checkbox(label="Past IHD / PCI / CABG", value=False)
                high_cholesterol = gr.Checkbox(label="Cholesterol >250 mg/dl", value=False)
                has_ckd = gr.Checkbox(label="Chronic Kidney Disease", value=False)

            with gr.Column(scale=1):
                gr.Markdown("### Chest Pain Assessment (IJAM Table 1)")
                chest_pain_precipitant = gr.Dropdown(
                    ["Not applicable", "Exertion, relieved by rest",
                     "Emotional stress / cold / after meal",
                     "Unpredictable", "Breathing in/out"],
                    label="Precipitating Factor", value="Not applicable",
                )
                chest_pain_location = gr.Dropdown(
                    ["Retrosternal / neck / shoulder / jaw / arm / epigastric",
                     "Right side / sub-mammary / localized"],
                    label="Location",
                    value="Retrosternal / neck / shoulder / jaw / arm / epigastric",
                )
                chest_pain_type = gr.Dropdown(
                    ["Constricting / cramping / heavy / tight / burning / dull",
                     "Stabbing / sharp", "Reproducible by manual pressure"],
                    label="Type",
                    value="Constricting / cramping / heavy / tight / burning / dull",
                )
                chest_pain_duration = gr.Dropdown(
                    ["Less than 15 minutes", "Few seconds only", "More than 15 minutes"],
                    label="Duration", value="Less than 15 minutes",
                )

                gr.Markdown("### Laboratory Values")
                ldl = gr.Number(label="LDL (mg/dl)", value=155)
                hba1c = gr.Number(label="HbA1c (%)", value=8.5)
                egfr = gr.Number(label="eGFR (ml/min/1.73m2)", value=None)
                troponin_elevated = gr.Checkbox(label="Troponin Elevated", value=False)
                bnp_elevated = gr.Checkbox(label="BNP/NT-proBNP Elevated", value=True)

                gr.Markdown("### ECG Upload")
                ecg_image = gr.Image(label="Upload 12-lead ECG", type="filepath")

                assess_btn = gr.Button(
                    "Run Clinical Assessment", variant="primary", size="lg",
                )

        gr.Markdown("---")

        with gr.Row():
            risk_badge = gr.Textbox(label="Risk Category", interactive=False)

        report_output = gr.Markdown(label="Clinical Assessment Report")

        assess_btn.click(
            fn=run_assessment,
            inputs=[
                age, gender, bmi, waist_cm, heart_rate, bp_sys, bp_dia,
                dm, htn, smoking, family_hx, past_ihd, high_cholesterol,
                chest_pain_precipitant, chest_pain_location,
                chest_pain_type, chest_pain_duration,
                ldl, hba1c, troponin_elevated, bnp_elevated, egfr,
                has_ckd,
            ],
            outputs=[risk_badge, report_output],
        )

        gr.Markdown("---")
        gr.Markdown(
            "*Disclaimer: This is a clinical decision support tool, not a substitute for clinical judgment. "
            "All recommendations require physician confirmation. "
            "Guidelines: IJAM 2023 Consensus, ESC 2019, Indian National Consensus.*"
        )

    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(server_name="0.0.0.0", server_port=7860)
