import gradio as gr
from src.knowledge.schemas import (
    PatientProfile, Gender, ComorbidityProfile,
)
from src.knowledge.indian_guidelines import run_full_assessment
from src.knowledge.south_asian_phenotype import assess_south_asian_phenotype


RISK_EXPLANATIONS = {
    "HIGH": {
        "en": "Your heart risk level is HIGH. This means you need urgent medical attention. Please see a cardiologist as soon as possible.",
        "hi": "Aapka hriday jokhim star UCHCH hai. Iska matlab hai ki aapko turant chikitsa dhyan ki zaroorat hai. Kripya jald se jald hriday rog visheshagya se milein.",
    },
    "INTERMEDIATE": {
        "en": "Your heart risk level is MODERATE. This means you should take steps to manage your health and see your doctor regularly.",
        "hi": "Aapka hriday jokhim star MADHYAM hai. Iska matlab hai ki aapko apne swasthya ka dhyan rakhna chahiye aur niyamit roop se doctor se milna chahiye.",
    },
    "LOW": {
        "en": "Your heart risk level is LOW. Continue maintaining a healthy lifestyle. Regular check-ups are still important.",
        "hi": "Aapka hriday jokhim star KAM hai. Swasth jeevan shaili banaye rakhein. Niyamit jaanch ab bhi mahatvapurn hai.",
    },
}

LIFESTYLE_GUIDANCE = {
    "en": """### What You Can Do:
- **Diet:** Reduce salt (namak) intake, limit fried foods and ghee, eat more vegetables and fruits
- **Exercise:** Walk for 30 minutes daily at a comfortable pace
- **Tobacco:** If you use tobacco (cigarettes, bidi, gutka), please stop. Ask your doctor for help
- **Medications:** Take all medicines as prescribed by your doctor, at the same time every day
- **Monitoring:** Check your blood pressure regularly if you have a BP machine
""",
    "hi": """### Aap Kya Kar Sakte Hain:
- **Aahar:** Namak kam karein, tala hua khana aur ghee seimit karein, adhik sabzi aur phal khayen
- **Vyayam:** Har din 30 minute aram se chalen
- **Tambaaku:** Agar aap tambaaku ka sevan karte hain (cigarette, bidi, gutka), kripya band karein
- **Dawai:** Doctor dwara batayi gayi sabhi dawai samay par lein
- **Nigrani:** Agar aapke paas BP machine hai to niyamit roop se BP jaanchein
""",
}

EMERGENCY_SIGNS = {
    "en": """### Go to Hospital Immediately If:
- Severe chest pain lasting more than 15 minutes
- Breathlessness at rest
- Sudden fainting or loss of consciousness
- Severe dizziness with sweating
- Swelling in legs getting worse suddenly
""",
    "hi": """### Turant Aspatal Jayen Agar:
- 15 minute se adhik samay tak seene mein tez dard
- Aaram mein saans ki takleef
- Achanak behosh hona
- Tez chakkar aana aur paseena aana
- Pairo mein sujan achanak badhna
""",
}


def patient_assessment(age, gender, bmi, dm, htn, smoking, ldl, hba1c, language):
    patient = PatientProfile(
        age=int(age),
        gender=Gender(gender.lower()),
        bmi=float(bmi) if bmi else None,
        has_diabetes=dm,
        ldl_mg_dl=float(ldl) if ldl else None,
        hba1c=float(hba1c) if hba1c else None,
        comorbidities=ComorbidityProfile(
            diabetes_mellitus=dm,
            hypertension=htn,
            current_smoker=smoking,
        ),
    )

    result = run_full_assessment(patient)
    phenotype = assess_south_asian_phenotype(patient)
    risk = result["risk_assessment"]
    targets = result["treatment_targets"]

    lang = "hi" if language == "Hindi" else "en"
    risk_level = risk.risk_category.value

    risk_explanation = RISK_EXPLANATIONS.get(risk_level, RISK_EXPLANATIONS["LOW"])[lang]

    output = f"## {risk_explanation}\n\n"
    output += LIFESTYLE_GUIDANCE[lang] + "\n"
    output += EMERGENCY_SIGNS[lang] + "\n"

    if lang == "en":
        if targets.specific_recommendations:
            output += "### Your Doctor May Recommend:\n"
            for rec in targets.specific_recommendations[:3]:
                output += f"- {rec}\n"
    else:
        if targets.specific_recommendations:
            output += "### Aapke Doctor Sujhav De Sakte Hain:\n"
            for rec in targets.specific_recommendations[:3]:
                output += f"- {rec}\n"

    return risk_level, output


def create_patient_portal():
    with gr.Blocks(title="Cardio-Sahayak: Heart Health Guide", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Cardio-Sahayak: Your Heart Health Guide")
        gr.Markdown("*Simple heart risk assessment -- not a substitute for your doctor*")

        language = gr.Radio(["English", "Hindi"], label="Language / Bhasha", value="English")

        with gr.Row():
            age = gr.Number(label="Your Age / Aapki Umar", value=50)
            gender = gr.Dropdown(["Male", "Female"], label="Gender / Ling", value="Male")
            bmi = gr.Number(label="BMI (if known)", value=None)

        gr.Markdown("### Health Conditions / Swasthya Sthiti")
        dm = gr.Checkbox(label="Diabetes / Sugar ki bimari")
        htn = gr.Checkbox(label="High Blood Pressure / Uchch Raktchap")
        smoking = gr.Checkbox(label="Tobacco use / Tambaaku sevan (bidi, cigarette, gutka)")

        gr.Markdown("### Lab Values (if available)")
        with gr.Row():
            ldl = gr.Number(label="LDL Cholesterol (mg/dl)", value=None)
            hba1c = gr.Number(label="HbA1c (%)", value=None)

        assess_btn = gr.Button("Check My Heart Risk / Mera Jokhim Jaanein", variant="primary", size="lg")

        risk_badge = gr.Textbox(label="Risk Level / Jokhim Star", interactive=False)
        output = gr.Markdown()

        assess_btn.click(
            fn=patient_assessment,
            inputs=[age, gender, bmi, dm, htn, smoking, ldl, hba1c, language],
            outputs=[risk_badge, output],
        )

        gr.Markdown("---")
        gr.Markdown("*Yah aapke doctor ki jagah nahi hai. Hamesha apne doctor se salah lein.*")

    return demo


if __name__ == "__main__":
    demo = create_patient_portal()
    demo.launch(server_name="0.0.0.0", server_port=7861)
