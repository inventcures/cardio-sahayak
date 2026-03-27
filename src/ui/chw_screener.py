import gradio as gr


SCREENING_QUESTIONS = [
    ("age_above_threshold", "Age > 45 (male) or > 55 (female)?"),
    ("known_diabetic", "Known diabetic / Sugar ki bimari?"),
    ("known_hypertensive", "Known hypertensive / BP ki dawai lete hain?"),
    ("tobacco_user", "Tobacco user (bidi, cigarette, gutka, khaini)?"),
    ("family_history", "Family member had heart attack before age 60?"),
    ("chest_pain_exertion", "Chest pain on walking or climbing?"),
    ("chest_pain_rest_relief", "Pain relieved by sitting/resting?"),
    ("breathlessness", "Breathlessness on mild activity?"),
]


def screen_patient(
    age_flag, diabetic, hypertensive, tobacco, family_hx,
    chest_pain, pain_relief, breathlessness, bp_sys, bp_dia, blood_sugar,
):
    score = sum([
        age_flag, diabetic, hypertensive, tobacco, family_hx,
        chest_pain, pain_relief, breathlessness,
    ])

    bp_high = False
    if bp_sys and bp_dia:
        bp_high = int(bp_sys) >= 140 or int(bp_dia) >= 90
        if bp_high:
            score += 1

    sugar_high = False
    if blood_sugar:
        sugar_high = float(blood_sugar) >= 200
        if sugar_high:
            score += 1

    if score >= 5 or (chest_pain and pain_relief):
        risk = "RED"
        color = "HIGH RISK / UCHCH JOKHIM"
        referral = "Refer to District Hospital IMMEDIATELY / Turant Zila Aspatal bhejein"
        actions = [
            "Call ambulance or arrange transport NOW",
            "Keep patient lying down comfortably",
            "Give aspirin 325mg if available and no allergy",
            "Do not let patient walk or exert",
            "Inform PHC medical officer immediately",
        ]
    elif score >= 3:
        risk = "YELLOW"
        color = "MODERATE RISK / MADHYAM JOKHIM"
        referral = "Refer to PHC within 1 week / PHC mein 1 hafte ke andar dikhayein"
        actions = [
            "Schedule appointment at PHC",
            "Advise patient to reduce salt and oil",
            "If diabetic, check blood sugar regularly",
            "If hypertensive, take BP medicines daily",
            "Follow up in 2 weeks",
        ]
    else:
        risk = "GREEN"
        color = "LOW RISK / KAM JOKHIM"
        referral = "Lifestyle counseling. Rescreen in 6 months / 6 maah baad dobara jaanch"
        actions = [
            "Counsel on reducing salt and tobacco",
            "Encourage 30 minutes daily walking",
            "Encourage fruits and vegetables",
            "Rescreen in 6 months",
        ]

    additional = []
    if bp_high:
        additional.append(f"BP {int(bp_sys)}/{int(bp_dia)} is HIGH - needs treatment")
    if sugar_high:
        additional.append(f"Random blood sugar {float(blood_sugar)} is HIGH - needs evaluation")

    action_text = f"""## {color}

### Referral: {referral}

### Action Items:
"""
    for i, action in enumerate(actions, 1):
        action_text += f"{i}. {action}\n"

    if additional:
        action_text += "\n### Additional Findings:\n"
        for finding in additional:
            action_text += f"- {finding}\n"

    action_text += f"\n**Screening Score: {score}/10**"

    return risk, action_text


def create_chw_screener():
    with gr.Blocks(title="Cardio-Sahayak: CHW Screening", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Cardio-Sahayak: Heart Screening Tool")
        gr.Markdown("*For ASHA / ANM / Health Workers*")

        gr.Markdown("### Screening Checklist / Jaanch Soochi")

        age_flag = gr.Checkbox(label="Age > 45 (male) or > 55 (female)")
        diabetic = gr.Checkbox(label="Known diabetic / Sugar ki bimari")
        hypertensive = gr.Checkbox(label="Known hypertensive / BP ki dawai")
        tobacco = gr.Checkbox(label="Tobacco user (bidi, cigarette, gutka)")
        family_hx = gr.Checkbox(label="Family member: heart attack before 60")
        chest_pain = gr.Checkbox(label="Chest pain on walking / climbing")
        pain_relief = gr.Checkbox(label="Pain goes away with rest")
        breathlessness = gr.Checkbox(label="Breathlessness on mild activity")

        gr.Markdown("### Measurements (if available)")
        with gr.Row():
            bp_sys = gr.Number(label="BP Systolic", value=None)
            bp_dia = gr.Number(label="BP Diastolic", value=None)
        blood_sugar = gr.Number(label="Random Blood Sugar (mg/dl)", value=None)

        screen_btn = gr.Button("Screen / Jaanch Karein", variant="primary", size="lg")

        risk_badge = gr.Textbox(label="Risk Level", interactive=False)
        output = gr.Markdown()

        screen_btn.click(
            fn=screen_patient,
            inputs=[
                age_flag, diabetic, hypertensive, tobacco, family_hx,
                chest_pain, pain_relief, breathlessness, bp_sys, bp_dia, blood_sugar,
            ],
            outputs=[risk_badge, output],
        )

        gr.Markdown("---")
        gr.Markdown("*This is a screening tool only. All RED and YELLOW patients must see a doctor.*")

    return demo


if __name__ == "__main__":
    demo = create_chw_screener()
    demo.launch(server_name="0.0.0.0", server_port=7862)
