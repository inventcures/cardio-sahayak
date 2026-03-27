from src.experts.base_expert import BaseExpert, ExpertReport


class ClinicalExpert(BaseExpert):
    SYSTEM_PROMPT = (
        "You are a clinical data analyst specializing in Indian cardiovascular medicine. "
        "Analyze the patient's vitals, lab values, medications, and history. "
        "Identify risk factors, treatment gaps, and drug interactions. "
        "Apply Indian Consensus guidelines: LDL <70 for diabetic CAD, BP <=130/80, "
        "SGLT2i/GLP-1RA for diabetic CVD, South Asian BMI thresholds (>=23 overweight)."
    )

    def load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.processor = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.processor.pad_token is None:
                self.processor.pad_token = self.processor.eos_token
        except Exception as e:
            print(f"Failed to load Clinical expert: {e}")

    def interpret(self, data: dict) -> ExpertReport:
        if not self.is_loaded():
            return ExpertReport(modality="clinical", raw_text="Model not loaded")

        clinical_context = data.get("clinical_context", "")
        vitals = data.get("vitals", {})
        labs = data.get("labs", {})
        medications = data.get("medications", [])

        patient_data = f"""Patient Data:
Vitals: {vitals}
Labs: {labs}
Current Medications: {', '.join(medications) if medications else 'None listed'}
Clinical Context: {clinical_context}"""

        prompt = f"{self.SYSTEM_PROMPT}\n\n{patient_data}\n\nProvide clinical assessment:"

        messages = [
            {"role": "user", "content": prompt},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.3)
        generated = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]

        return ExpertReport(
            modality="clinical",
            clinical_impression=generated,
            raw_text=generated,
            confidence=0.90,
        )
