from src.experts.base_expert import BaseExpert, ExpertReport


class ECGExpert(BaseExpert):
    SYSTEM_PROMPT = (
        "You are an expert ECG interpreter specializing in South Asian cardiovascular medicine. "
        "Analyze the 12-lead ECG and provide structured findings: rate, rhythm, intervals (PR, QRS, QT/QTc), "
        "axis, ST-T changes, and clinical impression. Flag South Asian-specific concerns "
        "(premature CAD, lower BMI thresholds, MYBPC3 screening if HCM suspected)."
    )

    def load_model(self):
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
            import torch

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load ECG expert: {e}")

    def interpret(self, data: dict) -> ExpertReport:
        if not self.is_loaded():
            return ExpertReport(modality="ecg", raw_text="Model not loaded")

        image = data.get("ecg_image")
        clinical_context = data.get("clinical_context", "")

        prompt = f"{self.SYSTEM_PROMPT}\n\nClinical context: {clinical_context}\n\nInterpret this ECG:"

        messages = [{"role": "user", "content": []}]
        if image is not None:
            messages[0]["content"].append({"type": "image", "image": image})
        messages[0]["content"].append({"type": "text", "text": prompt})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if image is not None:
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.3)
        generated = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]

        return ExpertReport(
            modality="ecg",
            clinical_impression=generated,
            raw_text=generated,
            confidence=0.85,
        )
