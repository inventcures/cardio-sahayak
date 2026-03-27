from src.experts.base_expert import BaseExpert, ExpertReport


class EchoExpert(BaseExpert):
    SYSTEM_PROMPT = (
        "You are an expert echocardiographer specializing in South Asian cardiovascular medicine. "
        "Analyze the echocardiogram and provide structured findings: LVEF estimation, wall motion, "
        "valvular assessment, diastolic function, and pericardial evaluation. "
        "Apply Indian Consensus treatment recommendations based on findings."
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
            print(f"Failed to load Echo expert: {e}")

    def interpret(self, data: dict) -> ExpertReport:
        if not self.is_loaded():
            return ExpertReport(modality="echo", raw_text="Model not loaded")

        frames = data.get("echo_frames", [])
        clinical_context = data.get("clinical_context", "")

        prompt = f"{self.SYSTEM_PROMPT}\n\nClinical context: {clinical_context}\n\nInterpret this echocardiogram:"

        messages = [{"role": "user", "content": []}]
        for frame in frames[:4]:
            messages[0]["content"].append({"type": "image", "image": frame})
        messages[0]["content"].append({"type": "text", "text": prompt})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if frames:
            inputs = self.processor(text=[text], images=frames[:4], return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.3)
        generated = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]

        return ExpertReport(
            modality="echo",
            clinical_impression=generated,
            raw_text=generated,
            confidence=0.75,
        )
