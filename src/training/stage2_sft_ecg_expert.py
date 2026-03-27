"""
Stage 2 SFT: ECG Expert Model

Fine-tunes Qwen2.5-VL-3B-Instruct on ECG image-report pairs.
Data: ECGBench + PTB-XL + MIMIC-IV-ECG images
Run: modal run src/training/stage2_sft_ecg_expert.py
"""
import os

try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

if HAS_MODAL:
    app = modal.App("cardio-sahayak-ecg-expert-sft")

    training_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.1.0",
            "transformers>=4.45.0",
            "peft>=0.7.0",
            "accelerate",
            "bitsandbytes",
            "trl>=0.7.0",
            "datasets",
            "qwen-vl-utils",
            "Pillow",
            "huggingface_hub",
        )
    )

    @app.function(
        image=training_image,
        gpu="A100-80GB",
        timeout=86400,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    def train_ecg_expert():
        _run_training()


def _run_training():
    import torch
    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset

    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    HUB_REPO = "tp53/cardio-sahayak-ecg-expert-v3"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    dataset = load_dataset("PULSE-ECG/ECGBench", "ptb-test-report", split="train")

    training_args = TrainingArguments(
        output_dir="ecg_expert_sft_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        push_to_hub=True,
        hub_model_id=HUB_REPO,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        max_seq_length=2048,
    )

    trainer.train()
    trainer.push_to_hub()
    print(f"ECG Expert SFT complete. Pushed to {HUB_REPO}")


if __name__ == "__main__":
    if HAS_MODAL:
        with app.run():
            train_ecg_expert.remote()
    else:
        _run_training()
