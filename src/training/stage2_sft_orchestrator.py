"""
Stage 2 SFT: Orchestrator Model

Fine-tunes Qwen2.5-VL-7B-Instruct for multi-expert synthesis and tool-calling.
Data: Multi-expert synthesis examples (expert reports -> unified assessment)
Run: modal run src/training/stage2_sft_orchestrator.py
"""
import os

try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

if HAS_MODAL:
    app = modal.App("cardio-sahayak-orchestrator-sft")

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
    def train_orchestrator():
        _run_training()


def _run_training():
    import json
    import torch
    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    HUB_REPO = "tp53/cardio-sahayak-orchestrator-v3"

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

    SYNTHESIS_DATA = "data/processed_datasets/orchestrator_synthesis_v3.jsonl"

    records = []
    if os.path.exists(SYNTHESIS_DATA):
        with open(SYNTHESIS_DATA) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if not records:
        print("No synthesis training data found. Generate it first.")
        print("Expected: data/processed_datasets/orchestrator_synthesis_v3.jsonl")
        return

    SYSTEM_PROMPT = (
        "You are a cardiac clinical decision support system for Indian patients. "
        "You receive reports from specialist models (ECG, Echo, Clinical) and "
        "synthesize them into a unified assessment using Indian Consensus guidelines. "
        "Output structured JSON with risk category, treatment plan, and referral decision."
    )

    def format_record(record):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record.get("output", "")},
        ]
        return {"text": processor.tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = Dataset.from_list([format_record(r) for r in records if r.get("output")])

    training_args = TrainingArguments(
        output_dir="orchestrator_sft_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=1,
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
        max_seq_length=4096,
    )

    trainer.train()
    trainer.push_to_hub()
    print(f"Orchestrator SFT complete. Pushed to {HUB_REPO}")


if __name__ == "__main__":
    if HAS_MODAL:
        with app.run():
            train_orchestrator.remote()
    else:
        _run_training()
