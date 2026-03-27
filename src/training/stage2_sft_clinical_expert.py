"""
Stage 2 SFT: Clinical Expert Model

Fine-tunes Qwen2.5-3B-Instruct on Indian cardiology clinical data.
Data: V3 dataset (EkaCare filtered + synthetic vignettes)
Run: modal run src/training/stage2_sft_clinical_expert.py
"""
import os

try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

if HAS_MODAL:
    app = modal.App("cardio-sahayak-clinical-expert-sft")

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
            "huggingface_hub",
        )
    )

    @app.function(
        image=training_image,
        gpu="A100-80GB",
        timeout=86400,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    def train_clinical_expert():
        _run_training()


def _run_training():
    import json
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    HUB_REPO = "tp53/cardio-sahayak-clinical-expert-v3"
    DATASET_PATH = "data/processed_datasets/cardio_sahayak_india_instruct_v3.jsonl"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    def format_record(record):
        messages = [
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record.get("output", "")},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = Dataset.from_list([format_record(r) for r in records if r.get("output")])

    training_args = TrainingArguments(
        output_dir="clinical_expert_sft_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
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
        processing_class=tokenizer,
        max_seq_length=4096,
    )

    trainer.train()
    trainer.push_to_hub()
    print(f"Clinical Expert SFT complete. Pushed to {HUB_REPO}")


if __name__ == "__main__":
    if HAS_MODAL:
        with app.run():
            train_clinical_expert.remote()
    else:
        _run_training()
