# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0", "bitsandbytes", "transformers", "accelerate", "datasets", "trackio"]
# ///

import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import trackio

def train():
    model_id = "google/medgemma-27b-it"
    dataset_id = "tp53/cardio-sahayak-india-instruct-v0"
    output_dir = "tp53/cardio-sahayak-india-v0"

    # 1. Load Dataset
    dataset = load_dataset(dataset_id)
    
    # 2. BitsAndBytes Config (4-bit quantization for 27B model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 3. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Training Arguments
    args = SFTConfig(
        output_dir=output_dir,
        push_to_hub=True,
        hub_model_id=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        bf16=True,
        report_to="trackio",
        project="cardio-sahayak-india",
        run_name="medgemma-27b-fine-tune",
        max_length=1024,
    )

    # 5. Initialize Trainer
    trainer = SFTTrainer(
        model=model_id,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        args=args,
        tokenizer=AutoTokenizer.from_pretrained(model_id),
    )

    # 6. Start Training
    trainer.train()

    # 7. Push to Hub
    trainer.push_to_hub()

if __name__ == "__main__":
    train()
