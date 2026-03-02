import os
import modal

# 1. Define the Modal App
app = modal.App("cardio-sahayak-vlm-finetune")

# 2. Define the Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "trl>=0.12.0",
        "peft>=0.7.0",
        "bitsandbytes",
        "transformers>=4.45.0", # Ensure latest for Gemma 3
        "accelerate",
        "datasets",
        "huggingface_hub",
        "hf_transfer",
        "Pillow",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# 3. Define the Training Function
@app.function(
    image=image,
    gpu="A100-80GB", # High VRAM for 27B multimodal
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_vlm():
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor, 
        Gemma3ForConditionalGeneration, 
        BitsAndBytesConfig,
        Trainer
    )
    from trl import SFTConfig
    from huggingface_hub import login
    from PIL import Image

    # Ensure HF Token is set
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN secret not found.")
    login(token=hf_token)

    model_id = "google/medgemma-27b-it"
    dataset_id = "PULSE-ECG/ECGBench"
    dataset_subset = "ptb-test-report" # High-quality reports
    output_dir = "cardio-sahayak-vlm-output"
    hub_model_id = "tp53/cardio-sahayak-vlm"

    print(f"Loading processor and model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token
    )

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Data Preparation
    print(f"Loading dataset: {dataset_id}/{dataset_subset}")
    # ECGBench already has images as PIL objects in the 'image' column
    full_dataset = load_dataset(dataset_id, dataset_subset, split="test", token=hf_token)

    def format_for_trainer(example):
        messages = []
        for i, turn in enumerate(example["conversations"]):
            role = "user" if turn["from"] == "human" else "assistant"
            content = []
            
            # Add image only to the first user turn
            if i == 0 and role == "user":
                content.append({"type": "image"})
            
            # Clean up the text just in case there are <image> tags
            text_val = turn["value"].replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            content.append({"type": "text", "text": text_val})
            
            messages.append({"role": role, "content": content})
        
        return {"messages": messages, "images": [example["image"]]}

    print("Formatting dataset...")
    # Map once and remove original columns
    processed_ds = full_dataset.map(format_for_trainer, remove_columns=full_dataset.column_names)
    train_ds = processed_ds.train_test_split(test_size=0.1)

    # 6. Training Arguments
    args = SFTConfig(
        output_dir=output_dir,
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_private_repo=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=1,
        max_length=2048,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Custom Data Collator for VLM
    def collate_fn(examples):
        texts = [processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False) for ex in examples]
        images = [ex["images"] for ex in examples]
        
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(torch.bfloat16)
        
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds["train"],
        eval_dataset=train_ds["test"],
        data_collator=collate_fn,
    )

    print("Starting VLM Multimodal Tuning...")
    trainer.train()

    print(f"Pushing VLM adapters to {hub_model_id}")
    trainer.push_to_hub()
    print("VLM Training and push complete!")

@app.local_entrypoint()
def main():
    train_vlm.remote()
