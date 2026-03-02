import os
import modal

# 1. Define the Modal App
app = modal.App("cardio-sahayak-finetune")

# 2. Define the Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "trl>=0.12.0",
        "peft>=0.7.0",
        "bitsandbytes",
        "transformers",
        "accelerate",
        "datasets",
        "trackio",
        "huggingface_hub",
        "hf_transfer", # For faster downloads
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# 3. Define the Training Function
@app.function(
    image=image,
    gpu="A100-80GB", # Recommended format
    timeout=86400,   # 24 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train():
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig
    import trackio
    from huggingface_hub import login

    # Ensure HF Token is set
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN secret not found in Modal environment.")
    login(token=hf_token)

    model_id = "google/medgemma-27b-it"
    dataset_id = "tp53/cardio-sahayak-india-instruct-v0"
    output_dir = "cardio-sahayak-output" # Local dir in container
    hub_model_id = "tp53/cardio-sahayak"  # Target HF repo

    print(f"Loading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Gemma 3 requires token_type_ids. We'll pre-tokenize the dataset.
    def preprocess_function(examples):
        # Format messages into a single string using the tokenizer's chat template
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
        # Tokenize and ensure token_type_ids are returned
        batch = tokenizer(texts, truncation=True, max_length=1024, return_token_type_ids=True)
        # For causal LM, labels are usually the same as input_ids
        batch["labels"] = batch["input_ids"].copy()
        return batch

    print("Pre-tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    # Use a custom data collator to ensure token_type_ids and labels are passed to the model
    def custom_data_collator(features):
        import torch
        from transformers.tokenization_utils_base import BatchEncoding
        
        # Extract labels and other keys
        labels = [feature.pop("labels", None) for feature in features]
        
        # Use tokenizer.pad for the remaining features (input_ids, attention_mask, token_type_ids)
        batch = tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        # Manually pad labels
        if labels[0] is not None:
            max_label_length = batch["input_ids"].shape[1]
            padding_side = tokenizer.padding_side
            
            padded_labels = []
            for label in labels:
                remainder = max_label_length - len(label)
                if padding_side == "right":
                    padded_labels.append(label + [-100] * remainder)
                else:
                    padded_labels.append([-100] * remainder + label)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        if "token_type_ids" not in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
            
        return batch

    # BitsAndBytes Config (4-bit quantization for 27B model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Training Arguments
    args = SFTConfig(
        output_dir=output_dir,
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_private_repo=True,
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
        project="cardio-sahayak-india",
        run_name="medgemma-27b-modal-fine-tune",
        max_length=1024,
    )

    print(f"Initializing Trainer for {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        peft_config=peft_config,
        args=args,
        data_collator=custom_data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Pushing final model to {hub_model_id}")
    trainer.push_to_hub()
    print("Training and push complete!")

@app.local_entrypoint()
def main():
    train.remote()
