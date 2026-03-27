"""
Stage 3: Group Relative Policy Optimization (GRPO)

Binary correctness reward on MCQ datasets.
Dramatically improves accuracy (proven by MARCUS: 34-45% margin over frontier models).
Run: modal run src/training/stage3_grpo.py
"""
import os

try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

if HAS_MODAL:
    app = modal.App("cardio-sahayak-grpo")

    training_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.1.0",
            "transformers>=4.45.0",
            "peft>=0.7.0",
            "accelerate",
            "bitsandbytes",
            "trl>=0.10.0",
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
    def train_grpo(model_name: str, sft_adapter_path: str, mcq_dataset_path: str, hub_repo: str):
        _run_grpo(model_name, sft_adapter_path, mcq_dataset_path, hub_repo)


def binary_correctness_reward(completions: list[str], answers: list[str]) -> list[float]:
    rewards = []
    for completion, answer in zip(completions, answers):
        completion_clean = completion.strip().upper()
        answer_clean = answer.strip().upper()
        if answer_clean in completion_clean or completion_clean.startswith(answer_clean):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def _run_grpo(
    model_name: str,
    sft_adapter_path: str,
    mcq_dataset_path: str,
    hub_repo: str,
):
    import json
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, sft_adapter_path, is_trainable=True)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = []
    with open(mcq_dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    def format_mcq(record):
        question = record.get("question", "")
        options = record.get("options", [])
        correct = record.get("correct_answer", "")

        formatted = f"{question}\n\n"
        for i, opt in enumerate(options):
            formatted += f"{chr(65+i)}. {opt}\n"
        formatted += "\nAnswer with just the letter:"

        return {"prompt": formatted, "answer": correct}

    formatted = [format_mcq(r) for r in records]
    dataset = Dataset.from_list(formatted)

    def reward_fn(completions, prompts=None, **kwargs):
        batch_answers = [formatted[i]["answer"] for i in range(len(completions))]
        return binary_correctness_reward(completions, batch_answers)

    grpo_config = GRPOConfig(
        output_dir=f"grpo_checkpoints_{hub_repo.split('/')[-1]}",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=1e-6,
        num_generations=4,
        max_prompt_length=2048,
        max_completion_length=256,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        push_to_hub=True,
        hub_model_id=hub_repo,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    trainer.train()
    trainer.push_to_hub()
    print(f"GRPO training complete. Pushed to {hub_repo}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python stage3_grpo.py <model_name> <sft_adapter_path> <mcq_dataset_path> <hub_repo>")
        print("Example: python stage3_grpo.py Qwen/Qwen2.5-3B-Instruct tp53/cardio-sahayak-clinical-expert-v3 data/benchmarks/cardioqa_india_v1.jsonl tp53/cardio-sahayak-clinical-expert-v3-grpo")
        sys.exit(1)

    if HAS_MODAL:
        with app.run():
            train_grpo.remote(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        _run_grpo(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
