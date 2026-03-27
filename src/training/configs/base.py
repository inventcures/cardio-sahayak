from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints"
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 4096
    save_strategy: str = "epoch"
    logging_steps: int = 10
    bf16: bool = True
    gradient_checkpointing: bool = True
    push_to_hub: bool = True
    hub_model_id: str = ""


@dataclass
class GRPOConfig:
    num_epochs: int = 10
    per_device_train_batch_size: int = 4
    learning_rate: float = 1e-6
    group_size: int = 4
    kl_coeff: float = 0.01
    max_prompt_length: int = 2048
    max_completion_length: int = 512


ORCHESTRATOR_CONFIG = ModelConfig(model_name="Qwen/Qwen2.5-VL-7B-Instruct")
ECG_EXPERT_CONFIG = ModelConfig(model_name="Qwen/Qwen2.5-VL-3B-Instruct")
ECHO_EXPERT_CONFIG = ModelConfig(model_name="Qwen/Qwen2.5-VL-3B-Instruct")
CLINICAL_EXPERT_CONFIG = ModelConfig(model_name="Qwen/Qwen2.5-3B-Instruct")
