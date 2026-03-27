"""
GGUF conversion for Qwen2.5 models.

Unlike the previous MedGemma pipeline (modal_gguf_convert.py) which required
fragile Gemma3->Gemma2 architectural patching, Qwen2.5 is natively supported
in llama.cpp.
"""
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass


@dataclass
class GGUFConfig:
    model_name: str
    adapter_path: str
    output_dir: str = "gguf_output"
    quant_types: list[str] = None
    hub_repo: str = ""

    def __post_init__(self):
        if self.quant_types is None:
            self.quant_types = ["Q4_K_M"]


MODELS = {
    "orchestrator": GGUFConfig(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        adapter_path="tp53/cardio-sahayak-orchestrator-v3",
        hub_repo="tp53/cardio-sahayak-v3-gguf",
    ),
    "ecg_expert": GGUFConfig(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        adapter_path="tp53/cardio-sahayak-ecg-expert-v3",
        hub_repo="tp53/cardio-sahayak-v3-gguf",
    ),
    "echo_expert": GGUFConfig(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        adapter_path="tp53/cardio-sahayak-echo-expert-v3",
        hub_repo="tp53/cardio-sahayak-v3-gguf",
    ),
    "clinical_expert": GGUFConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        adapter_path="tp53/cardio-sahayak-clinical-expert-v3",
        hub_repo="tp53/cardio-sahayak-v3-gguf",
    ),
}


def merge_adapter(model_name: str, adapter_path: str, output_dir: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {model_name}")
    base = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()

    merged_path = os.path.join(output_dir, "merged")
    os.makedirs(merged_path, exist_ok=True)
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved to {merged_path}")
    return merged_path


def convert_to_gguf(merged_path: str, output_path: str, llama_cpp_dir: str = "llama.cpp") -> str:
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")

    if not os.path.exists(convert_script):
        print(f"llama.cpp not found at {llama_cpp_dir}. Clone it first:")
        print(f"  git clone https://github.com/ggml-org/llama.cpp.git")
        return ""

    cmd = ["python3", convert_script, merged_path, "--outfile", output_path]
    print(f"Converting to GGUF: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_path


def quantize_gguf(input_path: str, output_path: str, quant_type: str = "Q4_K_M",
                  llama_cpp_dir: str = "llama.cpp") -> str:
    quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")

    if not os.path.exists(quantize_bin):
        print(f"llama-quantize not found. Build llama.cpp first:")
        print(f"  cd {llama_cpp_dir} && cmake -B build && cmake --build build --config Release")
        return ""

    cmd = [quantize_bin, input_path, output_path, quant_type]
    print(f"Quantizing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_path


def convert_model(config: GGUFConfig, llama_cpp_dir: str = "llama.cpp"):
    os.makedirs(config.output_dir, exist_ok=True)
    model_id = config.adapter_path.split("/")[-1]

    merged_path = merge_adapter(config.model_name, config.adapter_path, config.output_dir)

    fp16_path = os.path.join(config.output_dir, f"{model_id}-fp16.gguf")
    convert_to_gguf(merged_path, fp16_path, llama_cpp_dir)

    for quant in config.quant_types:
        quant_path = os.path.join(config.output_dir, f"{model_id}-{quant}.gguf")
        quantize_gguf(fp16_path, quant_path, quant, llama_cpp_dir)
        print(f"Quantized: {quant_path}")


def convert_all_models(llama_cpp_dir: str = "llama.cpp"):
    for name, config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Converting {name}: {config.model_name}")
        print(f"{'='*60}")
        convert_model(config, llama_cpp_dir)
