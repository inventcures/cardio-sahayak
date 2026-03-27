"""
Edge inference runtime using llama-cpp-python.

Supports dynamic model loading/unloading to fit within constrained RAM.
Target: Indian clinic laptop with 8-16GB RAM.
"""
from pathlib import Path
from dataclasses import dataclass

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False


@dataclass
class EdgeConfig:
    orchestrator_path: str = ""
    ecg_expert_path: str = ""
    echo_expert_path: str = ""
    clinical_expert_path: str = ""
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0


class EdgeRuntime:
    def __init__(self, config: EdgeConfig):
        self.config = config
        self._loaded_models: dict[str, "Llama"] = {}

    def _load_model(self, name: str, path: str) -> "Llama":
        if not HAS_LLAMA_CPP:
            raise RuntimeError("llama-cpp-python not installed")

        if not Path(path).exists():
            raise FileNotFoundError(f"GGUF not found: {path}")

        if name in self._loaded_models:
            return self._loaded_models[name]

        model = Llama(
            model_path=path,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False,
        )
        self._loaded_models[name] = model
        return model

    def unload_model(self, name: str):
        if name in self._loaded_models:
            del self._loaded_models[name]

    def unload_all(self):
        self._loaded_models.clear()

    def generate(self, model_name: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        paths = {
            "orchestrator": self.config.orchestrator_path,
            "ecg_expert": self.config.ecg_expert_path,
            "echo_expert": self.config.echo_expert_path,
            "clinical_expert": self.config.clinical_expert_path,
        }

        path = paths.get(model_name, "")
        if not path:
            return f"No GGUF path configured for {model_name}"

        model = self._load_model(model_name, path)

        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|endoftext|>", "<|im_end|>"],
        )

        return output["choices"][0]["text"]

    def loaded_models(self) -> list[str]:
        return list(self._loaded_models.keys())
