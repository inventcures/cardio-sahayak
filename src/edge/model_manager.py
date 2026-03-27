import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ModelInfo:
    name: str
    gguf_path: str
    size_gb: float
    priority: int


ESTIMATED_SIZES = {
    "orchestrator": 4.5,
    "ecg_expert": 2.0,
    "echo_expert": 2.0,
    "clinical_expert": 2.0,
}


def get_available_ram_gb() -> float:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except ImportError:
        return 8.0


def get_loadable_models(
    available_models: list[ModelInfo],
    ram_budget_gb: float | None = None,
) -> list[ModelInfo]:
    if ram_budget_gb is None:
        ram_budget_gb = get_available_ram_gb() * 0.8

    sorted_models = sorted(available_models, key=lambda m: m.priority)
    loadable = []
    remaining = ram_budget_gb

    for model in sorted_models:
        if model.size_gb <= remaining:
            loadable.append(model)
            remaining -= model.size_gb

    return loadable


def discover_gguf_models(gguf_dir: str) -> list[ModelInfo]:
    models = []
    gguf_path = Path(gguf_dir)

    if not gguf_path.exists():
        return models

    for gguf_file in gguf_path.glob("*.gguf"):
        name = gguf_file.stem.split("-Q")[0]
        size_gb = gguf_file.stat().st_size / (1024 ** 3)

        priority = 2
        if "orchestrator" in name:
            priority = 1

        models.append(ModelInfo(
            name=name,
            gguf_path=str(gguf_file),
            size_gb=round(size_gb, 2),
            priority=priority,
        ))

    return models
