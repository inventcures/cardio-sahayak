from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExpertReport:
    modality: str
    findings: dict = field(default_factory=dict)
    clinical_impression: str = ""
    confidence: float = 0.0
    raw_text: str = ""


class BaseExpert(ABC):
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def interpret(self, data: dict) -> ExpertReport:
        pass

    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self):
        self.model = None
        self.processor = None
