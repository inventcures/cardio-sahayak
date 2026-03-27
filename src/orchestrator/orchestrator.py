from dataclasses import dataclass, field

from src.experts.base_expert import ExpertReport
from src.experts.ecg_expert import ECGExpert
from src.experts.echo_expert import EchoExpert
from src.experts.clinical_expert import ClinicalExpert
from src.knowledge.indian_guidelines import run_full_assessment, detect_clinical_conditions
from src.knowledge.schemas import PatientProfile, ChestPainInput


@dataclass
class OrchestratorConfig:
    ecg_model_path: str = "tp53/cardio-sahayak-ecg-expert-v3"
    echo_model_path: str = "tp53/cardio-sahayak-echo-expert-v3"
    clinical_model_path: str = "tp53/cardio-sahayak-clinical-expert-v3"
    device: str = "auto"
    lazy_load: bool = True


class CardioSahayakOrchestrator:
    def __init__(self, config: OrchestratorConfig = OrchestratorConfig()):
        self.config = config
        self.ecg_expert = ECGExpert(config.ecg_model_path, config.device)
        self.echo_expert = EchoExpert(config.echo_model_path, config.device)
        self.clinical_expert = ClinicalExpert(config.clinical_model_path, config.device)

        if not config.lazy_load:
            self.load_all_experts()

    def load_all_experts(self):
        self.ecg_expert.load_model()
        self.echo_expert.load_model()
        self.clinical_expert.load_model()

    def process_case(
        self,
        patient: PatientProfile,
        chest_pain: ChestPainInput | None = None,
        ecg_image=None,
        echo_frames: list | None = None,
        vitals: dict | None = None,
        labs: dict | None = None,
        medications: list[str] | None = None,
    ) -> dict:
        expert_reports: list[ExpertReport] = []

        clinical_context = self._build_clinical_context(patient)

        if ecg_image is not None:
            if not self.ecg_expert.is_loaded():
                self.ecg_expert.load_model()
            ecg_report = self.ecg_expert.interpret({
                "ecg_image": ecg_image,
                "clinical_context": clinical_context,
            })
            expert_reports.append(ecg_report)

        if echo_frames:
            if not self.echo_expert.is_loaded():
                self.echo_expert.load_model()
            echo_report = self.echo_expert.interpret({
                "echo_frames": echo_frames,
                "clinical_context": clinical_context,
            })
            expert_reports.append(echo_report)

        if vitals or labs or medications:
            if not self.clinical_expert.is_loaded():
                self.clinical_expert.load_model()
            clinical_report = self.clinical_expert.interpret({
                "clinical_context": clinical_context,
                "vitals": vitals or {},
                "labs": labs or {},
                "medications": medications or [],
            })
            expert_reports.append(clinical_report)

        guideline_assessment = run_full_assessment(patient, chest_pain)

        return {
            "expert_reports": expert_reports,
            "guideline_assessment": guideline_assessment,
            "patient": patient,
        }

    def _build_clinical_context(self, patient: PatientProfile) -> str:
        parts = [
            f"{patient.age}-year-old {patient.gender.value}",
        ]
        if patient.bmi is not None:
            parts.append(f"BMI {patient.bmi}")
        if patient.has_diabetes:
            parts.append("diabetic")
        if patient.comorbidities.hypertension:
            parts.append("hypertensive")
        if patient.lvef_percent is not None:
            parts.append(f"LVEF {patient.lvef_percent}%")
        if patient.troponin_elevated:
            parts.append("elevated troponin")
        return ", ".join(parts)
