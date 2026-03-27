from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.knowledge.schemas import (
    RiskCategory,
    ReferralUrgency,
    ReferralDestination,
    DiamondResult,
)


@dataclass
class STChange:
    leads: str
    change_type: str
    magnitude_mm: float | None = None


@dataclass
class TWaveChange:
    leads: str
    change_type: str


@dataclass
class ECGReport:
    rate_bpm: int | None = None
    rhythm: str | None = None
    pr_interval_ms: int | None = None
    qrs_duration_ms: int | None = None
    qt_ms: int | None = None
    qtc_ms: int | None = None
    axis: str | None = None
    st_changes: list[STChange] = field(default_factory=list)
    t_wave_changes: list[TWaveChange] = field(default_factory=list)
    pathological_q_waves: list[str] = field(default_factory=list)
    lvh_criteria: bool = False
    clinical_impression: str = ""
    urgency: ReferralUrgency | None = None
    confidence: float = 0.0


@dataclass
class WallMotionAbnormality:
    segment: str
    status: str


@dataclass
class ValvularFinding:
    valve: str
    regurgitation: str | None = None
    stenosis: str | None = None


@dataclass
class EchoReport:
    lvef_percent: float | None = None
    lvidd_mm: float | None = None
    lvids_mm: float | None = None
    wall_motion_abnormalities: list[WallMotionAbnormality] = field(default_factory=list)
    valvular_findings: list[ValvularFinding] = field(default_factory=list)
    diastolic_grade: str | None = None
    e_a_ratio: float | None = None
    e_prime: float | None = None
    pericardial_effusion: bool = False
    clinical_impression: str = ""
    confidence: float = 0.0


@dataclass
class LabAbnormality:
    test: str
    value: float
    unit: str
    target: float | None = None
    status: str = "normal"


@dataclass
class ClinicalReport:
    risk_factors_present: list[str] = field(default_factory=list)
    lab_abnormalities: list[LabAbnormality] = field(default_factory=list)
    drug_interactions: list[str] = field(default_factory=list)
    treatment_gaps: list[str] = field(default_factory=list)
    south_asian_flags: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class MedicationRecommendation:
    drug_class: str
    specific_drug: str
    dose: str
    rationale: str
    guideline_source: str
    nlem_available: bool = True


@dataclass
class ReferralDecision:
    urgency: ReferralUrgency
    destination: ReferralDestination
    reason: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class MirageCheckResult:
    ecg_mirage_detected: bool = False
    echo_mirage_detected: bool = False
    ecg_confidence: float = 1.0
    echo_confidence: float = 1.0
    cross_check_contradictions: list[str] = field(default_factory=list)


@dataclass
class CardioSahayakOutput:
    patient_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""

    age: int = 0
    gender: str = ""
    bmi: float | None = None
    south_asian_bmi_category: str | None = None

    risk_category: RiskCategory = RiskCategory.LOW
    annual_cv_mortality_estimate: str | None = None
    chest_pain_score: int | None = None
    comorbidity_score: int = 0
    comorbidities_present: list[str] = field(default_factory=list)

    ecg_report: ECGReport | None = None
    echo_report: EchoReport | None = None
    clinical_report: ClinicalReport | None = None

    treatment_plan: DiamondResult | None = None
    referral_decision: ReferralDecision | None = None
    medication_recommendations: list[MedicationRecommendation] = field(default_factory=list)
    contraindicated_medications: list[str] = field(default_factory=list)
    treatment_gaps: list[str] = field(default_factory=list)

    ldl_target_mg_dl: float = 70.0
    bp_target: str = "<=130/80"
    hba1c_target: float | None = None

    confidence_scores: dict[str, float] = field(default_factory=dict)
    mirage_check: MirageCheckResult | None = None
    evidence_sources: list[str] = field(default_factory=list)

    doctor_summary: str = ""
    patient_summary_en: str = ""
    patient_summary_hi: str | None = None
    chw_action_items: list[str] = field(default_factory=list)
