from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class RiskCategory(Enum):
    HIGH = "HIGH"
    INTERMEDIATE = "INTERMEDIATE"
    LOW = "LOW"


class ReferralUrgency(Enum):
    EMERGENCY = "EMERGENCY"
    URGENT = "URGENT"
    ROUTINE = "ROUTINE"
    MANAGE_AT_PHC = "MANAGE_AT_PHC"


class ReferralDestination(Enum):
    CATHLAB = "CATHLAB"
    CARDIOLOGY_OPD = "CARDIOLOGY_OPD"
    DISTRICT_HOSPITAL = "DISTRICT_HOSPITAL"
    CHC = "CHC"
    PHC = "PHC"


class BMICategory(Enum):
    NORMAL = "normal"
    OVERWEIGHT = "overweight"
    OBESE = "obese"


# --- Chest Pain Scoring enums (IJAM 2023 Table 1) ---

class ChestPainPrecipitant(Enum):
    EXERTION_RELIEVED_BY_REST = "exertion_relieved_by_rest"
    EMOTIONAL_COLD_MEAL = "emotional_cold_meal"
    UNPREDICTABLE = "unpredictable"
    BREATHING = "breathing"


class ChestPainLocation(Enum):
    RETROSTERNAL_NECK_SHOULDER_JAW_ARM_EPIGASTRIC = "retrosternal_neck_shoulder_jaw_arm_epigastric"
    RIGHT_SIDE_SUBMAMMARY_LOCALIZED = "right_side_submammary_localized"


class ChestPainType(Enum):
    CONSTRICTING_CRAMPING_HEAVY_TIGHT_BURNING_DULL = "constricting_cramping_heavy_tight_burning_dull"
    STABBING_SHARP = "stabbing_sharp"
    REPRODUCIBLE_BY_PALPATION = "reproducible_by_palpation"


class ChestPainDuration(Enum):
    LESS_THAN_15_MIN = "less_than_15_min"
    FEW_SECONDS = "few_seconds"
    MORE_THAN_15_MIN = "more_than_15_min"


@dataclass
class ChestPainInput:
    precipitant: ChestPainPrecipitant
    location: ChestPainLocation
    pain_type: ChestPainType
    duration: ChestPainDuration


@dataclass
class ChestPainResult:
    score: int
    probability: str
    recommendation: str


# --- Comorbidity Checklist (IJAM 2023 Table 2) ---

@dataclass
class ComorbidityProfile:
    diabetes_mellitus: bool = False
    cholesterol_gt_250: bool = False
    current_smoker: bool = False
    family_history_cad_lt_60: bool = False
    hypertension: bool = False
    past_ihd: bool = False


@dataclass
class ComorbidityResult:
    score: int
    risk_level: str
    requires_cardiology_referral: bool
    factors_present: list[str]


# --- Drug Classes and Diamond Approach ---

class DrugClass(Enum):
    BB = "bb"
    DHP = "dhp"
    VER_DILT = "ver_dilt"
    IVAB = "ivab"
    NIC = "nic"
    NITR = "nitr"
    RAN = "ran"
    TRIM = "trim"
    ACEI = "acei"
    ARB = "arb"
    STATIN = "statin"
    SGLT2I = "sglt2i"
    GLP1RA = "glp1ra"
    ANTIPLATELET = "antiplatelet"


DRUG_CLASS_NAMES: dict[DrugClass, str] = {
    DrugClass.BB: "Beta-blockers (metoprolol, bisoprolol, carvedilol)",
    DrugClass.DHP: "DHP CCBs (amlodipine, nifedipine)",
    DrugClass.VER_DILT: "Verapamil / Diltiazem",
    DrugClass.IVAB: "Ivabradine",
    DrugClass.NIC: "Nicorandil",
    DrugClass.NITR: "Nitrates (isosorbide mononitrate/dinitrate, GTN)",
    DrugClass.RAN: "Ranolazine",
    DrugClass.TRIM: "Trimetazidine",
    DrugClass.ACEI: "ACE Inhibitors (ramipril, enalapril, perindopril)",
    DrugClass.ARB: "ARBs (telmisartan, losartan, valsartan)",
    DrugClass.STATIN: "Statins (atorvastatin, rosuvastatin)",
    DrugClass.SGLT2I: "SGLT2 Inhibitors (empagliflozin, dapagliflozin)",
    DrugClass.GLP1RA: "GLP-1 Receptor Agonists (liraglutide, semaglutide)",
    DrugClass.ANTIPLATELET: "Antiplatelets (aspirin, clopidogrel, ticagrelor)",
}


class ClinicalCondition(Enum):
    HIGH_HR = "high_hr"
    BRADYCARDIA = "bradycardia"
    HYPERTENSION = "hypertension"
    HYPOTENSION = "hypotension"
    LV_DYSFUNCTION = "lv_dysfunction"
    HEART_FAILURE = "heart_failure"
    ATRIAL_FIBRILLATION = "atrial_fibrillation"


@dataclass
class DiamondResult:
    conditions: list[ClinicalCondition]
    preferred: list[DrugClass]
    acceptable: list[DrugClass]
    contraindicated: list[DrugClass]
    rationale: list[str]


# --- Treatment Targets ---

@dataclass
class TreatmentTargets:
    ldl_target_mg_dl: float
    bp_systolic_target: int
    bp_diastolic_target: int
    hba1c_target: float | None = None
    specific_recommendations: list[str] = field(default_factory=list)


# --- Patient Profile ---

@dataclass
class PatientProfile:
    age: int
    gender: Gender
    weight_kg: float | None = None
    height_cm: float | None = None
    bmi: float | None = None
    waist_circumference_cm: float | None = None
    heart_rate_bpm: int | None = None
    bp_systolic: int | None = None
    bp_diastolic: int | None = None
    lvef_percent: float | None = None
    comorbidities: ComorbidityProfile = field(default_factory=ComorbidityProfile)
    has_diabetes: bool = False
    has_ckd: bool = False
    egfr: float | None = None
    hba1c: float | None = None
    ldl_mg_dl: float | None = None
    troponin_elevated: bool = False
    bnp_elevated: bool = False
    current_medications: list[str] = field(default_factory=list)


# --- Risk Assessment ---

@dataclass
class RiskAssessment:
    risk_category: RiskCategory
    chest_pain_score: int | None = None
    comorbidity_score: int = 0
    annual_cv_mortality_estimate: str | None = None
    factors: list[str] = field(default_factory=list)
    recommendation: str = ""
