# Cardio-Sahayak Next-Gen: Detailed Specifications (v3)

**Date:** 27 March 2026
**Status:** Design Specification
**Authors:** TP53 + Claude Opus 4.6 Deep Research

---

## Executive Summary

Cardio-Sahayak Next-Gen transforms the current monolithic 27B VLM into a **MARCUS-inspired agentic multi-expert system** purpose-built for the Indian cardiovascular disease burden. The redesign is driven by three convergent insights:

1. **MARCUS (Stanford/UCSF, March 2026)** proved that small (3B) modality-specific expert VLMs coordinated by an LLM orchestrator outperform GPT-5 and Gemini 2.5 Pro by 34-45 percentage points on cardiac interpretation tasks -- architecture matters more than scale.

2. **Indian Consensus on CV Risk Stratification (IJAM 2023)** provides actionable clinical protocols (Diamond Approach, Chest Pain Scoring, comorbidity-based risk grading) that no digital health system has yet implemented -- Cardio-Sahayak fills this gap.

3. **India's disease burden demands it**: 50-60% of CAD patients also have diabetes, cardiologist access is minimal, GPs are the frontline, and 17.9 million die annually from CVD globally with India disproportionately affected.

The next-gen system targets **three user classes**: cardiologists/GPs (clinical decision support), patients/caregivers (health education in Hindi/English), and community health workers (screening checklists). It processes ECG images, echocardiograms, and structured clinical data through dedicated expert models, synthesized by an agentic orchestrator with built-in mirage detection and deterministic Indian guideline enforcement.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Agentic Architecture](#2-agentic-architecture)
3. [Clinical Knowledge Engine](#3-clinical-knowledge-engine)
4. [Multimodal Pipeline](#4-multimodal-pipeline)
5. [Data Strategy](#5-data-strategy)
6. [Training Pipeline](#6-training-pipeline)
7. [User Interfaces](#7-user-interfaces)
8. [Structured Output Engine](#8-structured-output-engine)
9. [Safety and Mirage Detection](#9-safety-and-mirage-detection)
10. [Edge Deployment](#10-edge-deployment)
11. [Evaluation and Validation](#11-evaluation-and-validation)
12. [Regulatory and Ethics](#12-regulatory-and-ethics)
13. [Implementation Roadmap](#13-implementation-roadmap)
14. [Project Directory Structure](#14-project-directory-structure)
15. [Key Technical Decisions](#15-key-technical-decisions)

---

## 1. Current State Analysis

### 1.1 What Exists Today

| Component | Current State | Limitation |
|-----------|--------------|------------|
| **Base Model** | `google/medgemma-27b-it` (27B) | Too large for edge (16.6GB GGUF); requires fragile Gemma3->Gemma2 architectural patching for llama.cpp |
| **Vision** | `google/medsiglip-448` zero-shot + VLM adapters on ECGBench | ECG-only; no echo, no CMR, no structured clinical data |
| **Training** | 2-stage SFT only (Phase 1 general + Phase 2 Indian contextualization) | No reinforcement learning (GRPO); MARCUS showed RL is critical for accuracy |
| **Phase 2 Dataset** | 166 records (156 EkaCare + 2 synthetic + 8 mock ECG) | Critically small; many EkaCare records are non-cardiology (appendicitis, paracetamol) |
| **Data Scraping** | Mock scripts (`scrape_indian_ecg_portals.py`, `download_mimic_ecg.py`) | Generate placeholder metadata only; zero real data acquired |
| **Output** | Free-text generation via single `model.generate()` call | No structured output, no risk scores, no guideline-aligned recommendations |
| **Architecture** | Monolithic -- text and VLM paths are independent | No orchestration, no agentic behavior, no multi-expert synthesis |
| **Safety** | None | No mirage detection, no confidence scoring, no audit trail |
| **UI** | Two nearly identical Gradio apps (ECG upload -> text report) | No role-based interface, no patient mode, no Hindi support |
| **Evaluation** | `modal_eval_cardio_sahayak.py` for MedQA/PubMedQA | Generic benchmarks; no cardiology-specific or India-specific evaluation; no results published |

### 1.2 Critical Files in Current System

- `app.py` (lines 55-97): Main Gradio inference loop -- single `model.generate()` call, no orchestration
- `modal_train_cardio_sahayak_v2.py`: Phase 2 QLoRA SFT patterns (Modal.com deployment, adapter resumption)
- `modal_train_vlm_cardio_sahayak.py` (lines 94-150): Multimodal collation for Gemma3ForConditionalGeneration
- `compile_v2_dataset.py`: Dataset aggregation pattern (to be extended with quality filtering)
- `modal_gguf_convert.py` (lines 78-96): Gemma3 architectural patching workaround (to be eliminated)
- `synthetic_phenotype_shifter.py`: Phenotype shift prompting approach (to be scaled)

---

## 2. Agentic Architecture

### 2.1 Design Rationale

MARCUS (Stanford/UCSF, 2603.22179v1) demonstrated that a hierarchical architecture of modality-specific expert VLMs coordinated by an LLM orchestrator achieves:

- **87-91% MCQ accuracy on ECG** (vs 35-48% for GPT-5/Gemini)
- **67-86% on echocardiography** (vs 22-35% for frontier models)
- **70% on multimodal cases** (vs 22-28% for GPT-5/Gemini)
- **0% mirage rate** through counterfactual probing

The key architectural insight: **natural language communication between experts and orchestrator** (not shared embedding spaces) enables future-proofing, mirage detection, and modular expert replacement.

### 2.2 System Architecture

```
                    +------------------------+
                    |   USER INTERFACE LAYER  |
                    |  Doctor | Patient | CHW |
                    +------------+-----------+
                                 |
                    +------------v-----------+
                    |   AGENTIC ORCHESTRATOR  |
                    |  Qwen2.5-VL-7B-Instruct |
                    |  (QLoRA, 3-stage trained)|
                    +--+--------+--------+---+
                       |        |        |
              +--------v--+ +---v----+ +-v----------+
              | ECG Expert| | Echo   | | Clinical   |
              | Qwen2.5-  | | Expert | | Expert     |
              | VL-3B     | | Qwen2.5| | Qwen2.5-3B |
              | (ECG SFT  | | -VL-3B | | (Text-only |
              |  + GRPO)  | | (Echo  | |  SFT+GRPO) |
              +-----+-----+ | SFT+  | +-----+------+
                    |        | GRPO)  |       |
                    |        +---+----+       |
                    |            |             |
              +-----v-----+ +---v----+ +------v-----+
              | MedSigLIP  | | Video  | | Structured |
              | 448 Encoder| | Frames | | Lab/Vitals |
              +------------+ +--------+ +------------+
                                 |
                    +------------v-----------+
                    | CLINICAL KNOWLEDGE     |
                    | ENGINE (Deterministic) |
                    | - Diamond Approach     |
                    | - Chest Pain Scoring   |
                    | - Risk Stratification  |
                    | - Drug Contraindications|
                    +------------------------+
                                 |
                    +------------v-----------+
                    |   SAFETY LAYER         |
                    | - Mirage Detection     |
                    | - Confidence Scoring   |
                    | - Guideline Cross-Check|
                    | - Audit Logging        |
                    +------------------------+
                                 |
                    +------------v-----------+
                    | STRUCTURED OUTPUT      |
                    | ENGINE (Pydantic JSON) |
                    +------------------------+
```

### 2.3 Orchestrator Workflow

The orchestrator receives a **case bundle** (patient demographics, symptoms, images, lab values) and executes:

1. **Triage**: Determine which experts to invoke based on available data
2. **Expert Dispatch**: Route ECG images to ECG expert, echo videos to Echo expert, clinical data to Clinical expert -- all in parallel
3. **Report Collection**: Gather natural language reports from each expert
4. **Mirage Detection**: Run counterfactual probes on all image-based reports
5. **Synthesis**: Combine expert reports, weighting by confidence scores
6. **Knowledge Engine Query**: Apply Indian guidelines (Diamond Approach, risk scoring) against synthesized findings
7. **Cross-Check**: Validate model outputs against deterministic guideline rules
8. **Output Generation**: Produce structured JSON output + natural language summaries for each user type

### 2.4 Model Selection Rationale

| Model | Role | Why This Model |
|-------|------|----------------|
| **Qwen2.5-VL-7B-Instruct** | Orchestrator | Strong reasoning for Indian guideline integration; native tool calling; natively supported in llama.cpp (no hacking); 7B is edge-deployable (~4.5GB Q4_K_M GGUF) |
| **Qwen2.5-VL-3B-Instruct** | ECG + Echo experts | MARCUS proved 3B is sufficient for modality-specific tasks; ~2GB Q4_K_M GGUF; native vision-language support |
| **Qwen2.5-3B-Instruct** | Clinical expert | Text-only (no vision needed for lab/vitals); same 3B size for consistency; fast inference |

**Why not keep MedGemma-27B?**
- 16.6GB GGUF is impractical for Indian clinic laptops (most have 8-16GB RAM)
- Requires fragile Gemma3->Gemma2 config hacking for llama.cpp (`modal_gguf_convert.py` lines 78-96)
- No native tool calling for agentic orchestration
- MARCUS proved specialized 3B experts outperform generalist 175B+ models
- MedGemma's medical pre-training advantage is preserved via the deterministic Clinical Knowledge Engine

### 2.5 New Files

```
src/orchestrator/
  orchestrator.py        -- Main agentic loop with tool-calling
  prompts.py             -- System prompts for orchestrator and expert routing
  tool_registry.py       -- Registry of knowledge engine tools callable by orchestrator

src/experts/
  base_expert.py         -- Abstract base class (load model, inference, structured output)
  ecg_expert.py          -- ECG modality expert
  echo_expert.py         -- Echo modality expert
  clinical_expert.py     -- Clinical data expert (text-only)
```

---

## 3. Clinical Knowledge Engine

### 3.1 Design Principle

For safety-critical clinical scoring, **deterministic Python code is more reliable than LLM inference**. The LLM handles natural language understanding (extracting symptoms, interpreting reports); the Knowledge Engine handles clinical logic (computing scores, selecting drugs, triggering referrals). This hybrid approach is a deliberate design choice not present in MARCUS (which is pure LLM) -- it leverages the Indian Consensus guidelines as both a clinical tool and a safety net.

### 3.2 Encoded Guidelines (from IJAM 2023 Consensus Statement)

#### 3.2.1 Chest Pain Scoring System (Table 1)

```
Input: precipitating_factor, location, type, duration
Output: score (0-4), probability_category

Scoring:
  precipitating_factor:
    "exertion_relieved_by_rest": 3
    "emotional_cold_meal": 1
    "unpredictable": 0
    "breathing": -1

  location:
    "retrosternal_neck_shoulder_jaw_arm_epigastric": 1
    "right_side_submammary_localized": 0

  type:
    "constricting_cramping_heavy_tight_burning_dull": 1
    "stabbing_sharp": 0
    "reproducible_by_palpation": -1

  duration:
    "less_than_15_min": 1
    "few_seconds": 0
    "more_than_15_min": -1

  Interpretation:
    score >= 3: HIGH probability angina -> expedited cardiology referral
    score 1-2: INTERMEDIATE -> further workup (stress echo/CCTA)
    score <= 0: LOW -> consider non-cardiac causes
```

#### 3.2.2 Comorbidity Risk Checklist (Table 2)

```
Factors (binary):
  - diabetes_mellitus: bool
  - cholesterol_gt_250: bool  (>6.47 mmol/l or >250 mg/dl)
  - current_smoker: bool
  - family_history_cad_lt_60: bool  (first-degree relative <60 years)
  - hypertension: bool
  - past_ihd: bool  (if True -> direct cardiology referral)

Risk score = count(True factors)
  >= 3: HIGH risk -> aggressive management
  1-2: MODERATE risk -> lifestyle + pharmacotherapy
  0: LOW risk -> primary prevention
```

#### 3.2.3 Diamond Approach Drug Selection (Table 3)

```
Input: patient comorbidity profile
Output: preferred_drugs, acceptable_drugs, contraindicated_drugs

Comorbidity Profiles:
  HIGH_HR (>=70 bpm):
    preferred: [BB, VER/DILT, IVAB]
    co_administered: [TRIM, RAN]
    contraindicated: [DHP, NITR, NIC]

  BRADYCARDIA:
    preferred: [DHP, NIC, NITR, TRIM, RAN]
    contraindicated: [BB, VER/DILT, IVAB]

  HYPERTENSION:
    preferred: [BB, DHP, VER/DILT, NITR, NIC]
    co_administered: [TRIM, RAN, IVAB]

  HYPOTENSION:
    preferred: [TRIM, RAN, IVAB]
    contraindicated: [BB, VER, DILT, DHP, NITR, NIC]

  LV_DYSFUNCTION:
    preferred: [BB]
    co_administered: [TRIM, IVAB, RAN, NITR]
    contraindicated: [DHP, VER, DILT, NIC]

  HEART_FAILURE:
    preferred: [BB, IVAB]
    co_administered: [TRIM, NITR, RAN]
    contraindicated: [DHP, VER, DILT, NIC]

  ATRIAL_FIBRILLATION:
    preferred: [BB, VER/DILT]
    co_administered: [TRIM, RAN]
    contraindicated: [DHP, NITR, NIC, IVAB]

Drug Legend:
  BB = beta-blockers
  DHP = dihydropyridine calcium-channel blockers
  DILT = diltiazem
  VER = verapamil
  IVAB = ivabradine
  NIC = nicorandil
  NITR = nitrates
  RAN = ranolazine
  TRIM = trimetazidine
```

#### 3.2.4 Risk Stratification Thresholds (ESC 2019 adapted for India)

```
Annual CV mortality estimate:
  HIGH: >3%
    Triggers: EF <35%, left main stenosis >50%, proximal LAD >50%,
              2-3 vessel disease with impaired LV, proven ischemia >10% LV
    Action: Early revascularization + OMT

  INTERMEDIATE: 1-3%
    Triggers: EF 35-50%, 2-vessel disease, intermediate DTS
    Action: Aggressive OMT, consider CCTA, stress imaging

  LOW: <1%
    Triggers: EF >50%, single vessel non-proximal, normal stress test
    Action: Lifestyle modification, primary prevention pharmacotherapy
```

#### 3.2.5 India-Specific Treatment Targets

```
Diabetic CAD patients (50-60% of Indian CAD):
  - LDL: <70 mg/dl (<1.8 mmol/l) or >=50% reduction from baseline
  - BP: <=130/80 mmHg (not <120/70)
  - HbA1c: <7%
  - Preferred agents: SGLT2i (empagliflozin, dapagliflozin), GLP-1RA (liraglutide, semaglutide)
  - ACE inhibitors for event prevention
  - High-dose statin (atorvastatin 80mg for ACS)

Non-diabetic CAD patients:
  - LDL: <70 mg/dl
  - BP: <130/80 mmHg
  - Statin even with mild LDL elevation (>100 mg/dl)

Women with CCS:
  - Treat more aggressively (traditional diagnostics less reliable)
  - Stress echo preferred over stress ECG
  - Lower threshold for invasive evaluation

South Asian phenotype adjustments:
  - BMI >=23 kg/m2 = overweight (vs >=25 Western)
  - MI onset age 5-10 years earlier than Western populations
  - Central adiposity more relevant than total BMI
  - Screen for MYBPC3 Delta-25bp in HCM/HF (4% South Asian prevalence)
  - Elevated Lp(a) as independent risk factor
```

### 3.3 New Files

```
src/knowledge/
  indian_guidelines.py        -- Master module importing all submodules
  diamond_approach.py          -- Table 3 drug selection logic
  chest_pain_scoring.py        -- Table 1 scoring system
  comorbidity_checklist.py     -- Table 2 risk factor enumeration
  risk_stratification.py       -- ESC 2019 risk categories + India adjustments
  drug_contraindications.py    -- Full drug interaction/contraindication matrix
  treatment_targets.py         -- LDL, BP, HbA1c targets by patient profile
  south_asian_phenotype.py     -- BMI thresholds, age adjustments, genetic markers
  schemas.py                   -- Pydantic models (PatientProfile, RiskScore, TreatmentPlan, ReferralDecision)
```

---

## 4. Multimodal Pipeline

### 4.1 Current vs Next-Gen

| Modality | Current | Next-Gen |
|----------|---------|----------|
| **ECG** | MedSigLIP zero-shot + VLM adapters on ECGBench | Dedicated 3B expert with 3-stage training on 50K+ ECG-report pairs |
| **Echo** | Not supported | Dedicated 3B expert trained on EchoNet-Dynamic + CAMUS |
| **CMR** | Not supported | Future expansion (Phase 2+) |
| **Clinical Data** | Free-text via EkaCare notes | Dedicated 3B text expert for structured lab/vitals/meds interpretation |
| **Integration** | None (independent paths) | Agentic orchestrator synthesizes all modalities |

### 4.2 ECG Pipeline (upgraded)

```
Input: 12-lead ECG image (scanned paper, digital printout, or PDF)
  |
  v
Image Preprocessing:
  - Normalize to 896x896 (MedSigLIP-448 compatible)
  - Auto-rotate detection (portrait vs landscape)
  - Contrast enhancement for scanned/photographed ECGs
  |
  v
ECG Expert (Qwen2.5-VL-3B, fine-tuned):
  - Visual encoder processes 16x16 patches
  - Cross-attention adapter integrates vision embeddings with LLM
  - Generates structured ECG report
  |
  v
Structured ECG Output:
  {
    "rate": 78,
    "rhythm": "normal_sinus",
    "pr_interval_ms": 160,
    "qrs_duration_ms": 88,
    "qt_qtc_ms": [380, 410],
    "axis": "normal",
    "st_changes": [
      {"lead": "V2-V4", "type": "elevation", "mm": 2.5}
    ],
    "t_wave": [
      {"lead": "V5-V6", "type": "inversion"}
    ],
    "pathological_q_waves": [],
    "lvh_criteria": false,
    "clinical_impression": "Anterior STEMI -- acute LAD occlusion",
    "urgency": "EMERGENCY",
    "confidence": 0.92
  }
```

### 4.3 Echo Pipeline (new)

```
Input: 2D echo video or still frames (A4C, PLAX, PSAX views)
  |
  v
Frame Extraction:
  - DICOM parsing with pydicom
  - Key-frame selection (systole + diastole landmarks)
  - Multi-view organization (A4C, PLAX, PSAX tagged by metadata)
  |
  v
Echo Expert (Qwen2.5-VL-3B, fine-tuned):
  - Temporal attention for video sequences
  - View-aware processing (different clinical significance per view)
  - Trained on EchoNet-Dynamic (10K videos) + CAMUS (500 patients)
  |
  v
Structured Echo Output:
  {
    "lvef_percent": 42,
    "lv_dimensions": {"lvidd_mm": 58, "lvids_mm": 45},
    "wall_motion": [
      {"segment": "anterior", "status": "hypokinetic"}
    ],
    "valvular": {
      "mitral": {"regurgitation": "moderate", "stenosis": "none"},
      "aortic": {"regurgitation": "trace", "stenosis": "none"}
    },
    "diastolic_function": {
      "e_a_ratio": 1.8,
      "e_prime": 6,
      "grade": "grade_2"
    },
    "pericardial_effusion": false,
    "clinical_impression": "Moderate LV systolic dysfunction with anterior WMA, moderate MR",
    "confidence": 0.78
  }
```

### 4.4 Clinical Data Pipeline (new)

```
Input: Structured patient data (vitals, labs, medications, history)
  |
  v
Schema Validation (Pydantic):
  - Vitals: HR, BP (sys/dia), SpO2, RR, Temp, Weight, Height, BMI
  - Labs: lipid panel, HbA1c, troponin, BNP/NT-proBNP, creatinine/eGFR, electrolytes
  - Meds: current medication list with doses
  - History: DM, HTN, smoking, family Hx, past events (MI, PCI, CABG)
  |
  v
Clinical Expert (Qwen2.5-3B, text-only, fine-tuned):
  - Analyzes lab trends and abnormalities
  - Identifies drug interactions
  - Applies Indian treatment targets
  |
  v
Structured Clinical Output:
  {
    "risk_factors_present": ["diabetes", "hypertension", "dyslipidemia"],
    "lab_abnormalities": [
      {"test": "LDL", "value": 142, "unit": "mg/dl", "target": 70, "status": "above_target"},
      {"test": "HbA1c", "value": 8.2, "unit": "%", "target": 7.0, "status": "above_target"}
    ],
    "drug_interactions": [],
    "treatment_gaps": [
      "No SGLT2i despite diabetes + CVD",
      "LDL 142 mg/dl -- needs high-intensity statin (atorvastatin 40-80mg)"
    ],
    "south_asian_flags": [
      "BMI 24.5 exceeds South Asian overweight threshold (>=23)",
      "Age 48 -- premature CAD risk window for South Asian males"
    ],
    "confidence": 0.95
  }
```

### 4.5 New Files

```
src/pipelines/
  ecg_pipeline.py              -- ECG image preprocessing + expert inference
  echo_pipeline.py             -- Echo video frame extraction + expert inference
  clinical_pipeline.py         -- Structured clinical data processing
  multimodal_aggregator.py     -- Combines outputs for orchestrator synthesis
  image_preprocessing.py       -- Shared image normalization utilities
```

---

## 5. Data Strategy

### 5.1 Current Data Audit

| Source | Records | Quality | Issue |
|--------|---------|---------|-------|
| Phase 1 (`eyone/cardiology_dataset`) | 1,000 | Medium | Generic cardiology, not Indian |
| Phase 1 (handwritten Indian) | 3 | High | Critically small |
| EkaCare clinical notes | 156 | Low-Medium | Many are non-cardiology (appendicitis, paracetamol Rx) |
| Synthetic phenotype shifts | 2 | High | Critically small -- only 2 produced |
| Mock ECG metadata | 8 | None | Placeholder data, not real |
| VLM (ECGBench ptb-test-report) | ~5K | High | Good quality but no Indian context |

**Total usable Indian-specific cardiology data: ~30 records** (out of 166 labeled V2 records, only ~20-30 EkaCare records are actually cardiology-related).

### 5.2 Data Acquisition Tiers

#### Tier 1: Public Datasets (0-3 months)

| Dataset | Size | Access | Use Case |
|---------|------|--------|----------|
| **MIMIC-IV-ECG** | 800K+ ECGs | PhysioNet (CITI training required) | ECG expert training; waveform-to-image conversion via `wfdb.plot_wfdb()` |
| **PTB-XL** | 21,837 ECGs | Open access | ECG expert training; 12-lead with cardiologist annotations |
| **EchoNet-Dynamic** | 10,030 echo videos | Stanford license | Echo expert training; A4C view with LVEF labels |
| **CAMUS** | 500 patients | Open access | Echo expert training; expert contour annotations |
| **EkaCare (full splits)** | 500+ | HF Hub (gated) | Clinical expert training; filter for cardiology-relevant records |
| **ECGBench** | ~50K | HF Hub | ECG expert training (already used for VLM) |

**Target after Tier 1: ~80K ECG + ~10K echo + ~500 clinical records**

#### Tier 2: Indian-Specific Partnerships (3-6 months)

| Source | Data Type | Access Path |
|--------|-----------|-------------|
| **IIIT-Hyderabad iHub ECG** | 12-lead ECGs from Indian patients | Formal data access request (replace mock scraping) |
| **ScienceOpen South Asian ECG** | 1,937 ECGs from South Asian patients | Research access |
| **ICMR Data Portal** | National cardiac registry data | Institutional collaboration |
| **Hospital Partnerships** (AIIMS, PGIMER, CMC Vellore, Manipal) | De-identified ECG/echo/clinical data | IRB-approved data sharing agreements |

**Target after Tier 2: +5K Indian-specific ECG/echo/clinical records**

#### Tier 3: Synthetic Augmentation (continuous)

1. **Upgraded Phenotype Shifter**: Scale from 2 to 5,000 synthetic vignettes
   - Structured Chain-of-Thought: identify clinical entities -> systematically shift each
   - Cover all Diamond Approach comorbidity combinations (7 profiles x 50+ base vignettes)
   - Use Claude/Gemini for generation, cardiologist validation for a random 10% sample

2. **ECG Augmentation**: Image-level transforms on existing ECGs
   - Gaussian noise injection (simulating poor scan quality)
   - Baseline wander simulation
   - Gain variation
   - Rotation/skew (simulating photographed ECGs)
   - Target: 5x multiplication of visual training set

3. **Clinical Vignette Generator**: Generate Indian clinical scenarios from seed conditions
   - Input: diagnosis + comorbidity profile + demographic seed
   - Output: complete clinical vignette with vitals, labs, meds, history
   - Cover: ACS, chronic stable angina, HCM, DCM, AF, valvular disease, HF (HFrEF/HFpEF), pulmonary HTN

### 5.3 Data Quality Control Pipeline

Every record must pass these gates before inclusion in the v3 training dataset:

1. **Cardiology Relevance Filter**: NLI-based classifier (zero-shot with `facebook/bart-large-mnli`) confirming the record is cardiology-related. Reject non-cardiac records currently polluting the V2 dataset
2. **Completeness Check**: Both instruction and output are non-empty and >50 tokens
3. **Indian Context Verification**: At least one South Asian-specific element present (age/BMI adjustment, genetic marker, Indian drug name, Indian guideline reference)
4. **Deduplication**: MinHash-based near-duplicate detection across all sources
5. **Schema Validation**: All records conform to the v3 Pydantic schema

### 5.4 New Files

```
src/data/
  mimic_ecg_pipeline.py          -- MIMIC-IV-ECG download + waveform-to-image conversion
  ptbxl_pipeline.py              -- PTB-XL ingestion and image format conversion
  echonet_pipeline.py            -- EchoNet-Dynamic video frame extraction
  camus_pipeline.py              -- CAMUS dataset ingestion
  eka_cardio_filter.py           -- Filter EkaCare for cardiology-relevant records
  synthetic_vignette_generator.py -- Structured synthetic clinical case generation
  phenotype_shifter_v2.py        -- Upgraded phenotype shifting with CoT
  ecg_augmentation.py            -- Image-level ECG augmentations
  quality_gate.py                -- Relevance filter, completeness, dedup, schema validation
  compile_v3_dataset.py          -- Aggregator for next-gen training dataset
  schemas.py                     -- Pydantic schemas for all data formats
```

---

## 6. Training Pipeline

### 6.1 Current vs Next-Gen

| Aspect | Current | Next-Gen |
|--------|---------|----------|
| **Stages** | 2 (SFT only) | 3 (CPT -> SFT -> GRPO) |
| **Models** | 1 (27B MedGemma) | 4 (7B orchestrator + 3x 3B experts) |
| **RL** | None | GRPO with binary correctness reward |
| **Data** | 166 records Phase 2 | 10K+ records v3 dataset |
| **Compute** | 1x A100-80GB on Modal | 4x A100-80GB runs on Modal |
| **Max seq length** | 1024-2048 | 4096 |

### 6.2 Three-Stage Training Protocol

#### Stage 1: Continued Pre-Training (CPT)

**Goal**: Inject Indian medical knowledge into base models before SFT.

```
Model: Qwen2.5-VL-7B-Instruct (orchestrator) + Qwen2.5-VL-3B-Instruct (experts)
Data: Indian cardiology textbooks (digitized), IJAM Consensus guidelines, ICMR
      publications, Indian Journal of Cardiology articles, Indian Heart Journal
Method: Standard next-token prediction on unstructured text
Epochs: 1-2 over ~100M tokens of Indian cardiology text
LR: 2e-5 (low, to avoid catastrophic forgetting)
Quantization: QLoRA 4-bit NF4 (same as current -- fits on A100-80GB)
LoRA config: r=16, alpha=32, dropout=0.05
Target modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

#### Stage 2: Supervised Fine-Tuning (SFT)

**Goal**: Align models for instruction-following on clinical tasks.

```
Orchestrator SFT:
  Data: Multi-expert synthesis examples (orchestrator receives expert reports +
        clinical data, produces unified assessment)
  Target: 5K+ synthesis examples
  LR: 1e-4
  Epochs: 3-5
  Max seq length: 4096
  Structured output training: model must produce JSON-parseable output
  Tools: Knowledge engine functions registered as callable tools

ECG Expert SFT:
  Data: ECGBench + PTB-XL + MIMIC-IV-ECG images (50K+ image-report pairs)
  LR: 1e-4
  Epochs: 3
  Max seq length: 2048
  Batch size: 2 (with grad accum 8)

Echo Expert SFT:
  Data: EchoNet-Dynamic + CAMUS frames (10K+ video-report pairs)
  LR: 1e-4
  Epochs: 5 (fewer samples, more epochs)
  Max seq length: 2048
  Batch size: 1 (video processing is memory-intensive)

Clinical Expert SFT:
  Data: V3 clinical dataset (EkaCare filtered + synthetic vignettes)
  LR: 1e-4
  Epochs: 3
  Max seq length: 4096
```

#### Stage 3: Group Relative Policy Optimization (GRPO)

**Goal**: Dramatically improve accuracy through RL with binary correctness reward. This is the critical missing stage that MARCUS proved essential.

```
Implementation: TRL GRPOTrainer (or verl framework for larger scale)
Reward: Binary correctness (1.0 if MCQ answer matches ground truth, 0.0 otherwise)
Group size: n=4 candidate responses per prompt (same as MARCUS)
KL coefficient: 0.01 (regularization against reference model)

MCQ Datasets:
  - CardioQA-India (custom): 5K+ questions from Indian cardiology boards, IJAM consensus
  - MedQA-4options (cardiology subset): ~2K questions
  - ECG-MCQ (adapted from MARCUS template): 10K+ visual MCQ
  - Echo-MCQ (adapted from MARCUS template): 5K+ visual MCQ

Training per expert:
  LR: 1e-6 (very low, as in MARCUS)
  Epochs: 10-15
  Batch size: 32 (train), 16 (validation)
  Reference model: frozen copy of SFT checkpoint

Orchestrator GRPO:
  Data: Multimodal case resolution MCQs (given expert reports + patient data,
        select correct diagnosis/management)
  LR: 1e-6
  Epochs: 10
```

### 6.3 Compute Requirements

| Training Run | GPU | Duration | Modal Cost (est.) |
|--------------|-----|----------|-------------------|
| CPT (orchestrator + experts) | 1x A100-80GB x 4 runs | ~12h each | ~$100 |
| SFT (orchestrator) | 1x A100-80GB | ~24h | ~$50 |
| SFT (ECG expert) | 1x A100-80GB | ~24h | ~$50 |
| SFT (Echo expert) | 1x A100-80GB | ~36h | ~$75 |
| SFT (Clinical expert) | 1x A100-80GB | ~12h | ~$25 |
| GRPO (all models) | 1x A100-80GB x 4 runs | ~48h each | ~$400 |
| **Total** | | | **~$700-900** |

### 6.4 New Files

```
src/training/
  stage1_cpt.py                    -- Continued pre-training on Indian cardiology corpus
  stage2_sft_orchestrator.py       -- SFT for orchestrator with tool-calling training
  stage2_sft_ecg_expert.py         -- SFT for ECG expert
  stage2_sft_echo_expert.py        -- SFT for Echo expert
  stage2_sft_clinical_expert.py    -- SFT for Clinical expert
  stage3_grpo.py                   -- GRPO training with binary reward
  modal_train_v3.py                -- Modal.com deployment wrapper for all stages
  data_collators.py                -- Custom collators (multimodal, tool-calling, structured output)
  configs/
    orchestrator_cpt.yaml
    orchestrator_sft.yaml
    ecg_expert_sft.yaml
    echo_expert_sft.yaml
    clinical_expert_sft.yaml
    grpo_ecg.yaml
    grpo_echo.yaml
    grpo_clinical.yaml
    grpo_orchestrator.yaml
```

---

## 7. User Interfaces

### 7.1 Doctor Dashboard (Cardiologists, GPs, Diabetologists)

**Framework**: Gradio Blocks (consistent with existing codebase)

**Layout**:
```
+------------------------------------------------------------------+
|  CARDIO-SAHAYAK: Clinical Decision Support                       |
+------------------------------------------------------------------+
|                                                                    |
|  [Patient Info Panel]          [Risk Score Card]                   |
|  Age: ___  Gender: ___        +---------------------------+       |
|  BMI: ___  Waist: ___         | RISK: HIGH (>3% annual)   |       |
|  DM: Y/N   HTN: Y/N          | Chest Pain Score: 3/4      |       |
|  Smoking: Y/N                 | Comorbidity Score: 4/6     |       |
|  Family Hx: ___               | LVEF: 38% (if available)   |       |
|  Past IHD: ___                +---------------------------+       |
|                                                                    |
|  [Upload ECG Image]  [Upload Echo]  [Enter Labs]                  |
|                                                                    |
+------------------------------------------------------------------+
|  MODALITY INTERPRETATIONS                                         |
|  +------------------+ +------------------+ +------------------+   |
|  | ECG Report       | | Echo Report      | | Clinical Review  |   |
|  | Rate: 78 bpm     | | LVEF: 38%        | | LDL: 142 (>70)  |   |
|  | Rhythm: NSR      | | Ant. hypokinesis | | HbA1c: 8.2 (>7) |   |
|  | ST elev V2-V4    | | Mod MR           | | No SGLT2i        |   |
|  | [Confidence: 92%]| | [Confidence: 78%]| | [Confidence: 95%]|   |
|  +------------------+ +------------------+ +------------------+   |
+------------------------------------------------------------------+
|  TREATMENT RECOMMENDATION (Diamond Approach)                      |
|  Profile: HTN + LV Dysfunction + DM                               |
|  Preferred: BB (metoprolol/bisoprolol), ACEi (ramipril)           |
|  Add: SGLT2i (empagliflozin/dapagliflozin)                       |
|  Add: High-intensity statin (atorvastatin 40-80mg)                |
|  Contraindicated: verapamil, diltiazem, nifedipine                |
|  Target: LDL <70 | BP <=130/80 | HbA1c <7%                       |
+------------------------------------------------------------------+
|  REFERRAL DECISION                                                |
|  [!] URGENT: Anterior STEMI detected on ECG                      |
|  [!] Recommend immediate cardiac catheterization                  |
|  Evidence: ST elevation V2-V4 + troponin elevation + ant. WMA     |
+------------------------------------------------------------------+
|  [Evidence Sources]  [Download PDF Report]  [Audit Trail]         |
+------------------------------------------------------------------+
```

**Key features**:
- Real-time risk score computation (deterministic, via Knowledge Engine)
- Side-by-side modality interpretations with individual confidence badges
- Diamond Approach drug selection with comorbidity-aware contraindication alerts
- Referral urgency classification (Emergency / Urgent / Routine / Manage at PHC)
- PDF report generation for medical records
- Mirage detection indicator (green/yellow/red) on each modality panel

### 7.2 Patient Portal (Patients, Family Caregivers)

**Framework**: Gradio Blocks with simplified layout

**Language**: Hindi/English toggle (extensible to Tamil, Telugu, Bengali, Marathi)

**Features**:
- **Symptom Input**: Natural language description ("mujhe seene mein dard ho raha hai" / "I have chest pain")
- **Report Upload**: Photograph/scan of ECG printout or lab reports
- **Output** (plain language):
  - Condition explanation in simple terms
  - What your risk level means (traffic light: green/yellow/red)
  - Medication purpose explanation ("This medicine helps your heart pump better")
  - Lifestyle guidance (diet: reduce salt/ghee, exercise: 30 min walk daily, tobacco cessation)
  - Emergency warning signs ("Go to hospital immediately if: crushing chest pain >15 min, breathlessness at rest, fainting")
  - Medication adherence reminders
- **Disclaimer**: "This is not a substitute for your doctor's advice. Always consult your physician."

### 7.3 CHW Screener (ASHAs, ANMs, Health Workers)

**Framework**: Gradio with mobile-optimized layout

**Design**: Checklist-based, minimal text input, color-coded outputs

```
SCREENING QUESTIONNAIRE:
  [ ] Age > 45 (male) or > 55 (female)?
  [ ] Known diabetic?
  [ ] Known hypertensive?
  [ ] Current tobacco user (bidi/cigarette/gutka)?
  [ ] Family member had heart attack before age 60?
  [ ] Chest pain on exertion?
  [ ] Chest pain relieved by rest?
  [ ] Breathlessness on mild exertion?

  BP Reading: ___/___ mmHg
  Random Blood Sugar: ___ mg/dl (if available)

OUTPUT:
  [RED] HIGH RISK -- Refer to District Hospital immediately
  [YELLOW] MODERATE RISK -- Refer to PHC within 1 week
  [GREEN] LOW RISK -- Lifestyle counseling, rescreen in 6 months

  Action Items:
  1. _________________________
  2. _________________________
  3. _________________________
```

**Offline mode**: Runs entirely on GGUF backend, no internet required

### 7.4 New Files

```
src/ui/
  doctor_dashboard.py          -- Full clinical decision support interface
  patient_portal.py            -- Simplified patient-facing interface
  chw_screener.py              -- Mobile-optimized CHW screening tool
  components/
    risk_card.py               -- Reusable risk score visualization
    diamond_panel.py           -- Diamond Approach drug selection display
    referral_alert.py          -- Referral urgency alert component
    modality_report.py         -- Per-modality interpretation card with confidence
    ecg_viewer.py              -- ECG image display with annotation overlay
  i18n/
    hindi.json                 -- Hindi translations
    english.json               -- English strings
    common.json                -- Shared medical term translations
```

---

## 8. Structured Output Engine

### 8.1 Problem

The current system outputs only free text. A doctor receives a paragraph and must manually extract clinical decisions. This is unacceptable for clinical decision support -- structured, machine-parseable output enables:
- Automated risk alerting (trigger notifications for high-risk cases)
- EMR integration (structured data can flow into hospital systems)
- Audit trail (every field is traceable to evidence)
- Quality metrics (measure guideline compliance algorithmically)

### 8.2 Output Schema (Pydantic)

```python
class CardioSahayakOutput(BaseModel):
    patient_id: str
    timestamp: datetime
    session_id: str

    # Demographics
    age: int
    gender: Literal["male", "female", "other"]
    bmi: Optional[float]
    south_asian_bmi_category: Optional[Literal["normal", "overweight", "obese"]]

    # Risk Assessment
    risk_category: Literal["HIGH", "INTERMEDIATE", "LOW"]
    annual_cv_mortality_estimate: Optional[float]
    chest_pain_score: Optional[int]  # 0-4
    comorbidity_score: int  # 0-6
    comorbidities_present: list[str]

    # Modality Interpretations
    ecg_report: Optional[ECGReport]
    echo_report: Optional[EchoReport]
    clinical_report: Optional[ClinicalReport]

    # Clinical Decision Support
    treatment_plan: TreatmentPlan
    referral_decision: ReferralDecision
    medication_recommendations: list[MedicationRecommendation]
    contraindicated_medications: list[str]
    treatment_gaps: list[str]

    # Targets
    ldl_target_mg_dl: float
    bp_target: str
    hba1c_target: Optional[float]

    # Safety
    confidence_scores: dict[str, float]
    mirage_flags: list[str]
    evidence_sources: list[str]

    # Natural Language Summaries
    doctor_summary: str
    patient_summary_en: str
    patient_summary_hi: Optional[str]
    chw_action_items: list[str]

class ReferralDecision(BaseModel):
    urgency: Literal["EMERGENCY", "URGENT", "ROUTINE", "MANAGE_AT_PHC"]
    destination: Literal["CATHLAB", "CARDIOLOGY_OPD", "DISTRICT_HOSPITAL", "CHC", "PHC"]
    reason: str
    evidence: list[str]

class MedicationRecommendation(BaseModel):
    drug_class: str
    specific_drug: str
    dose: str
    rationale: str
    guideline_source: str
    nlem_available: bool  # Available in Indian National List of Essential Medicines
```

### 8.3 Implementation

Use **constrained decoding** (via `outlines` library or Qwen2.5's native JSON mode) to force the orchestrator to produce valid JSON matching the Pydantic schema. The Knowledge Engine fills deterministic fields (risk scores, drug contraindications); the LLM fills interpretive fields (summaries, clinical impressions).

### 8.4 New Files

```
src/output/
  schemas.py                   -- All Pydantic models (CardioSahayakOutput, ECGReport, etc.)
  constrained_decoder.py       -- Constrained decoding using outlines or native JSON mode
  report_generator.py          -- Combines model output + knowledge engine into final report
  pdf_renderer.py              -- Generate printable PDF clinical reports
  emr_formatter.py             -- Format output for Indian EMR systems (ABDM FHIR compatibility)
```

---

## 9. Safety and Mirage Detection

### 9.1 The Mirage Problem

MARCUS identified "mirage reasoning" as a critical safety issue: frontier models fabricate detailed clinical descriptions for images never provided. The current Cardio-Sahayak has zero mirage detection. If the VLM hallucinates ST elevation that isn't present in the ECG, there is no mechanism to catch it.

### 9.2 Three-Layer Safety System

#### Layer 1: Counterfactual Image-Absent Probe (from MARCUS)

For every image-based expert report:
1. Run inference with the image + clinical question (image-present)
2. Run identical inference with only the clinical question, no image (image-absent)
3. Compute similarity: `sim = jaccard(tokens_present, tokens_absent)`
4. If `sim > 0.85`: flag as MIRAGE -- the model is answering from language priors, not visual evidence
5. Down-weight flagged modality in orchestrator synthesis

MARCUS achieved **0% mirage rate** with this protocol.

#### Layer 2: 3-Rephrasing Consistency Check (from MARCUS)

1. Orchestrator generates 3 semantically equivalent but syntactically distinct rephrasings of the clinical question
2. All 3 are routed to the same expert model
3. Inter-rephrase consistency: `consistency = mean_pairwise_jaccard(responses)`
4. Low consistency (<0.6) = model is unstable on this case -> flag for human review

#### Layer 3: Deterministic Cross-Check (novel, India-specific)

Cross-check LLM outputs against Knowledge Engine rules:
- If model says "LVEF 45%" but recommends a drug contraindicated for LV dysfunction -> CONTRADICTION flag
- If model recommends a drug not in the Indian NLEM -> AVAILABILITY flag
- If risk score implies HIGH risk but referral says "manage at PHC" -> INCONSISTENCY flag
- If model recommends SGLT2i for a patient with eGFR <20 -> CONTRAINDICATION flag

This layer is unique to Cardio-Sahayak -- it leverages the deterministic Indian guidelines as a safety net that pure LLM systems (including MARCUS) lack.

### 9.3 Audit Trail

Every inference is logged with:
- Input data (anonymized patient ID, modality types, data checksums)
- All expert reports (raw text)
- Mirage detection results per modality
- Confidence scores per section
- Knowledge Engine queries and results
- Final output (full JSON)
- Timestamp, model versions, session ID

Required for CDSCO SaMD compliance and post-market surveillance.

### 9.4 New Files

```
src/safety/
  mirage_detector.py           -- Counterfactual probe + 3-rephrasing consistency
  cross_checker.py             -- Knowledge engine cross-validation rules
  confidence_calibrator.py     -- Temperature scaling for calibrated confidence
  audit_logger.py              -- Structured inference logging (JSONL format)
```

---

## 10. Edge Deployment

### 10.1 Current vs Next-Gen

| Aspect | Current | Next-Gen |
|--------|---------|----------|
| **Model** | MedGemma-27B | Qwen2.5-VL-7B (orchestrator) + Qwen2.5-VL-3B (experts) |
| **GGUF size** | 16.6 GB (single model) | ~4.5GB orchestrator + ~2GB per expert = 10.5GB total |
| **llama.cpp support** | Requires Gemma3->Gemma2 architectural patching | Native Qwen2.5 support (no hacking) |
| **Minimum RAM** | ~20GB | ~8GB (load orchestrator + 1 expert at a time) |
| **CPU inference** | Very slow (27B) | Usable (7B: ~15-20 tok/s on modern laptop) |

### 10.2 Deployment Tiers

**Tier 1: Full Edge (Rural PHC/Sub-Centre)**
- All models in GGUF on local machine
- No internet required
- Dynamic model loading: orchestrator always loaded; experts loaded on-demand
  (only load ECG expert when ECG is uploaded, release after inference)
- Target hardware: Intel i5/i7 laptop with 16GB RAM (typical government-procured)
- Alternative: Raspberry Pi 5 with 8GB for CHW Screener mode only (text-only, no vision)

**Tier 2: Hybrid (CHC/District Hospital with intermittent connectivity)**
- Orchestrator + Clinical expert local (always available)
- ECG/Echo experts via cloud API when connected; cached GGUF fallback when offline
- Automatic sync of patient reports when connectivity restored

**Tier 3: Cloud (Urban/Tertiary Hospital)**
- Full system on Modal.com or AWS with A100 inference
- Maximum accuracy (FP16 models, no quantization)
- Lowest latency for multi-expert queries
- EMR integration via ABDM (Ayushman Bharat Digital Mission) FHIR APIs

### 10.3 GGUF Conversion Pipeline (simplified)

```python
# No more Gemma3->Gemma2 architectural hacking!
# Qwen2.5 is natively supported in llama.cpp

for model_name in ["orchestrator", "ecg_expert", "echo_expert", "clinical_expert"]:
    # 1. Merge QLoRA adapters into base model
    merged = PeftModel.from_pretrained(base, adapters).merge_and_unload()

    # 2. Save merged model
    merged.save_pretrained(f"merged/{model_name}")

    # 3. Convert to GGUF (native support, no patching)
    # llama.cpp convert_hf_to_gguf.py handles Qwen2.5 natively
    subprocess.run(["python", "convert_hf_to_gguf.py", f"merged/{model_name}"])

    # 4. Quantize to Q4_K_M
    subprocess.run(["llama-quantize", f"{model_name}.gguf",
                     f"{model_name}-Q4_K_M.gguf", "Q4_K_M"])

    # 5. Upload to HuggingFace Hub
    upload_to_hub(f"{model_name}-Q4_K_M.gguf", repo="tp53/cardio-sahayak-v3-gguf")
```

### 10.4 New Files

```
src/edge/
  gguf_converter.py            -- Clean GGUF conversion for Qwen2.5 models
  edge_runtime.py              -- Local inference engine using llama-cpp-python
  model_manager.py             -- Dynamic model loading/unloading by available RAM
  offline_cache.py             -- Cache knowledge engine data + common responses
  connectivity_manager.py      -- Detect online/offline state, manage API fallback
  modal_deploy_v3.py           -- Modal.com deployment for cloud tier
```

---

## 11. Evaluation and Validation

### 11.1 Automated Benchmarks

| Benchmark | Type | Source | Target | MARCUS Baseline |
|-----------|------|--------|--------|-----------------|
| **MedQA-4options** | MCQ | Existing | >60% | N/A (general) |
| **PubMedQA** | MCQ | Existing | >70% | N/A (general) |
| **CardioQA-India** (custom) | MCQ | Indian cardiology boards + IJAM consensus | >75% | N/A (new) |
| **ECG-MCQ** | Visual MCQ | Adapted from MARCUS template | >80% | 87-91% |
| **Echo-MCQ** | Visual MCQ | Adapted from MARCUS template | >60% | 67-86% |
| **Mirage Rate** | Safety | Counterfactual probing | 0% | 0% |
| **Guideline Compliance** | Accuracy | Automated cross-check | >90% | N/A (new) |
| **Structured Output Validity** | Format | JSON schema validation | 100% | N/A |

### 11.2 CardioQA-India Benchmark (custom)

A new benchmark specifically for Indian cardiovascular medicine:

**Content**: 1,000+ MCQ questions covering:
- Diamond Approach drug selection (all 7 comorbidity profiles)
- Chest Pain Scoring interpretation
- Risk stratification (ESC 2019 categories with Indian adjustments)
- South Asian phenotype recognition (MYBPC3, lower BMI thresholds, early-onset MI)
- Diabetic-cardiac management (SGLT2i/GLP-1RA indications, LDL targets)
- Women-specific cardiac care (atypical presentation, aggressive treatment)
- Referral triggers (when to escalate from PHC to cardiology)
- Drug availability in India (NLEM compliance)
- ECG interpretation with Indian clinical context
- Echo interpretation with treatment decisions

**Construction**:
1. Extract 200 questions from Indian cardiology board exams (NEET-SS, DNB)
2. Generate 500 questions from IJAM Consensus paper tables using LLM templating
3. Adapt 300 questions from MedQA/PubMedQA with Indian phenotype context
4. Cardiologist validation of all questions (2 independent reviewers per question)

### 11.3 Human Evaluation Protocol

**Cardiologist Blinded Review**:
- Recruit 50+ Indian cardiologists (across AIIMS, PGIMER, CMC Vellore, private hospitals)
- 100 de-identified cases with multimodal data (ECG + labs + clinical history)
- Evaluate Cardio-Sahayak outputs using AMIE 10-domain rubric:
  1. History gathering completeness
  2. Diagnostic accuracy
  3. Clinical reasoning quality
  4. Management plan appropriateness
  5. Drug selection correctness
  6. Risk stratification accuracy
  7. Referral appropriateness
  8. Patient communication quality
  9. Guideline adherence (Indian Consensus)
  10. South Asian phenotype awareness
- Compare: (a) Cardio-Sahayak alone, (b) GP alone, (c) GP + Cardio-Sahayak
- Primary endpoint: error reduction and omission reduction (as in AMIE paper)

### 11.4 India-Specific Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **South Asian Phenotype Sensitivity** | On cases with Indian risk factors, does the model correctly adjust? | >90% |
| **NLEM Drug Compliance** | Are recommended drugs available in Indian NLEM? | >95% |
| **Hindi Output Accuracy** | Cardiologist validation of Hindi patient summaries | >85% clinically accurate |
| **Edge Latency** | Time-to-first-token on GGUF (Q4_K_M) on i7 laptop | <5 seconds |
| **Edge Accuracy Degradation** | MCQ accuracy drop from FP16 to Q4_K_M | <5% degradation |
| **Referral Appropriateness** | Correct urgency level and destination | >85% |

### 11.5 New Files

```
src/evaluation/
  cardioqa_india.py                -- Custom Indian cardiology benchmark loader
  cardioqa_india_generator.py      -- MCQ generation from guidelines and templates
  mirage_benchmark.py              -- Systematic mirage rate evaluation
  guideline_compliance.py          -- Automated compliance checker against Knowledge Engine
  structured_output_validator.py   -- JSON schema validation + internal consistency
  edge_benchmark.py                -- Latency and accuracy on GGUF models
  human_eval_protocol.py           -- Generate evaluation packets for cardiologists
  run_all_evals.py                 -- Master evaluation orchestrator

data/benchmarks/
  cardioqa_india_v1.jsonl          -- Custom benchmark dataset
```

---

## 12. Regulatory and Ethics

### 12.1 Indian Medical Device Regulations

Cardio-Sahayak, if deployed clinically, falls under **CDSCO (Central Drugs Standard Control Organisation)** regulations for **Software as a Medical Device (SaMD)**:

| Classification | Criteria | Cardio-Sahayak Status |
|---------------|----------|----------------------|
| **Class A** (low risk) | Informational only, no clinical decisions | Patient Portal (informational mode) |
| **Class B** (medium risk) | Clinical decision support with physician oversight | Doctor Dashboard (recommended classification) |
| **Class C** (higher risk) | Autonomous diagnosis | NOT targeted -- always human-in-the-loop |

**Regulatory Requirements for Class B SaMD**:
- Clinical investigation per CDSCO Schedule 4 (multi-site, minimum 3 Indian hospitals)
- Quality Management System (ISO 13485)
- Risk Management (ISO 14971)
- Post-Market Surveillance with mandatory adverse event reporting
- Technical documentation for CDSCO review

### 12.2 Data Privacy

**Digital Personal Data Protection Act (DPDPA) 2023**:
- All patient data processed with explicit consent
- Data minimization: collect only what's needed for clinical assessment
- Purpose limitation: data used only for clinical decision support
- Data localization: patient data must remain within Indian borders (edge deployment is compliant by design)
- Right to erasure: patients can request deletion of their data
- Data fiduciary responsibilities for hospital deployers

**ABDM (Ayushman Bharat Digital Mission) Compliance**:
- Structured output in FHIR format for interoperability
- ABHA (Ayushman Bharat Health Account) integration for patient identification
- Health Information Exchange consent framework

### 12.3 Ethical Safeguards

1. **Always Human-in-the-Loop**: Every output is presented as a "suggestion" requiring physician confirmation, never as an autonomous decision
2. **Informed Consent**: Patient mode includes clear disclaimers and consent flows
3. **Bias Monitoring**: Track model performance across demographic subgroups (gender, age, socioeconomic status, geography)
4. **Transparency**: Structured output with evidence sources enables review of reasoning
5. **Equity**: Edge deployment ensures rural patients have access to the same AI capabilities as urban patients

### 12.4 New Files

```
src/compliance/
  audit_trail.py               -- Comprehensive inference logging (CDSCO compliant)
  consent_flow.py              -- Patient consent management
  disclaimer.py                -- Regulatory disclaimers for all interfaces
  data_privacy.py              -- DPDPA compliance utilities (anonymization, erasure)
  fhir_formatter.py            -- ABDM FHIR output formatting
```

---

## 13. Implementation Roadmap

### Phase 0: Foundation (Weeks 1-4)

| Task | Deliverable |
|------|-------------|
| Set up `src/` directory structure | All submodule directories created |
| Implement Knowledge Engine | `diamond_approach.py`, `chest_pain_scoring.py`, `comorbidity_checklist.py`, `risk_stratification.py` with unit tests |
| Build Pydantic schemas | `src/output/schemas.py`, `src/knowledge/schemas.py` with validation tests |
| Design CardioQA-India benchmark | 200+ manually curated questions from Indian cardiology boards |
| Obtain PhysioNet credentials | CITI training completed, MIMIC-IV-ECG access granted |

### Phase 1: Data Pipeline (Weeks 5-10)

| Task | Deliverable |
|------|-------------|
| Download MIMIC-IV-ECG (50K subset) | `src/data/mimic_ecg_pipeline.py` with waveform-to-image conversion |
| Download PTB-XL + EchoNet-Dynamic + CAMUS | Ingestion pipelines for all datasets |
| Filter EkaCare for cardiology | `src/data/eka_cardio_filter.py` -- reduce 156 to ~30 true cardiology records |
| Scale synthetic phenotype shifter | 5,000 synthetic Indian clinical vignettes |
| ECG image augmentation pipeline | 5x multiplication of ECG visual training set |
| Compile v3 dataset with quality gates | `cardio_sahayak_india_instruct_v3.jsonl` (10K+ records) |
| Expand CardioQA-India to 1,000+ MCQs | LLM-generated + cardiologist-validated questions |

### Phase 2: Expert Model Training (Weeks 11-18)

| Task | Deliverable |
|------|-------------|
| Stage 1 CPT on Indian cardiology text | Pre-trained base weights for all 4 models |
| Stage 2 SFT: ECG expert | ECG expert adapter on ECGBench + PTB-XL + MIMIC-IV |
| Stage 2 SFT: Echo expert | Echo expert adapter on EchoNet-Dynamic + CAMUS |
| Stage 2 SFT: Clinical expert | Clinical expert adapter on v3 text dataset |
| Stage 3 GRPO: All experts | RL-refined experts with binary correctness reward |
| Baseline evaluation | ECG-MCQ, Echo-MCQ, CardioQA-India benchmarks |

### Phase 3: Orchestrator + Integration (Weeks 19-24)

| Task | Deliverable |
|------|-------------|
| Generate orchestrator training data | 5K+ multi-expert synthesis examples |
| Stage 2+3: Orchestrator SFT + GRPO | Trained orchestrator with tool calling |
| Integrate Knowledge Engine as tools | Orchestrator calls deterministic guidelines |
| Implement mirage detection | Counterfactual probe + 3-rephrasing + cross-check |
| Implement structured output engine | Constrained decoding producing valid JSON |
| End-to-end integration test | Full pipeline: input -> experts -> orchestrator -> structured output |

### Phase 4: Interfaces + Edge (Weeks 25-30)

| Task | Deliverable |
|------|-------------|
| Build Doctor Dashboard | `src/ui/doctor_dashboard.py` with all components |
| Build Patient Portal | `src/ui/patient_portal.py` with Hindi/English |
| Build CHW Screener | `src/ui/chw_screener.py` with offline mode |
| GGUF conversion (all models) | Q4_K_M GGUF for orchestrator + all experts |
| Edge runtime testing | Latency and accuracy on target hardware profiles |
| PDF report generation | `src/output/pdf_renderer.py` |

### Phase 5: Evaluation + Validation (Weeks 31-36)

| Task | Deliverable |
|------|-------------|
| Run all automated benchmarks | MedQA, PubMedQA, CardioQA-India, ECG-MCQ, Echo-MCQ |
| Mirage rate evaluation | Systematic counterfactual probing across test set |
| Guideline compliance audit | Automated + expert review |
| Edge performance benchmarks | Latency, accuracy degradation measurements |
| Cardiologist blinded review | 50+ cardiologists, 100 cases, AMIE rubric |
| Write v6 preprint | Comprehensive paper documenting next-gen system |

### Phase 6: Pilot Deployment (Weeks 37+)

| Task | Deliverable |
|------|-------------|
| Select 3-5 pilot sites | Mix: rural PHC, urban CHC, tertiary hospital |
| Deploy edge systems | GGUF on clinic laptops with audit logging |
| Deploy cloud tier | Modal.com/AWS for tertiary hospitals |
| Collect real-world feedback | Clinician and patient feedback surveys |
| Iterate | Address feedback, retrain models on new data |
| Prepare CDSCO SaMD submission | Technical documentation, clinical investigation results |

---

## 14. Project Directory Structure (Proposed)

```
cardio-sahayak/
  # --- Existing (preserved) ---
  app.py                          # Legacy Gradio app (deprecated by doctor_dashboard)
  gradio_app.py                   # Legacy (deprecated)
  modal_train_cardio_sahayak.py   # Phase 1 training (preserved for reference)
  modal_train_cardio_sahayak_v2.py # Phase 2 training (preserved for reference)
  modal_train_vlm_cardio_sahayak.py # VLM training (preserved for reference)
  data/                           # Existing datasets (preserved)
  examples/                       # Existing ECG examples (preserved)
  out/                            # Existing preprints (preserved)
  papers/                         # Reference papers (preserved)
  docs_plans/                     # Specifications (this document added)

  # --- New: Next-Gen System ---
  src/
    __init__.py

    orchestrator/
      __init__.py
      orchestrator.py             # Main agentic loop with tool-calling
      prompts.py                  # System prompts for orchestrator and routing
      tool_registry.py            # Knowledge engine tools callable by orchestrator

    experts/
      __init__.py
      base_expert.py              # Abstract base class
      ecg_expert.py               # ECG modality expert (Qwen2.5-VL-3B)
      echo_expert.py              # Echo modality expert (Qwen2.5-VL-3B)
      clinical_expert.py          # Clinical data expert (Qwen2.5-3B)

    knowledge/
      __init__.py
      indian_guidelines.py        # Master module
      diamond_approach.py         # Table 3 drug selection
      chest_pain_scoring.py       # Table 1 scoring
      comorbidity_checklist.py    # Table 2 risk factors
      risk_stratification.py      # ESC 2019 + India adjustments
      drug_contraindications.py   # Full drug interaction matrix
      treatment_targets.py        # LDL, BP, HbA1c targets by profile
      south_asian_phenotype.py    # BMI thresholds, age adjustments, genetics
      schemas.py                  # Pydantic clinical data models

    pipelines/
      __init__.py
      ecg_pipeline.py             # ECG preprocessing + expert inference
      echo_pipeline.py            # Echo video processing + expert inference
      clinical_pipeline.py        # Structured clinical data processing
      multimodal_aggregator.py    # Combine outputs for orchestrator
      image_preprocessing.py      # Shared image utilities

    data/
      __init__.py
      mimic_ecg_pipeline.py       # MIMIC-IV-ECG download + conversion
      ptbxl_pipeline.py           # PTB-XL ingestion
      echonet_pipeline.py         # EchoNet-Dynamic frames
      camus_pipeline.py           # CAMUS dataset
      eka_cardio_filter.py        # Filter EkaCare for cardiology
      synthetic_vignette_generator.py  # Clinical case generation
      phenotype_shifter_v2.py     # Upgraded phenotype shifting
      ecg_augmentation.py         # Image-level ECG augmentations
      quality_gate.py             # Relevance filter, completeness, dedup
      compile_v3_dataset.py       # V3 dataset aggregator
      schemas.py                  # Data format schemas

    training/
      __init__.py
      stage1_cpt.py               # Continued pre-training
      stage2_sft_orchestrator.py  # Orchestrator SFT
      stage2_sft_ecg_expert.py    # ECG expert SFT
      stage2_sft_echo_expert.py   # Echo expert SFT
      stage2_sft_clinical_expert.py # Clinical expert SFT
      stage3_grpo.py              # GRPO with binary reward
      modal_train_v3.py           # Modal.com wrapper
      data_collators.py           # Custom collators
      configs/                    # YAML configs per training stage

    output/
      __init__.py
      schemas.py                  # Pydantic output models
      constrained_decoder.py      # Structured output enforcement
      report_generator.py         # Final report assembly
      pdf_renderer.py             # PDF generation
      emr_formatter.py            # ABDM FHIR formatting

    safety/
      __init__.py
      mirage_detector.py          # Counterfactual probe + consistency
      cross_checker.py            # Knowledge engine cross-validation
      confidence_calibrator.py    # Calibrated confidence scoring
      audit_logger.py             # JSONL inference logging

    evaluation/
      __init__.py
      cardioqa_india.py           # Custom benchmark loader
      cardioqa_india_generator.py # MCQ generation
      mirage_benchmark.py         # Mirage rate evaluation
      guideline_compliance.py     # Compliance checker
      structured_output_validator.py # JSON validation
      edge_benchmark.py           # Edge performance
      human_eval_protocol.py      # Evaluation packet generation
      run_all_evals.py            # Master orchestrator

    edge/
      __init__.py
      gguf_converter.py           # Clean Qwen2.5 GGUF conversion
      edge_runtime.py             # llama-cpp-python inference
      model_manager.py            # Dynamic model loading
      offline_cache.py            # Offline knowledge cache
      connectivity_manager.py     # Online/offline detection
      modal_deploy_v3.py          # Cloud deployment

    ui/
      __init__.py
      doctor_dashboard.py         # Clinical decision support
      patient_portal.py           # Patient-facing interface
      chw_screener.py             # CHW screening tool
      components/
        risk_card.py
        diamond_panel.py
        referral_alert.py
        modality_report.py
        ecg_viewer.py
      i18n/
        hindi.json
        english.json
        common.json

    compliance/
      __init__.py
      audit_trail.py              # CDSCO-compliant logging
      consent_flow.py             # Patient consent
      disclaimer.py               # Regulatory disclaimers
      data_privacy.py             # DPDPA utilities
      fhir_formatter.py           # ABDM FHIR output

  data/
    benchmarks/
      cardioqa_india_v1.jsonl     # Custom benchmark
    raw_datasets/                 # Existing
    processed_datasets/           # Existing

  tests/
    test_knowledge_engine.py      # Unit tests for all clinical scoring
    test_schemas.py               # Pydantic model validation tests
    test_mirage_detection.py      # Mirage detection unit tests
    test_cross_checker.py         # Cross-check logic tests
    test_structured_output.py     # Output schema validation tests
```

---

## 15. Key Technical Decisions and Rationale

### Decision 1: Qwen2.5 over MedGemma

| Factor | MedGemma-27B (current) | Qwen2.5-VL-7B/3B (proposed) |
|--------|----------------------|------------------------------|
| Edge GGUF size | 16.6 GB | 4.5 GB (7B) / 2 GB (3B) |
| llama.cpp support | Requires Gemma3->Gemma2 patching | Native |
| Tool calling | Not supported | Native |
| Min RAM | ~20 GB | ~8 GB |
| Medical pre-training | Yes (advantage) | No (compensated by Knowledge Engine + domain CPT) |

**MedGemma's medical knowledge advantage is preserved** through the deterministic Clinical Knowledge Engine rather than relying on model weights. Indian clinical guidelines are encoded as code (always correct) rather than learned parameters (potentially hallucinated).

### Decision 2: Expert-per-Modality Architecture

MARCUS proved that **specialized 3B experts outperform generalist 175B+ models** on cardiac interpretation. A 3B model trained exclusively on ECGs (87-91% accuracy) beats GPT-5 (35-48%) because domain-specific training on high-quality clinical data matters more than model scale. The agentic orchestration pattern handles cross-modal synthesis without requiring a single model to master everything.

### Decision 3: GRPO over DPO/RLHF

MARCUS's key finding: GRPO with simple **binary correctness reward** (correct MCQ answer = 1.0, incorrect = 0.0) is dramatically effective and requires no expensive human preference data. The TRL library already supports `GRPOTrainer`. DPO requires preference pairs (expensive to collect for medical domains); RLHF requires reward model training. GRPO is simpler, cheaper, and proven.

### Decision 4: Deterministic Knowledge Engine + LLM (Hybrid)

Pure LLM systems (including MARCUS) can hallucinate clinical scores and drug recommendations. Our hybrid approach:
- **LLM**: Natural language understanding, interpretation, explanation
- **Code**: Clinical scoring, drug selection, risk stratification, cross-checking

This provides a **safety floor** that pure LLM systems cannot guarantee. If the LLM hallucinates, the Knowledge Engine catches contradictions. This is a novel contribution not present in MARCUS or any comparable system.

### Decision 5: Three User Modes from Day One

Indian healthcare has extreme user diversity. A cardiologist at AIIMS needs structured clinical decision support; an ASHA worker in rural Bihar needs a simple color-coded checklist. Designing for all three from the start (Doctor, Patient, CHW) avoids costly retrofitting and ensures the system serves the full continuum of Indian healthcare.

---

## Appendix A: Reference Papers

1. **MARCUS**: O'Sullivan et al. "An agentic, multimodal vision-language model for cardiac diagnosis and management." arXiv 2603.22179v1, March 2026. [GitHub: AshleyLab/MARCUS]
2. **Indian Consensus**: Balaji et al. "Consensus statement on cardiovascular risk stratification and aggressive management of chronic coronary syndromes." Int J Adv Med 2023;10(5):425-432.
3. **AMIE**: "A large language model for complex cardiology care." Nature Medicine, s41591-025-04190-9.
4. **EchoJEPA**: Latent predictive foundation model for echocardiography. arXiv 2602.02603v4.
5. **ECG-JEPA**: Learning general representations of 12-lead ECG. arXiv 2410.08559v4.

## Appendix B: Abbreviations

| Abbreviation | Meaning |
|-------------|---------|
| ABDM | Ayushman Bharat Digital Mission |
| ASHA | Accredited Social Health Activist |
| ANM | Auxiliary Nurse Midwife |
| BB | Beta-blockers |
| CAD | Coronary Artery Disease |
| CCTA | Coronary Computed Tomography Angiography |
| CCS | Chronic Coronary Syndromes |
| CHC | Community Health Centre |
| CHW | Community Health Worker |
| CPT | Continued Pre-Training |
| CDSCO | Central Drugs Standard Control Organisation |
| DHP | Dihydropyridine calcium-channel blockers |
| DPDPA | Digital Personal Data Protection Act |
| DTS | Duke Treadmill Score |
| FHIR | Fast Healthcare Interoperability Resources |
| GRPO | Group Relative Policy Optimization |
| HCM | Hypertrophic Cardiomyopathy |
| IVAB | Ivabradine |
| NLEM | National List of Essential Medicines |
| NIC | Nicorandil |
| OMT | Optimal Medical Treatment |
| PHC | Primary Health Centre |
| QLoRA | Quantized Low-Rank Adaptation |
| RAN | Ranolazine |
| SaMD | Software as a Medical Device |
| SFT | Supervised Fine-Tuning |
| TRIM | Trimetazidine |
| VLM | Vision-Language Model |

## Appendix C: Key Metrics Summary

| Metric | Current System | Next-Gen Target | MARCUS Benchmark |
|--------|---------------|-----------------|-----------------|
| ECG MCQ Accuracy | Not measured | >80% | 87-91% |
| Echo MCQ Accuracy | N/A (not supported) | >60% | 67-86% |
| Multimodal Accuracy | N/A | >65% | 70% |
| Mirage Rate | Unknown (no detection) | 0% | 0% |
| Guideline Compliance | Unknown | >90% | N/A |
| Edge GGUF Size | 16.6 GB | ~10.5 GB total | N/A (cloud only) |
| Min RAM | ~20 GB | ~8 GB | N/A |
| Training Data | 166 records (Phase 2) | 10K+ records | 13.5M images |
| Indian Cardiology MCQ | N/A | >75% | N/A |
| Structured Output | None (free text only) | 100% valid JSON | N/A |
| User Modes | 1 (doctor, basic) | 3 (doctor, patient, CHW) | 1 (researcher) |
| Languages | English only | Hindi + English | English only |
| Compute Cost (training) | ~$100 | ~$700-900 | >$50K (H100 DGX) |
