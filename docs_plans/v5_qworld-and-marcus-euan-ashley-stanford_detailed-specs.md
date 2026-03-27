# Cardio-Sahayak Next-Gen v5: Qworld + MARCUS Integrated Specification

**Date:** 27 March 2026
**Status:** Design Specification (v5 -- supersedes v3)
**Authors:** TP53 + Claude Opus 4.6 Deep Research
**Key References:**
- MARCUS (Stanford/UCSF, 2603.22179v1) -- Agentic multimodal cardiac VLM
- Qworld (Harvard, 2603.23522v1) -- Question-specific evaluation criteria via recursive expansion
- Indian Consensus on CV Risk Stratification (IJAM 2023) -- Clinical guidelines
- HealthBench (OpenAI, 2025) -- Medical LLM evaluation benchmark

---

## Executive Summary

This v5 specification extends v3 with a critical addition: **Qworld-powered question-specific evaluation** that transforms how we assess Cardio-Sahayak's clinical outputs. The three pillars are now:

1. **MARCUS Architecture** -- Agentic multi-expert system (ECG + Echo + Clinical experts coordinated by orchestrator) proven to outperform GPT-5 by 34-45% on cardiac interpretation.

2. **Indian Consensus Clinical Engine** -- Deterministic encoding of IJAM 2023 guidelines (Diamond Approach, Chest Pain Scoring, risk stratification) as code, not hallucination.

3. **Qworld Evaluation Framework** -- Question-specific, recursively-expanded evaluation criteria that replace static MCQ benchmarks with multi-dimensional assessment covering safety, equity, long-term impact, guideline adherence, and patient communication -- exactly the dimensions that matter in Indian cardiovascular care.

**Why Qworld changes everything for Cardio-Sahayak:** Our v3 evaluation relied on 22 static MCQs and simple accuracy metrics. Qworld's Recursive Expansion Tree (RET) generates ~45 fine-grained binary criteria per clinical question, covering dimensions like safety/risk management, health equity, caregiver support, follow-up continuity, and cultural sensitivity that static rubrics miss entirely. On HealthBench (5,000 medical queries), Qworld achieves 89% Coverage of expert-authored criteria while generating 79% novel criteria that experts validate as insightful. This means we can build an evaluation system that is both more rigorous AND more India-specific than anything currently available for cardiac AI.

---

## Table of Contents

1. [What Changed from v3](#1-what-changed-from-v3)
2. [Qworld Integration Architecture](#2-qworld-integration-architecture)
3. [India-Specific Qworld Evaluation Dimensions](#3-india-specific-qworld-evaluation-dimensions)
4. [CardioQA-India v2: Qworld-Enhanced Benchmark](#4-cardioqa-india-v2-qworld-enhanced-benchmark)
5. [Qworld-in-the-Loop Training (GRPO Enhancement)](#5-qworld-in-the-loop-training)
6. [Clinical Response Quality Assessment Pipeline](#6-clinical-response-quality-assessment-pipeline)
7. [MARCUS Architecture (unchanged from v3)](#7-marcus-architecture)
8. [Clinical Knowledge Engine (unchanged from v3)](#8-clinical-knowledge-engine)
9. [Safety: Mirage Detection + Qworld Cross-Validation](#9-safety-mirage-detection--qworld-cross-validation)
10. [Updated Evaluation Framework](#10-updated-evaluation-framework)
11. [Implementation Additions](#11-implementation-additions)
12. [Updated Roadmap](#12-updated-roadmap)

---

## 1. What Changed from v3

| Aspect | v3 Specification | v5 Addition |
|--------|-----------------|-------------|
| **Evaluation** | 22 static MCQs (CardioQA-India) + simple accuracy | Qworld RET generates ~45 criteria per clinical question across 25+ dimensions |
| **Benchmark** | Binary correct/incorrect scoring | Weighted multi-criteria scoring: each criterion has importance weight α_c, positive (desirable) and negative (penalizing) |
| **Quality dimensions** | Accuracy only | Safety & Risk Management, Guideline Adherence, Cultural Sensitivity, Health Equity, Caregiver Support, Follow-Up Continuity, Evidence Quality, Patient Empowerment |
| **GRPO reward** | Binary correctness (1/0) | Qworld criteria satisfaction score as reward signal (richer gradient) |
| **Response assessment** | Not implemented | Full Qworld pipeline: scenario expansion -> perspective expansion -> criteria expansion -> LLM-as-judge scoring |
| **Mirage detection** | Counterfactual probe + consistency check | + Qworld criteria cross-validation (does the response address criteria that REQUIRE visual evidence?) |
| **Training data quality** | Quality gate (relevance + completeness) | + Qworld-generated criteria used to evaluate and filter training data quality |
| **Patient communication** | Free-text Hindi/English output | Qworld criteria for patient-facing communication: clarity, actionability, literacy-appropriate language, cultural sensitivity |

---

## 2. Qworld Integration Architecture

### 2.1 What is Qworld

Qworld (One-Question-One-World) from Harvard Medical School generates question-specific evaluation criteria using a **Recursive Expansion Tree (RET)**:

```
Question (clinical case)
  |
  ├── Scenario 1 (e.g., "52yo Indian male, STEMI presentation")
  │     ├── Perspective 1 (Diagnostic Accuracy)
  │     │     ├── Criterion: "Identifies ST elevation in anterior leads"  [+10 pts]
  │     │     ├── Criterion: "Mentions troponin elevation timeline"       [+5 pts]
  │     │     └── Criterion: "Does NOT recommend thrombolytics if >12h"   [-10 pts]
  │     ├── Perspective 2 (Safety & Risk Management)
  │     │     ├── Criterion: "Recommends immediate cardiac catheterization" [+10 pts]
  │     │     ├── Criterion: "Checks for contraindications to anticoagulation" [+5 pts]
  │     │     └── Criterion: "Warns about reperfusion arrhythmias"         [+3 pts]
  │     └── Perspective 3 (Indian Guideline Adherence)
  │           ├── Criterion: "Applies South Asian BMI threshold (>=23)"    [+5 pts]
  │           ├── Criterion: "Mentions MYBPC3 screening if HCM suspected"  [+3 pts]
  │           └── Criterion: "Recommends SGLT2i for diabetic patient"      [+5 pts]
  ├── Scenario 2 (e.g., "rural PHC, limited cath lab access")
  │     ├── Perspective: Resource-Appropriate Management
  │     │     ├── Criterion: "Provides thrombolytic alternative if cath unavailable" [+8 pts]
  │     │     └── Criterion: "Specifies transfer protocol to nearest tertiary center" [+5 pts]
  │     └── Perspective: CHW Communication
  │           ├── Criterion: "Action items are simple enough for ASHA worker"  [+5 pts]
  │           └── Criterion: "Includes emergency transport instructions"       [+5 pts]
  └── Scenario 3 (e.g., "patient is a 45yo diabetic woman")
        └── Perspective: Gender & Comorbidity Equity
              ├── Criterion: "Acknowledges atypical presentation in women"     [+5 pts]
              └── Criterion: "Does NOT dismiss symptoms as non-cardiac"        [-8 pts]
```

### 2.2 Key Metrics from Qworld Paper

| Metric | Qworld Score | Best Prior Method | Significance for Cardio-Sahayak |
|--------|-------------|-------------------|-------------------------------|
| **Coverage** | 0.89 | 0.83 (EvalAgent) | Captures 89% of expert-authored medical criteria |
| **Uniqueness** | 0.79 | 0.50 (EvalAgent) | 79% of criteria are novel (experts miss them) |
| **Insight** (human) | 0.83 | 0.42 (RocketEval) | Criteria capture non-obvious safety & equity concerns |
| **Granularity** (human) | 0.85 | 0.83 (RocketEval) | Criteria are specific and actionable, not vague |
| **Specificity** (NIWF) | 0.09 | 0.04 (EvalAgent) | Uses domain-specific medical vocabulary |
| **Implicitness** | 0.87 | 0.83 (EvalAgent) | Surfaces requirements not stated in the question |

**Critical finding**: On HealthBench, models scored ~20% lower under Qworld criteria than expert criteria, because Qworld catches omissions that experts gloss over (missing caveats, incomplete follow-up, underspecified risk management). This makes Qworld **harder** and **more discriminating** -- exactly what we need for medical AI where "good enough" is not acceptable.

### 2.3 How Qworld Integrates with Cardio-Sahayak

```
                    +---------------------------+
                    |  Clinical Question/Case    |
                    +-------------+-------------+
                                  |
              +-------------------v-------------------+
              |         QWORLD CRITERIA GENERATOR      |
              |  Recursive Expansion Tree (RET)        |
              |  - 3 scenario expansions               |
              |  - 4 perspective expansions per scenario|
              |  - 3 criteria expansions per perspective|
              |  Average: ~45 criteria per question     |
              +-------------------+-------------------+
                                  |
                    +-------------v-------------+
                    |   Question-Specific        |
                    |   Evaluation Criteria       |
                    |   (weighted binary items)   |
                    +-------------+-------------+
                                  |
              +-------------------v-------------------+
              |                                        |
    +---------v---------+              +-------v--------+
    | TRAINING (GRPO)   |              | EVALUATION     |
    | Use criteria       |              | LLM-as-Judge   |
    | satisfaction as    |              | scores response |
    | reward signal      |              | against criteria|
    | (richer than 1/0)  |              |                |
    +-------------------+              +----------------+
```

### 2.4 Integration via pip

```python
# Qworld is pip-installable
# pip install qworld

from qworld import CriteriaGenerator

gen = CriteriaGenerator(
    model="claude-sonnet-4-6",  # or gpt-4.1, gemini-3-flash
    n_scenario_expands=3,
    n_perspective_expands=4,
    n_criteria_expands=3,
    dedup_threshold=0.7,
    temperature=0.4,
)

# Generate criteria for a clinical case
result = gen.generate(
    "A 52-year-old Indian male with BMI 25.5, diabetes, hypertension, "
    "and chest pain on exertion presents to a PHC. ECG shows ST elevation "
    "in V2-V4. What is the diagnosis, risk stratification, and management plan?"
)

criteria = result["final_criteria"]
# Returns ~45 weighted binary criteria covering:
# - Diagnostic accuracy
# - Safety & risk management
# - Indian guideline adherence
# - Patient communication
# - Resource-appropriate management
# - Gender/equity considerations
# - Follow-up & continuity
```

---

## 3. India-Specific Qworld Evaluation Dimensions

### 3.1 Standard Qworld Dimensions (from HealthBench taxonomy)

Qworld on HealthBench generates criteria across these expert-validated dimensions:
- Accuracy, Completeness, Safety & Risk Management
- Empathy & Support, Evidence Quality, Factual Correctness
- Follow-Up & Continuity, Guideline Adherence, Health Equity & Accessibility
- Personalization, Caregiver Support, Clarity, Cultural Sensitivity
- Emergency Recognition, Shared Decision-Making, Transparency
- Sustainability & Long-term Impact

### 3.2 India-Specific Dimensions (novel, to be added via custom prompting)

We extend Qworld's default perspectives with India-specific dimensions through custom scenario and perspective prompts:

| Dimension | What It Evaluates | Why It Matters for India |
|-----------|-------------------|------------------------|
| **South Asian Phenotype Awareness** | Does the response adjust for SA-specific risk factors (lower BMI thresholds, premature CAD, MYBPC3, Lp(a))? | Generic models apply Western thresholds, missing 5-10 year earlier MI onset |
| **Indian Guideline Compliance** | Does the response align with IJAM 2023 Consensus, not just ACC/AHA/ESC? | Indian guidelines have specific Diamond Approach tables, aggressive LDL targets for diabetics |
| **NLEM Drug Availability** | Are recommended drugs available in the Indian National List of Essential Medicines? | Recommending ivabradine in a rural PHC where only metoprolol is available is useless |
| **Resource-Stratified Management** | Does the response differentiate management by setting (PHC vs CHC vs tertiary)? | India has extreme variability in healthcare infrastructure |
| **Diabetes-Cardiac Integration** | Does the response address the 50-60% CAD-DM overlap? | India is the diabetes capital; most cardiac patients also have diabetes |
| **Hindi/Regional Language Appropriateness** | For patient-facing output, is the Hindi clinically accurate AND understandable? | Low health literacy requires careful language calibration |
| **CHW Actionability** | Can an ASHA/ANM worker understand and act on the output? | CHWs are the frontline in rural India; output must be simple |
| **Family/Caregiver Inclusion** | Does the response include guidance for family caregivers? | Indian healthcare is family-centric; patients rarely manage alone |
| **Cost Sensitivity** | Does the response consider affordability of investigations/drugs? | Out-of-pocket healthcare expenditure is a major barrier |
| **Referral Appropriateness** | Is the referral urgency and destination correct for the clinical scenario? | Over-referral wastes limited tertiary resources; under-referral costs lives |

### 3.3 Custom Qworld System Prompt for Indian Cardiology

```python
INDIA_CARDIOLOGY_SYSTEM_PROMPT = """
When generating evaluation criteria for Indian cardiology clinical questions,
ensure the following India-specific perspectives are explored:

1. SOUTH ASIAN PHENOTYPE: BMI >=23 (not 25) is overweight. MI onset 5-10 years
   earlier. Screen MYBPC3 Delta-25bp in HCM/HF (4% prevalence). Elevated Lp(a)
   as independent risk factor.

2. INDIAN GUIDELINES: Apply IJAM 2023 Diamond Approach for drug selection.
   LDL <70 mg/dl for diabetic CAD patients. BP <=130/80 for diabetics.
   SGLT2i/GLP-1RA for diabetic CVD.

3. RESOURCE STRATIFICATION: Differentiate between rural PHC (no cath lab,
   basic drugs only), CHC (limited investigations), district hospital
   (echo available), and tertiary center (full cardiac services).

4. DIABETES INTEGRATION: 50-60% of Indian CAD patients have diabetes.
   Every cardiac assessment should address glycemic control, HbA1c targets,
   and cardio-renal-metabolic integration.

5. DRUG AVAILABILITY: Check against Indian NLEM 2022. Flag drugs not widely
   available in government healthcare facilities.

6. CULTURAL CONTEXT: Family-centric care decisions, tobacco includes bidi/gutka
   (not just cigarettes), diet context (ghee, salt, vegetarian considerations).

7. LANGUAGE: Patient-facing content must be appropriate for low health literacy.
   Hindi transliteration should use simple words, not medical jargon.
"""
```

---

## 4. CardioQA-India v2: Qworld-Enhanced Benchmark

### 4.1 The Problem with v1

CardioQA-India v1 has 22 static MCQs with binary correct/incorrect scoring. This is fundamentally insufficient:
- It tests recall (can the model pick the right answer?) not clinical reasoning
- It has no mechanism to detect omissions (what the model should say but doesn't)
- It can't distinguish between a model that gives a correct but dangerous answer vs. a correct and safe answer
- It saturates quickly -- all models will eventually score >90% on 22 MCQs

### 4.2 CardioQA-India v2 Design

**Structure**: 200 open-ended clinical questions (not MCQ) with Qworld-generated criteria.

Each question gets ~45 Qworld criteria, yielding ~9,000 total evaluation checkpoints (vs. 22 in v1).

**Question Categories** (200 total):

| Category | Count | Example |
|----------|-------|---------|
| Acute presentation (STEMI/NSTEMI/UA) | 30 | "48yo Indian male, crushing chest pain 2 hours, sweating, ECG shows..." |
| Chronic stable angina management | 25 | "55yo diabetic woman with exertional angina, on metoprolol + atorvastatin..." |
| Heart failure (HFrEF/HFpEF) | 20 | "62yo male, LVEF 30%, BNP 1200, bilateral pedal edema..." |
| Atrial fibrillation | 15 | "68yo hypertensive male, irregular pulse, palpitations..." |
| Valvular heart disease | 15 | "45yo female, progressive dyspnea, systolic murmur at apex..." |
| Hypertrophic cardiomyopathy | 10 | "35yo Indian male, family Hx of sudden death, MYBPC3 carrier..." |
| Diabetic-cardiac comorbidity | 25 | "52yo diabetic, HbA1c 9.2%, LDL 165, on metformin only..." |
| Risk stratification (asymptomatic) | 15 | "40yo Indian male, BMI 24.5, family Hx CAD, no symptoms..." |
| Women-specific cardiac care | 10 | "48yo postmenopausal woman, atypical chest discomfort..." |
| CHW screening scenarios | 15 | "ASHA worker screens 55yo male with BP 160/95, diabetic, tobacco user..." |
| Patient education | 10 | "Patient asks: What does my heart risk level mean?" |
| Resource-limited settings | 10 | "Rural PHC, no echo/cath lab, patient with suspected ACS..." |

### 4.3 Qworld Criteria Generation Pipeline

```python
from qworld import CriteriaGenerator
from pathlib import Path
import json

gen = CriteriaGenerator(
    model="claude-sonnet-4-6",
    n_scenario_expands=3,
    n_perspective_expands=4,
    n_criteria_expands=3,
)

questions = load_cardioqa_v2_questions()  # 200 clinical questions

for q in questions:
    result = gen.generate(
        q["question"],
        system_prompt=INDIA_CARDIOLOGY_SYSTEM_PROMPT,  # India-specific perspectives
    )
    q["qworld_criteria"] = result["final_criteria"]
    q["scenarios"] = result["scenarios"]
    q["perspectives"] = result["reviewed_perspectives"]

# Save: ~200 questions x ~45 criteria = ~9,000 evaluation checkpoints
save_benchmark("data/benchmarks/cardioqa_india_v2_qworld.jsonl", questions)
```

### 4.4 Scoring Protocol

For each question Q and model response A:

```
For each criterion c_i in Q's Qworld criteria:
    s_c(A, Q) = α_c   if LLM-judge determines A satisfies c_i
               0     otherwise

    where α_c > 0 rewards desirable attributes
    and   α_c < 0 penalizes harmful attributes

S(A, Q) = Σ s_c(A, Q) / Σ α_c   (for α_c > 0 only, per HealthBench normalization)
```

This produces a **normalized score in [negative, 1.0]** where:
- 1.0 = all positive criteria satisfied, no negative criteria triggered
- 0.0 = no criteria satisfied
- Negative = model triggered penalty criteria (e.g., recommended contraindicated drug)

---

## 5. Qworld-in-the-Loop Training (GRPO Enhancement)

### 5.1 The Limitation of Binary GRPO

In v3, GRPO uses binary correctness reward (1.0 for correct MCQ, 0.0 for incorrect). This provides minimal gradient signal -- the model learns "was I right?" but not "how comprehensively right was I?"

### 5.2 Qworld Criteria as GRPO Reward

Replace binary reward with **Qworld criteria satisfaction score**:

```python
def qworld_reward(question: str, completion: str, criteria: list[dict]) -> float:
    """
    Score a completion against Qworld-generated criteria.
    Returns normalized score in [-1, 1] range.

    Much richer signal than binary 1/0:
    - Partial credit for partially correct answers
    - Penalties for harmful/dangerous recommendations
    - Rewards for addressing safety, equity, follow-up
    """
    satisfied_score = 0
    max_positive_score = 0

    for criterion in criteria:
        alpha = criterion["points"]
        if alpha > 0:
            max_positive_score += alpha

        # LLM-as-judge check (or rule-based for deterministic criteria)
        if criterion_satisfied(completion, criterion["text"]):
            satisfied_score += alpha

    if max_positive_score == 0:
        return 0.0
    return satisfied_score / max_positive_score
```

### 5.3 Training Impact

| Reward Type | Signal Richness | Example |
|-------------|----------------|---------|
| **Binary (v3)** | 1 bit per question | "Correct diagnosis" = 1.0 |
| **Qworld (v5)** | ~45 bits per question | "Correct diagnosis" +10, "mentioned MYBPC3" +3, "recommended contraindicated drug" -10, "provided follow-up plan" +5, "addressed patient anxiety" +3, ... = 0.72 |

The richer reward signal means the model learns:
- Not just "is the diagnosis right?" but "is the management plan complete?"
- Penalty signals for dangerous recommendations (negative α_c)
- Reward for addressing dimensions experts consider important but don't explicitly ask about (Qworld's 79% novel criteria)

---

## 6. Clinical Response Quality Assessment Pipeline

### 6.1 End-to-End Assessment Flow

```
Patient Case Input
       |
       v
+------+--------+
| Cardio-Sahayak |
| Orchestrator   |
| (expert reports|
|  + guidelines) |
+------+--------+
       |
       v
Model Response (structured JSON + natural language)
       |
       +-----> Qworld Criteria Generator (if not pre-generated)
       |            |
       |            v
       |       ~45 Question-Specific Criteria
       |            |
       v            v
+------+------------+--------+
|    LLM-as-Judge              |
|    (scores response against  |
|     each criterion)          |
+------+----------------------+
       |
       v
+------+----------------------+
| Quality Report               |
| - Overall score (0-100%)     |
| - Per-dimension breakdown    |
| - Specific criteria failures |
| - Comparison to prior runs   |
+------------------------------+
```

### 6.2 Assessment Dimensions (from Qworld HealthBench taxonomy + India extensions)

**Core Medical Dimensions** (from Qworld HealthBench):
1. Diagnostic Accuracy
2. Safety & Risk Management
3. Evidence Quality & Factual Correctness
4. Guideline Adherence
5. Completeness
6. Emergency Recognition
7. Follow-Up & Continuity

**Patient-Facing Dimensions** (from Qworld):
8. Clarity & Communication Quality
9. Empathy & Support
10. Patient Empowerment & Shared Decision-Making
11. Caregiver Support
12. Health Literacy Adaptation
13. Cultural Sensitivity

**India-Specific Dimensions** (novel):
14. South Asian Phenotype Awareness
15. Indian Guideline Compliance (IJAM 2023)
16. NLEM Drug Availability
17. Resource-Stratified Management
18. Diabetes-Cardiac Integration
19. Hindi/Regional Language Quality
20. CHW Actionability
21. Referral Appropriateness
22. Cost Sensitivity
23. Family-Centric Care
24. Tobacco Context (bidi/gutka, not just cigarettes)
25. Dietary Context (ghee, vegetarian, regional cuisine)

### 6.3 Radar Chart Output (inspired by Qworld Figure 4)

For each model evaluation run, generate a radar chart across all 25 dimensions, enabling:
- **Comparison between model versions** (did v3 training improve safety but decrease completeness?)
- **Comparison between user modes** (doctor mode vs patient mode -- different dimension priorities)
- **Identification of systematic weaknesses** (e.g., all models score low on CHW Actionability)
- **Targeted improvement** (if Hindi Language Quality scores low, add more Hindi training data)

---

## 7. MARCUS Architecture (unchanged from v3)

The agentic architecture remains as specified in v3:
- **Orchestrator**: Qwen2.5-VL-7B-Instruct (~4.5GB GGUF)
- **ECG Expert**: Qwen2.5-VL-3B-Instruct (~2GB GGUF)
- **Echo Expert**: Qwen2.5-VL-3B-Instruct (~2GB GGUF)
- **Clinical Expert**: Qwen2.5-3B-Instruct (~2GB GGUF)
- **Communication**: Natural language between experts and orchestrator
- **Training**: 3-stage CPT -> SFT -> GRPO (now with Qworld-enhanced reward)

See v3 spec sections 2-6 for full architecture details.

---

## 8. Clinical Knowledge Engine (unchanged from v3)

All deterministic guideline modules remain as implemented:
- `src/knowledge/diamond_approach.py` -- IJAM Table 3
- `src/knowledge/chest_pain_scoring.py` -- IJAM Table 1
- `src/knowledge/comorbidity_checklist.py` -- IJAM Table 2
- `src/knowledge/risk_stratification.py` -- ESC 2019 + India
- `src/knowledge/treatment_targets.py` -- LDL/BP/HbA1c targets
- `src/knowledge/south_asian_phenotype.py` -- BMI/MYBPC3/Lp(a)
- `src/knowledge/drug_contraindications.py` -- NLEM flags

All 20 unit tests pass. See v3 spec section 3 for details.

---

## 9. Safety: Mirage Detection + Qworld Cross-Validation

### 9.1 Existing Safety Layers (from v3)

1. **Layer 1**: Counterfactual image-absent probe (from MARCUS)
2. **Layer 2**: 3-rephrasing consistency check (from MARCUS)
3. **Layer 3**: Deterministic cross-check against Knowledge Engine

### 9.2 New Layer 4: Qworld Criteria Cross-Validation

Qworld criteria can detect a class of errors that mirage detection and deterministic cross-checking miss: **omission errors** and **context-inappropriate responses**.

**How it works**: Generate Qworld criteria for the clinical question. After the model responds, check whether criteria that *require* visual evidence (e.g., "Identifies ST elevation in V2-V4") are satisfied. If the model satisfies such criteria WITHOUT having been given an image, this is a more nuanced form of mirage detection -- the model is fabricating findings it couldn't possibly know from text alone.

**Omission detection**: Qworld criteria with high importance weights (α_c > 5) that are NOT satisfied represent critical omissions. The cross-validator flags these:
- "Did NOT mention troponin timeline" (α = 5, FAILED)
- "Did NOT address medication interactions" (α = 8, FAILED)
- "Did NOT provide follow-up plan" (α = 5, FAILED)

These omission flags are injected into the Doctor Dashboard as warnings.

### 9.3 Safety Layer Summary (v5)

| Layer | Method | Detects | Source |
|-------|--------|---------|--------|
| 1 | Counterfactual probe | Visual hallucination | MARCUS |
| 2 | 3-rephrasing consistency | Output instability | MARCUS |
| 3 | Knowledge Engine cross-check | Guideline contradictions | v3 original |
| 4 | Qworld criteria cross-validation | Omissions + context errors | v5 new |

---

## 10. Updated Evaluation Framework

### 10.1 Evaluation Tiers

**Tier 1: Deterministic Knowledge Engine Tests** (existing, 20 tests)
- Unit tests for all clinical scoring modules
- Runtime: <1 second, no GPU needed
- Purpose: Ensure guideline logic is correct

**Tier 2: CardioQA-India v1 MCQs** (existing, 22 questions)
- Binary accuracy on static MCQs
- Runtime: minutes with GGUF, seconds with GPU
- Purpose: Quick regression check

**Tier 3: CardioQA-India v2 Qworld** (new, 200 questions x ~45 criteria)
- Qworld criteria-based scoring with LLM-as-judge
- Runtime: ~1 hour with API-based judge (GPT-4.1 or Claude)
- Purpose: Comprehensive multi-dimensional assessment
- Produces radar charts across 25 dimensions

**Tier 4: Cardiologist Blinded Review** (planned)
- 50+ Indian cardiologists evaluate on 100 cases
- AMIE 10-domain rubric + Qworld-generated criteria
- Purpose: Gold-standard clinical validation

### 10.2 Qworld vs. Static Evaluation Comparison

| Metric | CardioQA-India v1 (static) | CardioQA-India v2 (Qworld) |
|--------|---------------------------|---------------------------|
| Questions | 22 MCQ | 200 open-ended |
| Evaluation points | 22 (1 per question) | ~9,000 (~45 per question) |
| Dimensions assessed | 1 (accuracy) | 25 (safety, equity, guidelines, ...) |
| Score saturation | High (models plateau at >90%) | Low (~20% score reduction vs static, per Qworld paper) |
| Omission detection | None | Yes (unsatisfied high-weight criteria) |
| Penalty for harm | None | Yes (negative α_c for dangerous recommendations) |
| India-specific | Partially (questions are India-focused) | Fully (criteria cover SA phenotype, NLEM, resource stratification) |
| Cost | Free (deterministic) | ~$20 per eval run (API calls for LLM-as-judge) |

### 10.3 Target Metrics

| Dimension | Target Score | Justification |
|-----------|-------------|---------------|
| Diagnostic Accuracy | >0.80 | MARCUS ECG expert achieves 87-91% |
| Safety & Risk Management | >0.85 | Safety is non-negotiable in clinical AI |
| Indian Guideline Compliance | >0.90 | Deterministic Knowledge Engine should guarantee this |
| NLEM Drug Availability | >0.95 | Binary check against NLEM database |
| South Asian Phenotype Awareness | >0.80 | Core differentiation of Cardio-Sahayak |
| Patient Communication Clarity | >0.75 | Harder -- requires natural language quality |
| CHW Actionability | >0.80 | Simple checklist output should score well |
| Overall Qworld Score | >0.50 | Qworld baseline across frontier models is 26-35% (Table 2) |

---

## 11. Implementation Additions

### 11.1 New Files (v5 additions to existing codebase)

```
src/evaluation/
  qworld_integration.py        -- Qworld CriteriaGenerator wrapper with India-specific prompts
  qworld_evaluator.py          -- LLM-as-judge scoring against Qworld criteria
  cardioqa_india_v2.py         -- V2 benchmark with open-ended questions + Qworld criteria
  cardioqa_v2_generator.py     -- Generate 200 clinical questions from templates
  dimension_radar.py           -- Radar chart visualization across 25 dimensions
  qworld_training_reward.py    -- Qworld criteria as GRPO reward function

src/safety/
  qworld_cross_validator.py    -- Layer 4 safety: criteria cross-validation for omissions

data/benchmarks/
  cardioqa_india_v2_questions.jsonl     -- 200 open-ended clinical questions
  cardioqa_india_v2_qworld.jsonl        -- Questions + Qworld-generated criteria (~9K criteria)
```

### 11.2 Dependencies Addition

```
# requirements.txt additions
qworld>=0.1.0                    # Qworld criteria generation
# qworld[local-embeddings]       # Optional: local embeddings for dedup (avoids API calls)
```

### 11.3 Integration Points with Existing Code

| Existing Module | v5 Integration |
|----------------|----------------|
| `src/evaluation/run_all_evals.py` | Add Tier 3 Qworld evaluation |
| `src/training/stage3_grpo.py` | Replace `binary_correctness_reward` with `qworld_reward` |
| `src/safety/mirage_detector.py` | Add Layer 4 criteria cross-validation |
| `src/ui/doctor_dashboard.py` | Add Qworld quality score + omission warnings |
| `src/orchestrator/orchestrator.py` | Run Qworld assessment on orchestrator output |

---

## 12. Updated Roadmap

### Phase 0-5: Completed (as per v3 spec)

All 45 Python modules implemented and tested. See git log for commits.

### Phase 6: Qworld Integration (Weeks 37-42)

| Task | Deliverable |
|------|-------------|
| `pip install qworld` + integration wrapper | `src/evaluation/qworld_integration.py` |
| Generate 200 open-ended clinical questions | `data/benchmarks/cardioqa_india_v2_questions.jsonl` |
| Run Qworld RET on all 200 questions with India-specific prompts | `data/benchmarks/cardioqa_india_v2_qworld.jsonl` (~9K criteria) |
| Build LLM-as-judge evaluator | `src/evaluation/qworld_evaluator.py` |
| Build 25-dimension radar chart | `src/evaluation/dimension_radar.py` |
| Integrate Qworld reward into GRPO | `src/evaluation/qworld_training_reward.py` |
| Add Layer 4 safety cross-validation | `src/safety/qworld_cross_validator.py` |
| Run baseline evaluation on current system | Radar chart + scores for v3 models |

### Phase 7: Qworld-Enhanced GRPO Training (Weeks 43-48)

| Task | Deliverable |
|------|-------------|
| Generate Qworld criteria for all v3 training data | Quality-scored training set |
| Filter training data by Qworld quality score | Remove low-quality records |
| Re-run GRPO with Qworld reward function | v4 expert models with multi-criteria optimization |
| Evaluate v4 models on CardioQA-India v2 | Comparative radar charts (v3 vs v4) |

### Phase 8: Clinical Validation + Pilot (Weeks 49+)

| Task | Deliverable |
|------|-------------|
| Cardiologist blinded review with Qworld criteria | 50+ cardiologists, 100 cases |
| Compare Qworld vs expert criteria coverage | Coverage/Uniqueness metrics |
| Pilot deployment with Qworld quality monitoring | Real-time quality scoring in production |

---

## Appendix A: Qworld Paper Key Results

**Coverage comparison** (Table 1):
- TICK: 0.46 Coverage, 0.24 Uniqueness
- RocketEval: 0.53 Coverage, 0.26 Uniqueness
- OpenRubrics: 0.54 Coverage, 0.37 Uniqueness
- EvalAgent: 0.83 Coverage, 0.50 Uniqueness
- **Qworld: 0.89 Coverage, 0.79 Uniqueness**
- **Qworld (retrieval-augmented): 0.90 Coverage, 0.82 Uniqueness**

**HealthBench model rankings change** (Table 2):
- Under Qworld criteria, Qwen3-30B moves from 6th to 2nd (strong on patient-facing dimensions)
- GPT-5 remains #1 but scores ~20% lower (more omissions detected)
- Score compression: all models ~20% lower, better discrimination

**Score saturation** (Figure 12):
- As criteria count increases from 20 to 100, model rankings remain stable but absolute scores decrease
- This means added criteria are meaningful (not noise) and raise the evaluation bar

## Appendix B: Reference Papers

1. **MARCUS**: O'Sullivan et al. "An agentic, multimodal vision-language model for cardiac diagnosis and management." arXiv 2603.22179v1, March 2026.
2. **Qworld**: Gao et al. "Question-Specific Evaluation Criteria for LLMs." arXiv 2603.23522v1, March 2026.
3. **Indian Consensus**: Balaji et al. "Consensus statement on cardiovascular risk stratification and aggressive management of chronic coronary syndromes." IJAM 2023;10(5):425-432.
4. **HealthBench**: Arora et al. "Evaluating large language models towards improved human health." arXiv 2505.08775, 2025.
5. **AMIE**: "A large language model for complex cardiology care." Nature Medicine, 2025.

## Appendix C: Qworld Dimensions Mapping to Cardio-Sahayak User Modes

| Qworld Dimension | Doctor Dashboard | Patient Portal | CHW Screener |
|-----------------|-----------------|----------------|--------------|
| Diagnostic Accuracy | Primary | Low priority | Not shown |
| Safety & Risk Management | Primary | Simplified | Color-coded |
| Guideline Adherence | Primary | Not shown | Not shown |
| Evidence Quality | Primary | Not shown | Not shown |
| Completeness | Primary | Moderate | Low priority |
| Emergency Recognition | High priority | High priority | Primary |
| Follow-Up & Continuity | Primary | High priority | Moderate |
| Clarity | Moderate | Primary | Primary |
| Empathy & Support | Low priority | Primary | Moderate |
| Cultural Sensitivity | Moderate | Primary | Primary |
| Patient Empowerment | Low priority | Primary | Low priority |
| Caregiver Support | Moderate | Primary | Primary |
| South Asian Phenotype | Primary | Simplified | Not shown |
| NLEM Availability | Primary | Not shown | Not shown |
| Resource Stratification | Primary | Not shown | Primary |
| CHW Actionability | Not applicable | Not applicable | Primary |
| Hindi Quality | Not applicable | Primary | Primary |
| Cost Sensitivity | Moderate | High priority | Moderate |
| Referral Appropriateness | Primary | Simplified | Primary |
