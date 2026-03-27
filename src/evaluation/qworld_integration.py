from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    from qworld import CriteriaGenerator
    HAS_QWORLD = True
except ImportError:
    HAS_QWORLD = False


INDIA_CARDIOLOGY_SYSTEM_PROMPT = (
    "When generating evaluation criteria for Indian cardiology clinical questions, "
    "ensure the following India-specific perspectives are explored:\n\n"
    "1. SOUTH ASIAN PHENOTYPE: BMI >=23 (not 25) is overweight. MI onset 5-10 years "
    "earlier. Screen MYBPC3 Delta-25bp in HCM/HF (4% prevalence). Elevated Lp(a) "
    "as independent risk factor.\n\n"
    "2. INDIAN GUIDELINES: Apply IJAM 2023 Diamond Approach for drug selection. "
    "LDL <70 mg/dl for diabetic CAD patients. BP <=130/80 for diabetics. "
    "SGLT2i/GLP-1RA for diabetic CVD.\n\n"
    "3. RESOURCE STRATIFICATION: Differentiate between rural PHC (no cath lab, "
    "basic drugs only), CHC (limited investigations), district hospital "
    "(echo available), and tertiary center (full cardiac services).\n\n"
    "4. DIABETES INTEGRATION: 50-60% of Indian CAD patients have diabetes. "
    "Every cardiac assessment should address glycemic control, HbA1c targets, "
    "and cardio-renal-metabolic integration.\n\n"
    "5. DRUG AVAILABILITY: Check against Indian NLEM 2022. Flag drugs not widely "
    "available in government healthcare facilities.\n\n"
    "6. CULTURAL CONTEXT: Family-centric care decisions, tobacco includes bidi/gutka "
    "(not just cigarettes), diet context (ghee, salt, vegetarian considerations).\n\n"
    "7. LANGUAGE: Patient-facing content must be appropriate for low health literacy. "
    "Hindi transliteration should use simple words, not medical jargon."
)


@dataclass
class QworldConfig:
    model: str = "gpt-4.1"
    n_scenario_expands: int = 3
    n_perspective_expands: int = 4
    n_criteria_expands: int = 3
    dedup_threshold: float = 0.7
    max_workers: int = 8
    temperature: float = 0.4
    system_prompt: str = INDIA_CARDIOLOGY_SYSTEM_PROMPT


@dataclass
class CriterionItem:
    text: str
    points: float
    dimension: str = ""


@dataclass
class QworldResult:
    question: str
    criteria: list[CriterionItem] = field(default_factory=list)
    scenarios: list[str] = field(default_factory=list)
    perspectives: list[str] = field(default_factory=list)
    total_positive_points: float = 0.0
    total_negative_points: float = 0.0


def create_generator(config: QworldConfig | None = None) -> "CriteriaGenerator | None":
    if not HAS_QWORLD:
        return None
    if config is None:
        config = QworldConfig()
    return CriteriaGenerator(
        model=config.model,
        n_scenario_expands=config.n_scenario_expands,
        n_perspective_expands=config.n_perspective_expands,
        n_criteria_expands=config.n_criteria_expands,
        dedup_threshold=config.dedup_threshold,
        max_workers=config.max_workers,
        temperature=config.temperature,
    )


def generate_criteria(
    question: str,
    generator: "CriteriaGenerator | None" = None,
    config: QworldConfig | None = None,
) -> QworldResult:
    if generator is None:
        generator = create_generator(config)
    if generator is None:
        return _generate_fallback_criteria(question)

    cfg = config or QworldConfig()

    raw = generator.generate(question, system_prompt=cfg.system_prompt)

    criteria = []
    for c in raw.get("final_criteria", []):
        text = c.get("criterion", c.get("text", ""))
        points = c.get("points", c.get("weight", 1.0))
        criteria.append(CriterionItem(text=text, points=float(points)))

    pos = sum(c.points for c in criteria if c.points > 0)
    neg = sum(c.points for c in criteria if c.points < 0)

    return QworldResult(
        question=question,
        criteria=criteria,
        scenarios=raw.get("scenarios", []),
        perspectives=raw.get("reviewed_perspectives", raw.get("perspectives", [])),
        total_positive_points=pos,
        total_negative_points=neg,
    )


def _generate_fallback_criteria(question: str) -> QworldResult:
    criteria = [
        CriterionItem("Provides a specific diagnosis or differential diagnosis", 10),
        CriterionItem("Mentions relevant risk factors for South Asian patients", 5),
        CriterionItem("Recommends appropriate investigations", 5),
        CriterionItem("Provides a management plan aligned with Indian guidelines", 10),
        CriterionItem("Addresses medication with dose and rationale", 5),
        CriterionItem("Mentions follow-up plan and monitoring", 5),
        CriterionItem("Identifies emergency warning signs", 5),
        CriterionItem("Applies South Asian BMI thresholds (>=23 overweight)", 3),
        CriterionItem("Considers drug availability in Indian NLEM", 3),
        CriterionItem("Does NOT recommend contraindicated medications", -10),
        CriterionItem("Does NOT dismiss symptoms without adequate workup", -8),
        CriterionItem("Does NOT apply Western-only risk thresholds", -5),
    ]
    pos = sum(c.points for c in criteria if c.points > 0)
    neg = sum(c.points for c in criteria if c.points < 0)
    return QworldResult(
        question=question,
        criteria=criteria,
        total_positive_points=pos,
        total_negative_points=neg,
    )


def generate_batch_criteria(
    questions: list[dict],
    output_path: str | Path,
    config: QworldConfig | None = None,
) -> int:
    generator = create_generator(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for q in questions:
            question_text = q.get("question", q.get("text", ""))
            if not question_text:
                continue

            result = generate_criteria(question_text, generator, config)

            record = {
                "question": result.question,
                "criteria": [{"text": c.text, "points": c.points, "dimension": c.dimension} for c in result.criteria],
                "scenarios": result.scenarios,
                "perspectives": result.perspectives,
                "total_positive_points": result.total_positive_points,
                "total_negative_points": result.total_negative_points,
                **{k: v for k, v in q.items() if k not in ("question", "text")},
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Generated criteria for {count} questions -> {output_path}")
    return count
