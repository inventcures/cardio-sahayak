from dataclasses import dataclass, field

from src.evaluation.qworld_integration import CriterionItem
from src.evaluation.qworld_evaluator import _keyword_judge


@dataclass
class CrossValidationResult:
    critical_omissions: list[str] = field(default_factory=list)
    visual_evidence_flags: list[str] = field(default_factory=list)
    safety_gaps: list[str] = field(default_factory=list)
    passed: bool = True
    omission_score: float = 0.0


VISUAL_EVIDENCE_KEYWORDS = [
    "ecg", "electrocardiogram", "st elevation", "st depression",
    "t-wave", "qrs", "rhythm strip", "12-lead",
    "echo", "echocardiogram", "lvef", "wall motion",
    "ejection fraction", "valvular", "pericardial",
    "ultrasound", "doppler",
]

SAFETY_CRITICAL_KEYWORDS = [
    "emergency", "immediate", "urgent", "stat",
    "contraindicated", "do not", "avoid", "warning",
    "anaphylaxis", "bleeding", "hemorrhage",
    "cardiac arrest", "code blue", "resuscitation",
]


def cross_validate_response(
    response: str,
    criteria: list[CriterionItem],
    has_image_input: bool = False,
    judge_fn=None,
) -> CrossValidationResult:
    if judge_fn is None:
        judge_fn = _keyword_judge

    result = CrossValidationResult()
    response_lower = response.lower()

    high_weight_threshold = 5.0
    omitted_weight = 0.0
    total_high_weight = 0.0

    for criterion in criteria:
        if criterion.points < high_weight_threshold:
            continue

        total_high_weight += criterion.points
        is_satisfied = judge_fn(response, criterion.text)

        if not is_satisfied:
            omitted_weight += criterion.points
            result.critical_omissions.append(
                f"[{criterion.points}pts] {criterion.text}"
            )

    if total_high_weight > 0:
        result.omission_score = omitted_weight / total_high_weight

    if not has_image_input:
        for criterion in criteria:
            criterion_lower = criterion.text.lower()
            requires_visual = any(kw in criterion_lower for kw in VISUAL_EVIDENCE_KEYWORDS)
            if requires_visual and judge_fn(response, criterion.text):
                result.visual_evidence_flags.append(
                    f"Response satisfies visual criterion WITHOUT image input: {criterion.text}"
                )

    for criterion in criteria:
        criterion_lower = criterion.text.lower()
        is_safety = any(kw in criterion_lower for kw in SAFETY_CRITICAL_KEYWORDS)
        if is_safety and criterion.points >= 5 and not judge_fn(response, criterion.text):
            result.safety_gaps.append(
                f"[SAFETY] {criterion.text}"
            )

    result.passed = (
        len(result.visual_evidence_flags) == 0
        and len(result.safety_gaps) == 0
        and result.omission_score < 0.5
    )

    return result
