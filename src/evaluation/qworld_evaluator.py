import json
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation.qworld_integration import QworldResult, CriterionItem


EVALUATION_DIMENSIONS = [
    "diagnostic_accuracy",
    "safety_risk_management",
    "evidence_quality",
    "guideline_adherence",
    "completeness",
    "emergency_recognition",
    "followup_continuity",
    "clarity_communication",
    "empathy_support",
    "patient_empowerment",
    "caregiver_support",
    "health_literacy",
    "cultural_sensitivity",
    "south_asian_phenotype",
    "indian_guideline_compliance",
    "nlem_drug_availability",
    "resource_stratification",
    "diabetes_cardiac_integration",
    "hindi_language_quality",
    "chw_actionability",
    "referral_appropriateness",
    "cost_sensitivity",
    "family_centric_care",
    "tobacco_context",
    "dietary_context",
]


@dataclass
class CriterionScore:
    criterion: str
    points: float
    satisfied: bool
    score: float
    reasoning: str = ""


@dataclass
class QworldEvalResult:
    question: str
    response: str
    criteria_scores: list[CriterionScore] = field(default_factory=list)
    overall_score: float = 0.0
    dimension_scores: dict[str, float] = field(default_factory=dict)
    satisfied_count: int = 0
    total_count: int = 0
    critical_omissions: list[str] = field(default_factory=list)
    penalty_triggers: list[str] = field(default_factory=list)


def score_response_against_criteria(
    response: str,
    criteria: list[CriterionItem],
    judge_fn=None,
) -> QworldEvalResult:
    if judge_fn is None:
        judge_fn = _keyword_judge

    scores = []
    total_positive = 0.0
    earned = 0.0
    satisfied = 0
    omissions = []
    penalties = []

    for criterion in criteria:
        is_satisfied = judge_fn(response, criterion.text)

        if criterion.points > 0:
            total_positive += criterion.points
            if is_satisfied:
                earned += criterion.points
                satisfied += 1
            elif criterion.points >= 5:
                omissions.append(f"[{criterion.points}pts] {criterion.text}")
        else:
            if is_satisfied:
                earned += criterion.points
                penalties.append(f"[{criterion.points}pts] {criterion.text}")

        scores.append(CriterionScore(
            criterion=criterion.text,
            points=criterion.points,
            satisfied=is_satisfied,
            score=criterion.points if is_satisfied else 0.0,
        ))

    overall = earned / total_positive if total_positive > 0 else 0.0

    return QworldEvalResult(
        question="",
        response=response,
        criteria_scores=scores,
        overall_score=overall,
        satisfied_count=satisfied,
        total_count=len(criteria),
        critical_omissions=omissions,
        penalty_triggers=penalties,
    )


STOPWORDS = {
    "should", "could", "would", "about", "their", "there",
    "which", "where", "these", "those", "other", "after",
    "before", "between", "through", "during", "against",
    "without", "within", "provide", "include", "mention",
    "address", "assess", "consider", "recommend", "ensure",
}


def _keyword_judge(response: str, criterion: str) -> bool:
    response_lower = response.lower()
    criterion_lower = criterion.lower()

    key_terms = [
        word for word in criterion_lower.split()
        if len(word) > 4 and word not in STOPWORDS
    ]

    if not key_terms:
        return False

    matched = sum(1 for term in key_terms if term in response_lower)
    return matched >= max(1, len(key_terms) // 3)


def evaluate_model_responses(
    responses: list[dict],
    criteria_path: str | Path,
    judge_fn=None,
) -> list[QworldEvalResult]:
    criteria_path = Path(criteria_path)
    criteria_by_question: dict[str, list[CriterionItem]] = {}

    if criteria_path.exists():
        with open(criteria_path) as f:
            for line in f:
                data = json.loads(line.strip())
                q = data["question"]
                criteria_by_question[q] = [
                    CriterionItem(text=c["text"], points=c["points"], dimension=c.get("dimension", ""))
                    for c in data["criteria"]
                ]

    results = []
    for resp in responses:
        question = resp.get("question", "")
        response_text = resp.get("response", resp.get("output", ""))
        criteria = criteria_by_question.get(question, [])

        if not criteria:
            continue

        result = score_response_against_criteria(response_text, criteria, judge_fn)
        result.question = question
        results.append(result)

    return results


def compute_aggregate_scores(results: list[QworldEvalResult]) -> dict:
    if not results:
        return {}

    scores = [r.overall_score for r in results]
    total_omissions = sum(len(r.critical_omissions) for r in results)
    total_penalties = sum(len(r.penalty_triggers) for r in results)

    return {
        "num_questions": len(results),
        "mean_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "total_critical_omissions": total_omissions,
        "total_penalty_triggers": total_penalties,
        "avg_criteria_satisfied": sum(r.satisfied_count for r in results) / len(results),
        "avg_criteria_total": sum(r.total_count for r in results) / len(results),
    }


def print_eval_report(results: list[QworldEvalResult]):
    agg = compute_aggregate_scores(results)
    if not agg:
        print("No evaluation results.")
        return

    print(f"\nQworld Evaluation Report")
    print(f"{'=' * 60}")
    print(f"Questions evaluated: {agg['num_questions']}")
    print(f"Mean score: {agg['mean_score']:.1%}")
    print(f"Score range: [{agg['min_score']:.1%}, {agg['max_score']:.1%}]")
    print(f"Avg criteria satisfied: {agg['avg_criteria_satisfied']:.1f} / {agg['avg_criteria_total']:.1f}")
    print(f"Critical omissions: {agg['total_critical_omissions']}")
    print(f"Penalty triggers: {agg['total_penalty_triggers']}")
