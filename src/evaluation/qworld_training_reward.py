from src.evaluation.qworld_integration import QworldResult, CriterionItem
from src.evaluation.qworld_evaluator import score_response_against_criteria, _keyword_judge


def qworld_reward(
    completions: list[str],
    criteria_sets: list[list[CriterionItem]],
    judge_fn=None,
) -> list[float]:
    """
    Qworld-based reward for GRPO training.

    Returns normalized score in [-1, 1] for each completion.
    Much richer signal than binary 1/0:
    - Partial credit for partially correct answers
    - Penalties for harmful recommendations (negative alpha)
    - Rewards for addressing safety, equity, follow-up
    """
    if judge_fn is None:
        judge_fn = _keyword_judge

    rewards = []
    for completion, criteria in zip(completions, criteria_sets):
        result = score_response_against_criteria(completion, criteria, judge_fn)
        rewards.append(result.overall_score)

    return rewards


def hybrid_reward(
    completions: list[str],
    answers: list[str],
    criteria_sets: list[list[CriterionItem]] | None = None,
    mcq_weight: float = 0.4,
    qworld_weight: float = 0.6,
    judge_fn=None,
) -> list[float]:
    """
    Combines binary MCQ correctness with Qworld criteria score.

    Use when both MCQ answer and Qworld criteria are available.
    Allows GRPO to optimize for both factual correctness AND
    comprehensive clinical quality simultaneously.
    """
    rewards = []
    for i, (completion, answer) in enumerate(zip(completions, answers)):
        completion_clean = completion.strip().upper()
        answer_clean = answer.strip().upper()
        mcq_score = 1.0 if answer_clean in completion_clean else 0.0

        qworld_score = 0.0
        if criteria_sets and i < len(criteria_sets) and criteria_sets[i]:
            if judge_fn is None:
                judge_fn = _keyword_judge
            result = score_response_against_criteria(completion, criteria_sets[i], judge_fn)
            qworld_score = result.overall_score

        combined = mcq_weight * mcq_score + qworld_weight * qworld_score
        rewards.append(combined)

    return rewards
