"""
Mirage detection protocol adapted from MARCUS (Stanford/UCSF 2026).

Layer 1: Counterfactual image-absent probe -- run inference with and without image,
         flag if outputs are too similar (model answering from language priors, not visual evidence).
Layer 2: 3-rephrasing consistency check -- rephrase question 3 ways, check consistency.
"""
from dataclasses import dataclass


@dataclass
class MirageResult:
    mirage_detected: bool
    image_present_text: str
    image_absent_text: str
    similarity_score: float
    consistency_score: float
    rephrased_responses: list[str]


SIMILARITY_THRESHOLD = 0.85
CONSISTENCY_THRESHOLD = 0.6


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def check_mirage(
    image_present_text: str,
    image_absent_text: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[bool, float]:
    similarity = compute_jaccard_similarity(image_present_text, image_absent_text)
    is_mirage = similarity > threshold
    return is_mirage, similarity


def check_consistency(
    responses: list[str],
    threshold: float = CONSISTENCY_THRESHOLD,
) -> tuple[bool, float]:
    if len(responses) < 2:
        return True, 1.0

    similarities = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            sim = compute_jaccard_similarity(responses[i], responses[j])
            similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    is_consistent = avg_similarity >= threshold
    return is_consistent, avg_similarity


def generate_rephrasings(question: str) -> list[str]:
    rephrasings = [
        question,
        f"Based on the provided data, {question.lower().rstrip('?.')}?",
        f"Please assess: {question.lower().rstrip('?.')}.",
    ]
    return rephrasings


def run_mirage_detection(
    expert_fn,
    data_with_image: dict,
    data_without_image: dict,
    question: str,
) -> MirageResult:
    image_present = expert_fn(data_with_image)
    image_absent = expert_fn(data_without_image)

    is_mirage, similarity = check_mirage(
        image_present.raw_text, image_absent.raw_text
    )

    rephrasings = generate_rephrasings(question)
    rephrased_responses = []
    for rephrase in rephrasings:
        data_rephrased = {**data_with_image, "clinical_context": rephrase}
        response = expert_fn(data_rephrased)
        rephrased_responses.append(response.raw_text)

    is_consistent, consistency = check_consistency(rephrased_responses)

    return MirageResult(
        mirage_detected=is_mirage or not is_consistent,
        image_present_text=image_present.raw_text,
        image_absent_text=image_absent.raw_text,
        similarity_score=similarity,
        consistency_score=consistency,
        rephrased_responses=rephrased_responses,
    )
