from pathlib import Path

from src.evaluation.cardioqa_india import load_benchmark, evaluate_model, print_results
from src.evaluation.cardioqa_india_generator import generate_full_benchmark
from src.evaluation.qworld_integration import generate_criteria
from src.evaluation.qworld_evaluator import score_response_against_criteria, compute_aggregate_scores
from src.evaluation.cardioqa_v2_generator import generate_cardioqa_v2
from src.knowledge.indian_guidelines import run_full_assessment
from src.knowledge.schemas import PatientProfile, Gender, ComorbidityProfile


BENCHMARK_V1 = Path("data/benchmarks/cardioqa_india_v1.jsonl")
BENCHMARK_V2 = Path("data/benchmarks/cardioqa_india_v2_questions.jsonl")


def ensure_benchmarks_exist():
    if not BENCHMARK_V1.exists():
        generate_full_benchmark(BENCHMARK_V1)
    if not BENCHMARK_V2.exists():
        generate_cardioqa_v2(BENCHMARK_V2)


def run_knowledge_engine_eval():
    print("\n" + "=" * 60)
    print("Tier 1: Knowledge Engine Self-Test")
    print("=" * 60)

    test_cases = [
        {
            "name": "High-risk diabetic Indian male",
            "patient": PatientProfile(
                age=52, gender=Gender.MALE, bmi=25.5,
                has_diabetes=True, ldl_mg_dl=155, hba1c=8.5,
                heart_rate_bpm=82, bp_systolic=148, bp_diastolic=92,
                lvef_percent=38, bnp_elevated=True,
                comorbidities=ComorbidityProfile(
                    diabetes_mellitus=True, hypertension=True, current_smoker=True,
                ),
            ),
            "expected_risk": "HIGH",
        },
        {
            "name": "Low-risk young female",
            "patient": PatientProfile(
                age=35, gender=Gender.FEMALE, bmi=21.0,
                lvef_percent=65,
            ),
            "expected_risk": "LOW",
        },
        {
            "name": "Intermediate-risk elderly with HTN",
            "patient": PatientProfile(
                age=65, gender=Gender.MALE,
                bp_systolic=145, bp_diastolic=88,
                lvef_percent=48,
                comorbidities=ComorbidityProfile(
                    hypertension=True, cholesterol_gt_250=True,
                ),
            ),
            "expected_risk": "INTERMEDIATE",
        },
    ]

    passed = 0
    for case in test_cases:
        result = run_full_assessment(case["patient"])
        actual_risk = result["risk_assessment"].risk_category.value
        ok = actual_risk == case["expected_risk"]
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {case['name']} -> {actual_risk} (expected {case['expected_risk']})")
        if ok:
            passed += 1

    print(f"\nKnowledge Engine: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def run_mcq_eval(predict_fn=None):
    print("\n" + "=" * 60)
    print("Tier 2: CardioQA-India v1 (MCQ)")
    print("=" * 60)

    questions = load_benchmark(BENCHMARK_V1)
    if not questions:
        print("No benchmark questions found.")
        return

    if predict_fn is None:
        print(f"Benchmark has {len(questions)} MCQs ready. Provide predict_fn to evaluate.")
        return

    result = evaluate_model(predict_fn, questions)
    print_results(result)
    return result


def run_qworld_eval(predict_fn=None, sample_size: int = 5):
    print("\n" + "=" * 60)
    print("Tier 3: Qworld Question-Specific Evaluation (CardioQA v2)")
    print("=" * 60)

    import json
    if not BENCHMARK_V2.exists():
        generate_cardioqa_v2(BENCHMARK_V2)

    questions = []
    with open(BENCHMARK_V2) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line.strip()))

    print(f"CardioQA v2: {len(questions)} open-ended questions available")

    sample = questions[:sample_size]
    print(f"Running Qworld evaluation on {len(sample)} sample questions (fallback criteria)")

    eval_results = []
    for q in sample:
        qw = generate_criteria(q["question"])
        print(f"  Q: {q['question'][:80]}...")
        print(f"  Criteria: {len(qw.criteria)} (pos={qw.total_positive_points}, neg={qw.total_negative_points})")

        if predict_fn is not None:
            response = predict_fn(q["question"])
            result = score_response_against_criteria(response, qw.criteria)
            print(f"  Score: {result.overall_score:.1%} ({result.satisfied_count}/{result.total_count} criteria)")
            eval_results.append(result)

    if eval_results:
        agg = compute_aggregate_scores(eval_results)
        print(f"\nAggregate: mean={agg['mean_score']:.1%}, omissions={agg['total_critical_omissions']}")

    return eval_results


def run_all(predict_fn=None):
    print("Cardio-Sahayak Next-Gen: Full Evaluation Suite (v5)")
    print("=" * 60)

    ensure_benchmarks_exist()
    ke_ok = run_knowledge_engine_eval()
    run_mcq_eval(predict_fn)
    run_qworld_eval(predict_fn)

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Tier 1 Knowledge Engine: {'PASS' if ke_ok else 'FAIL'}")
    print(f"Tier 2 CardioQA v1 MCQ: {BENCHMARK_V1} ({load_benchmark(BENCHMARK_V1).__len__()} questions)")
    print(f"Tier 3 CardioQA v2 Qworld: {BENCHMARK_V2} (200 questions, ~45 criteria each)")
    print("Tier 4 Cardiologist Review: requires clinical study")
    if predict_fn is None:
        print("\nModel evaluation skipped (no predict_fn). Provide predict_fn to evaluate.")


if __name__ == "__main__":
    run_all()
