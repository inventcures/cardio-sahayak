from pathlib import Path

from src.evaluation.cardioqa_india import load_benchmark, evaluate_model, print_results
from src.evaluation.cardioqa_india_generator import generate_full_benchmark
from src.knowledge.indian_guidelines import run_full_assessment
from src.knowledge.schemas import PatientProfile, Gender, ComorbidityProfile


BENCHMARK_PATH = Path("data/benchmarks/cardioqa_india_v1.jsonl")


def ensure_benchmark_exists():
    if not BENCHMARK_PATH.exists():
        print("Generating CardioQA-India benchmark...")
        generate_full_benchmark(BENCHMARK_PATH)


def run_knowledge_engine_eval():
    print("\n" + "=" * 60)
    print("Knowledge Engine Self-Test")
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


def run_benchmark_eval(predict_fn=None):
    ensure_benchmark_exists()
    questions = load_benchmark(BENCHMARK_PATH)

    if not questions:
        print("No benchmark questions found.")
        return

    if predict_fn is None:
        print(f"\nNo model predict function provided. Skipping model evaluation.")
        print(f"Benchmark has {len(questions)} questions ready for evaluation.")
        return

    result = evaluate_model(predict_fn, questions)
    print_results(result)
    return result


def run_all():
    print("Cardio-Sahayak Next-Gen: Full Evaluation Suite")
    print("=" * 60)

    ke_ok = run_knowledge_engine_eval()
    run_benchmark_eval()

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Knowledge Engine Self-Test: {'PASS' if ke_ok else 'FAIL'}")
    print(f"CardioQA-India Benchmark: Ready ({BENCHMARK_PATH})")
    print("Model evaluation: requires predict_fn (GPU/GGUF model)")


if __name__ == "__main__":
    run_all()
