import json
from pathlib import Path
from dataclasses import dataclass, field


BENCHMARK_PATH = Path("data/benchmarks/cardioqa_india_v1.jsonl")


@dataclass
class MCQuestion:
    question: str
    options: list[str]
    correct_answer: str
    category: str = ""
    difficulty: str = "medium"
    guideline_source: str = ""


@dataclass
class BenchmarkResult:
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    by_category: dict[str, dict] = field(default_factory=dict)


def load_benchmark(path: str | Path = BENCHMARK_PATH) -> list[MCQuestion]:
    path = Path(path)
    if not path.exists():
        return []

    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            questions.append(MCQuestion(
                question=data["question"],
                options=data["options"],
                correct_answer=data["correct_answer"],
                category=data.get("category", ""),
                difficulty=data.get("difficulty", "medium"),
                guideline_source=data.get("guideline_source", ""),
            ))
    return questions


def evaluate_model(
    predict_fn,
    questions: list[MCQuestion],
) -> BenchmarkResult:
    result = BenchmarkResult(total=len(questions))

    for q in questions:
        formatted = f"{q.question}\n\n"
        for i, opt in enumerate(q.options):
            formatted += f"{chr(65+i)}. {opt}\n"
        formatted += "\nAnswer with just the letter:"

        prediction = predict_fn(formatted).strip().upper()
        correct = q.correct_answer.strip().upper()

        is_correct = prediction.startswith(correct)
        if is_correct:
            result.correct += 1

        cat = q.category or "uncategorized"
        if cat not in result.by_category:
            result.by_category[cat] = {"total": 0, "correct": 0}
        result.by_category[cat]["total"] += 1
        if is_correct:
            result.by_category[cat]["correct"] += 1

    result.accuracy = result.correct / result.total if result.total > 0 else 0.0

    for cat_data in result.by_category.values():
        cat_data["accuracy"] = (
            cat_data["correct"] / cat_data["total"]
            if cat_data["total"] > 0 else 0.0
        )

    return result


def print_results(result: BenchmarkResult):
    print(f"\nCardioQA-India Benchmark Results")
    print(f"{'='*50}")
    print(f"Overall: {result.correct}/{result.total} = {result.accuracy:.1%}")
    print(f"\nBy Category:")
    for cat, data in sorted(result.by_category.items()):
        print(f"  {cat}: {data['correct']}/{data['total']} = {data['accuracy']:.1%}")
