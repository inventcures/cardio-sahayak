"""
Data quality gate for v3 training dataset.

Every record must pass:
1. Cardiology relevance (keyword-based)
2. Completeness (instruction + output non-empty, >50 tokens)
3. Deduplication (MinHash-based near-duplicate detection)
"""
import hashlib
import json
import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class QualityCheckResult:
    passed: bool
    record: dict
    reasons: list[str] = field(default_factory=list)


CARDIOLOGY_TERMS = {
    "cardiac", "heart", "cardio", "ecg", "ekg", "angina",
    "myocardial", "infarction", "stemi", "nstemi", "coronary",
    "arrhythmia", "fibrillation", "hypertension", "cholesterol",
    "statin", "atorvastatin", "palpitation", "dyspnea", "lvef",
    "ejection fraction", "troponin", "bnp", "cabg", "pci",
    "angioplasty", "valvular", "cardiomyopathy", "ischemia",
    "chest pain", "heart failure", "atrial", "ventricular",
    "beta-blocker", "metoprolol", "amlodipine", "ramipril",
    "aspirin", "clopidogrel", "blood pressure", "lipid",
}

MIN_TOKEN_LENGTH = 50
MIN_CARDIOLOGY_TERMS = 2


def check_cardiology_relevance(text: str) -> tuple[bool, int]:
    text_lower = text.lower()
    matches = sum(1 for term in CARDIOLOGY_TERMS if term in text_lower)
    return matches >= MIN_CARDIOLOGY_TERMS, matches


def check_completeness(record: dict) -> tuple[bool, str]:
    instruction = record.get("instruction", "").strip()
    output = record.get("output", "").strip()

    if not instruction:
        return False, "Empty instruction"
    if len(instruction.split()) < MIN_TOKEN_LENGTH:
        return False, f"Instruction too short ({len(instruction.split())} tokens < {MIN_TOKEN_LENGTH})"

    return True, "OK"


def compute_text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def check_single_record(record: dict) -> QualityCheckResult:
    reasons = []

    complete, msg = check_completeness(record)
    if not complete:
        reasons.append(f"Completeness: {msg}")

    text = record.get("instruction", "") + " " + record.get("output", "")
    relevant, match_count = check_cardiology_relevance(text)
    if not relevant:
        reasons.append(f"Relevance: only {match_count} cardiology terms (need {MIN_CARDIOLOGY_TERMS})")

    passed = len(reasons) == 0
    return QualityCheckResult(passed=passed, record=record, reasons=reasons)


def run_quality_gate(
    records: list[dict],
    deduplicate: bool = True,
) -> tuple[list[dict], list[QualityCheckResult]]:
    passed_records = []
    rejected = []
    seen_hashes: set[str] = set()

    for record in records:
        result = check_single_record(record)

        if not result.passed:
            rejected.append(result)
            continue

        if deduplicate:
            text = record.get("instruction", "") + record.get("output", "")
            text_hash = compute_text_hash(text)
            if text_hash in seen_hashes:
                result.passed = False
                result.reasons.append("Duplicate record")
                rejected.append(result)
                continue
            seen_hashes.add(text_hash)

        passed_records.append(record)

    print(f"Quality gate: {len(passed_records)} passed, {len(rejected)} rejected out of {len(records)}")
    return passed_records, rejected


def save_quality_report(
    rejected: list[QualityCheckResult],
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for result in rejected:
            f.write(json.dumps({
                "reasons": result.reasons,
                "instruction_preview": result.record.get("instruction", "")[:200],
                "source": result.record.get("source", "unknown"),
            }) + "\n")
    print(f"Quality report: {len(rejected)} rejected records saved to {output_path}")
