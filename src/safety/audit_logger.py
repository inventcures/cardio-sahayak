import json
import os
from datetime import datetime
from pathlib import Path


AUDIT_LOG_DIR = Path("data/audit_logs")


def log_inference(
    session_id: str,
    patient_id: str,
    input_data: dict,
    expert_reports: list[dict],
    guideline_assessment: dict,
    mirage_results: dict | None,
    cross_check_results: dict | None,
    final_output: dict,
    model_versions: dict | None = None,
):
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "patient_id": patient_id,
        "input_modalities": list(input_data.keys()),
        "expert_reports_count": len(expert_reports),
        "expert_modalities": [r.get("modality", "unknown") for r in expert_reports],
        "risk_category": guideline_assessment.get("risk_assessment", {}).get("risk_category", "UNKNOWN"),
        "mirage_detected": mirage_results.get("mirage_detected", False) if mirage_results else False,
        "cross_check_passed": cross_check_results.get("passed", True) if cross_check_results else True,
        "cross_check_contradictions": cross_check_results.get("contradictions", []) if cross_check_results else [],
        "model_versions": model_versions or {},
        "final_output_keys": list(final_output.keys()),
    }

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = AUDIT_LOG_DIR / f"audit_{date_str}.jsonl"

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_path
