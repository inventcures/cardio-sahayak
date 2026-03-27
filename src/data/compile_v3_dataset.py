"""
V3 dataset compiler.

Aggregates:
- EkaCare filtered cardiology notes
- Synthetic Indian clinical vignettes
- ECG-report pairs (MIMIC-IV, PTB-XL)
- Echo-report pairs (EchoNet-Dynamic)
- Existing V2 dataset (quality-filtered)

Runs quality gate on all records before output.
"""
import json
from pathlib import Path

from src.data.quality_gate import run_quality_gate, save_quality_report

DATA_DIR = Path("data")
V2_DATASET = DATA_DIR / "processed_datasets" / "cardio_sahayak_india_instruct_v2.jsonl"
EKA_FILTERED = DATA_DIR / "processed_datasets" / "eka_cardio_filtered.jsonl"
SYNTHETIC_VIGNETTES = DATA_DIR / "processed_datasets" / "synthetic_indian_vignettes_v3.jsonl"
MIMIC_METADATA = DATA_DIR / "raw_datasets" / "mimic_iv_ecg" / "metadata.jsonl"
PTBXL_METADATA = DATA_DIR / "raw_datasets" / "ptbxl" / "metadata.jsonl"
ECHONET_METADATA = DATA_DIR / "raw_datasets" / "echonet" / "metadata.jsonl"

V3_OUTPUT = DATA_DIR / "processed_datasets" / "cardio_sahayak_india_instruct_v3.jsonl"
QUALITY_REPORT = DATA_DIR / "processed_datasets" / "v3_quality_report.jsonl"


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def format_ecg_record(record: dict) -> dict:
    report = record.get("report", "")
    if not report:
        return {}
    return {
        "instruction": (
            "Analyze this 12-lead ECG and provide a comprehensive clinical interpretation. "
            "Include heart rate, rhythm, intervals, axis, ST-T changes, and clinical impression. "
            "Apply South Asian cardiovascular risk context."
        ),
        "output": report,
        "source": record.get("source", "ecg"),
        "reference_image": record.get("image_path", ""),
    }


def format_echo_record(record: dict) -> dict:
    lvef = record.get("lvef")
    if lvef is None:
        return {}
    return {
        "instruction": (
            "Analyze this echocardiogram and provide a structured assessment. "
            "Include LVEF estimation, wall motion abnormalities, valvular findings, "
            "and diastolic function. Apply Indian Consensus treatment recommendations."
        ),
        "output": (
            f"Left ventricular ejection fraction (LVEF): {lvef:.0f}%. "
            f"{'Normal systolic function.' if lvef >= 50 else 'Reduced systolic function.' if lvef < 40 else 'Mildly reduced systolic function.'}"
        ),
        "source": "echonet_dynamic",
        "reference_frames": record.get("frame_paths", []),
    }


def compile_v3_dataset() -> Path:
    all_records: list[dict] = []

    v2_records = load_jsonl(V2_DATASET)
    for r in v2_records:
        r.setdefault("source", "v2_dataset")
    all_records.extend(v2_records)

    eka_records = load_jsonl(EKA_FILTERED)
    all_records.extend(eka_records)

    synthetic_records = load_jsonl(SYNTHETIC_VIGNETTES)
    all_records.extend(synthetic_records)

    for metadata_path in [MIMIC_METADATA, PTBXL_METADATA]:
        ecg_raw = load_jsonl(metadata_path)
        for r in ecg_raw:
            formatted = format_ecg_record(r)
            if formatted:
                all_records.append(formatted)

    echo_raw = load_jsonl(ECHONET_METADATA)
    for r in echo_raw:
        formatted = format_echo_record(r)
        if formatted:
            all_records.append(formatted)

    print(f"Total records before quality gate: {len(all_records)}")

    passed, rejected = run_quality_gate(all_records, deduplicate=True)

    V3_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(V3_OUTPUT, "w") as f:
        for record in passed:
            f.write(json.dumps(record) + "\n")

    save_quality_report(rejected, QUALITY_REPORT)

    print(f"V3 dataset compiled: {len(passed)} records saved to {V3_OUTPUT}")
    return V3_OUTPUT


if __name__ == "__main__":
    compile_v3_dataset()
