"""
PTB-XL dataset pipeline.

Dataset: https://physionet.org/content/ptb-xl/1.0.3/
21,837 12-lead ECGs with cardiologist annotations.
"""
import json
from pathlib import Path
from dataclasses import dataclass

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


OUTPUT_DIR = Path("data/processed_datasets/ptbxl_ecg_images")
METADATA_PATH = Path("data/raw_datasets/ptbxl/metadata.jsonl")


@dataclass
class PTBXLRecord:
    ecg_id: int
    patient_id: int
    image_path: str
    scp_codes: dict
    report: str
    age: int | None = None
    sex: int | None = None  # 0=male, 1=female
    source: str = "ptbxl"


def load_ptbxl_database(ptbxl_dir: str) -> list[dict]:
    if not HAS_PANDAS:
        return []

    db_path = Path(ptbxl_dir) / "ptbxl_database.csv"
    if not db_path.exists():
        print(f"PTB-XL database not found at {db_path}")
        return []

    df = pd.read_csv(db_path)
    return df.to_dict("records")


def convert_ptbxl_record(
    ptbxl_dir: str,
    record_row: dict,
    output_dir: Path = OUTPUT_DIR,
) -> PTBXLRecord | None:
    from src.data.mimic_ecg_pipeline import waveform_to_image

    ecg_id = record_row.get("ecg_id", 0)
    filename_hr = record_row.get("filename_hr", "")

    if not filename_hr:
        return None

    record_path = str(Path(ptbxl_dir) / filename_hr)
    image_path = output_dir / f"ptbxl_{ecg_id}.png"

    if not waveform_to_image(record_path, image_path):
        return None

    scp_codes = {}
    scp_raw = record_row.get("scp_codes", "{}")
    if isinstance(scp_raw, str):
        try:
            import ast
            scp_codes = ast.literal_eval(scp_raw)
        except (ValueError, SyntaxError):
            pass

    report = record_row.get("report", "") or ""
    report = report.strip() if isinstance(report, str) else ""

    return PTBXLRecord(
        ecg_id=ecg_id,
        patient_id=record_row.get("patient_id", 0),
        image_path=str(image_path),
        scp_codes=scp_codes,
        report=report,
        age=record_row.get("age"),
        sex=record_row.get("sex"),
    )


def process_ptbxl(
    ptbxl_dir: str,
    output_dir: Path = OUTPUT_DIR,
    max_records: int | None = None,
) -> list[PTBXLRecord]:
    rows = load_ptbxl_database(ptbxl_dir)
    if max_records:
        rows = rows[:max_records]

    records = []
    for row in rows:
        rec = convert_ptbxl_record(ptbxl_dir, row, output_dir)
        if rec is not None:
            records.append(rec)

    return records


def save_metadata(records: list[PTBXLRecord], output_path: Path = METADATA_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps({
                "ecg_id": rec.ecg_id,
                "patient_id": rec.patient_id,
                "image_path": rec.image_path,
                "scp_codes": rec.scp_codes,
                "report": rec.report,
                "age": rec.age,
                "sex": rec.sex,
                "source": rec.source,
            }) + "\n")
    print(f"Saved {len(records)} records to {output_path}")
