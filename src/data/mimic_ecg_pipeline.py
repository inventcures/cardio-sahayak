"""
MIMIC-IV-ECG data pipeline.

Requires PhysioNet credentials (CITI training).
Set PHYSIONET_USER and PHYSIONET_PASS environment variables.
Dataset: https://physionet.org/content/mimic-iv-ecg/1.0/
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


OUTPUT_DIR = Path("data/processed_datasets/mimic_ecg_images")
METADATA_PATH = Path("data/raw_datasets/mimic_iv_ecg/metadata.jsonl")

ECG_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


@dataclass
class ECGRecord:
    record_id: str
    image_path: str
    report: str
    source: str = "mimic_iv_ecg"


def waveform_to_image(record_path: str, output_path: Path, dpi: int = 150) -> bool:
    """Convert a WFDB record to a 12-lead ECG image in standard 4x3 grid layout."""
    if not HAS_WFDB or not HAS_MATPLOTLIB:
        return False

    try:
        record = wfdb.rdrecord(record_path)
    except Exception:
        return False

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    fig.patch.set_facecolor("white")

    for idx, (lead_name, ax) in enumerate(zip(ECG_LEADS, axes.flatten())):
        if idx < record.n_sig:
            signal = record.p_signal[:, idx]
            time = [i / record.fs for i in range(len(signal))]
            ax.plot(time, signal, "k-", linewidth=0.5)
        ax.set_title(lead_name, fontsize=8, fontweight="bold")
        ax.set_xlim(0, 2.5)
        ax.grid(True, alpha=0.3, color="red", linewidth=0.3)
        ax.tick_params(labelsize=5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def process_mimic_ecg_batch(
    record_paths: list[str],
    reports: list[str],
    output_dir: Path = OUTPUT_DIR,
    max_records: int | None = None,
) -> list[ECGRecord]:
    records = []
    pairs = list(zip(record_paths, reports))
    if max_records:
        pairs = pairs[:max_records]

    for record_path, report in pairs:
        record_id = Path(record_path).stem
        image_path = output_dir / f"{record_id}.png"

        if waveform_to_image(record_path, image_path):
            records.append(ECGRecord(
                record_id=record_id,
                image_path=str(image_path),
                report=report,
            ))

    return records


def save_metadata(records: list[ECGRecord], output_path: Path = METADATA_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps({
                "record_id": rec.record_id,
                "image_path": rec.image_path,
                "report": rec.report,
                "source": rec.source,
            }) + "\n")
    print(f"Saved {len(records)} records to {output_path}")
