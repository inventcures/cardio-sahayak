"""
EchoNet-Dynamic dataset pipeline.

Dataset: https://echonet.github.io/dynamic/
10,030 apical-4-chamber echo videos with LVEF labels.
Requires download from Stanford after license agreement.
"""
import json
from pathlib import Path
from dataclasses import dataclass

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


OUTPUT_DIR = Path("data/processed_datasets/echonet_frames")
METADATA_PATH = Path("data/raw_datasets/echonet/metadata.jsonl")


@dataclass
class EchoRecord:
    filename: str
    lvef: float
    num_frames: int
    frame_paths: list[str]
    esv: float | None = None
    edv: float | None = None
    split: str = "train"
    source: str = "echonet_dynamic"


def extract_key_frames(
    video_path: str,
    output_dir: Path,
    num_frames: int = 8,
) -> list[str]:
    if not HAS_CV2:
        return []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frame_paths = []
    video_stem = Path(video_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = output_dir / f"{video_stem}_frame{idx:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))

    cap.release()
    return frame_paths


def load_echonet_filelist(echonet_dir: str) -> list[dict]:
    if not HAS_PANDAS:
        return []

    filelist_path = Path(echonet_dir) / "FileList.csv"
    if not filelist_path.exists():
        print(f"EchoNet FileList.csv not found at {filelist_path}")
        return []

    df = pd.read_csv(filelist_path)
    return df.to_dict("records")


def process_echonet(
    echonet_dir: str,
    output_dir: Path = OUTPUT_DIR,
    num_frames: int = 8,
    max_records: int | None = None,
) -> list[EchoRecord]:
    rows = load_echonet_filelist(echonet_dir)
    if max_records:
        rows = rows[:max_records]

    records = []
    videos_dir = Path(echonet_dir) / "Videos"

    for row in rows:
        filename = row.get("FileName", "")
        video_path = videos_dir / filename
        if not video_path.exists():
            continue

        video_output_dir = output_dir / Path(filename).stem
        frame_paths = extract_key_frames(str(video_path), video_output_dir, num_frames)

        if frame_paths:
            records.append(EchoRecord(
                filename=filename,
                lvef=float(row.get("EF", 0)),
                num_frames=len(frame_paths),
                frame_paths=frame_paths,
                esv=row.get("ESV"),
                edv=row.get("EDV"),
                split=row.get("Split", "train"),
            ))

    return records


def save_metadata(records: list[EchoRecord], output_path: Path = METADATA_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps({
                "filename": rec.filename,
                "lvef": rec.lvef,
                "num_frames": rec.num_frames,
                "frame_paths": rec.frame_paths,
                "esv": rec.esv,
                "edv": rec.edv,
                "split": rec.split,
                "source": rec.source,
            }) + "\n")
    print(f"Saved {len(records)} records to {output_path}")
