from dataclasses import dataclass, field


@dataclass
class V3Record:
    instruction: str
    output: str
    source: str
    reference_image: str = ""
    reference_frames: list[str] = field(default_factory=list)
    condition: str = ""
    comorbidities: list[str] = field(default_factory=list)
    age: int | None = None
    gender: str = ""
    bmi: float | None = None
    south_asian_flags: list[str] = field(default_factory=list)


@dataclass
class DatasetStats:
    total_records: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    by_modality: dict[str, int] = field(default_factory=dict)
    quality_passed: int = 0
    quality_rejected: int = 0


def compute_dataset_stats(records: list[dict]) -> DatasetStats:
    stats = DatasetStats(total_records=len(records))

    for record in records:
        source = record.get("source", "unknown")
        stats.by_source[source] = stats.by_source.get(source, 0) + 1

        if record.get("reference_image"):
            modality = "ecg_image"
        elif record.get("reference_frames"):
            modality = "echo_video"
        else:
            modality = "text"
        stats.by_modality[modality] = stats.by_modality.get(modality, 0) + 1

    return stats
