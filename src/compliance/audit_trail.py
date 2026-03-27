import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict


AUDIT_DIR = Path("data/audit_logs")


@dataclass
class AuditEntry:
    timestamp: str = ""
    session_id: str = ""
    patient_id: str = ""
    action: str = ""
    risk_category: str = ""
    modalities_used: list[str] = field(default_factory=list)
    mirage_detected: bool = False
    cross_check_passed: bool = True
    contradictions: list[str] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def write_audit_entry(entry: AuditEntry):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = AUDIT_DIR / f"audit_{date_str}.jsonl"

    with open(log_path, "a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")

    return log_path


def read_audit_log(date_str: str) -> list[AuditEntry]:
    log_path = AUDIT_DIR / f"audit_{date_str}.jsonl"
    if not log_path.exists():
        return []

    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                entries.append(AuditEntry(**data))
    return entries
