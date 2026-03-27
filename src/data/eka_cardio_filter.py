"""
Filter EkaCare clinical notes dataset for cardiology-relevant records.

The original dataset (ekacare/clinical_note_generation_dataset) contains
general clinical notes. Many are non-cardiology (appendicitis, paracetamol Rx).
This filter identifies cardiology-relevant records using keyword matching.
"""
import json
import re
from pathlib import Path
from dataclasses import dataclass


CARDIOLOGY_KEYWORDS = [
    r"\bheart\b", r"\bcardiac\b", r"\bcardio", r"\bchest pain\b",
    r"\bangina\b", r"\bmyocardial\b", r"\binfarction\b", r"\bmi\b",
    r"\bstemi\b", r"\bnstemi\b", r"\bacs\b",
    r"\becg\b", r"\bekg\b", r"\belectrocardiog",
    r"\becho\b", r"\bechocardiog",
    r"\bhypertension\b", r"\bhtn\b", r"\bbp\b",
    r"\bblood pressure\b", r"\bhigh bp\b",
    r"\bcholesterol\b", r"\blipid\b", r"\bldl\b", r"\bhdl\b",
    r"\bstatin\b", r"\batorvastatin\b", r"\brosuvastatin\b",
    r"\baspirin\b", r"\bclopidogrel\b", r"\bticagrelor\b",
    r"\bbeta.?blocker\b", r"\bmetoprolol\b", r"\bbisoprolol\b",
    r"\bamlodipine\b", r"\bramipril\b", r"\benalapril\b",
    r"\barrhythmia\b", r"\batrial fibrillation\b", r"\bafib\b", r"\baf\b",
    r"\bpalpitation\b", r"\btachycardia\b", r"\bbradycardia\b",
    r"\bheart failure\b", r"\bhf\b", r"\blvef\b", r"\bejection fraction\b",
    r"\bcoronary\b", r"\bcad\b", r"\bihd\b", r"\bischem",
    r"\bvalv", r"\bmitral\b", r"\baortic\b", r"\btricuspid\b",
    r"\bpericardi", r"\bcardiomyopathy\b", r"\bhcm\b", r"\bdcm\b",
    r"\bstent\b", r"\bangioplasty\b", r"\bcabg\b", r"\bpci\b",
    r"\btroponin\b", r"\bbnp\b", r"\bnt.?probnp\b",
    r"\bdiabetes\b", r"\bdiabetic\b", r"\bhba1c\b", r"\bsglt2\b",
    r"\bdyspnea\b", r"\bdyspnoea\b", r"\bbreathless",
    r"\bsyncope\b", r"\bedema\b", r"\boedema\b",
]

COMPILED_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in CARDIOLOGY_KEYWORDS]

MIN_KEYWORD_MATCHES = 2


@dataclass
class FilteredRecord:
    original_text: str
    instruction: str
    output: str
    keyword_matches: list[str]
    match_count: int
    source: str = "ekacare_filtered"


def is_cardiology_relevant(text: str, min_matches: int = MIN_KEYWORD_MATCHES) -> tuple[bool, list[str]]:
    matches = []
    for pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)

    return len(matches) >= min_matches, matches


def filter_eka_notes(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> list[FilteredRecord]:
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return []

    filtered = []
    total = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get("instruction", "") + " " + record.get("output", "")
            relevant, matches = is_cardiology_relevant(text)

            if relevant:
                filtered.append(FilteredRecord(
                    original_text=text,
                    instruction=record.get("instruction", ""),
                    output=record.get("output", ""),
                    keyword_matches=matches,
                    match_count=len(matches),
                ))

    print(f"Filtered {len(filtered)} cardiology-relevant records from {total} total")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for rec in filtered:
                f.write(json.dumps({
                    "instruction": rec.instruction,
                    "output": rec.output,
                    "keyword_matches": rec.keyword_matches,
                    "match_count": rec.match_count,
                    "source": rec.source,
                }) + "\n")
        print(f"Saved to {output_path}")

    return filtered
