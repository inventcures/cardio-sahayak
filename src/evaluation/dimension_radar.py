import json
import math
from pathlib import Path
from dataclasses import dataclass, field

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


DIMENSIONS = [
    ("Diagnostic\nAccuracy", "diagnostic_accuracy"),
    ("Safety &\nRisk Mgmt", "safety_risk_management"),
    ("Evidence\nQuality", "evidence_quality"),
    ("Guideline\nAdherence", "guideline_adherence"),
    ("Completeness", "completeness"),
    ("Emergency\nRecognition", "emergency_recognition"),
    ("Follow-Up\nContinuity", "followup_continuity"),
    ("Clarity", "clarity_communication"),
    ("Empathy\n& Support", "empathy_support"),
    ("Patient\nEmpowerment", "patient_empowerment"),
    ("Caregiver\nSupport", "caregiver_support"),
    ("Health\nLiteracy", "health_literacy"),
    ("Cultural\nSensitivity", "cultural_sensitivity"),
    ("SA Phenotype\nAwareness", "south_asian_phenotype"),
    ("Indian\nGuidelines", "indian_guideline_compliance"),
    ("NLEM\nAvailability", "nlem_drug_availability"),
    ("Resource\nStratification", "resource_stratification"),
    ("Diabetes-\nCardiac", "diabetes_cardiac_integration"),
    ("Hindi\nQuality", "hindi_language_quality"),
    ("CHW\nActionability", "chw_actionability"),
    ("Referral\nAppropriate", "referral_appropriateness"),
    ("Cost\nSensitivity", "cost_sensitivity"),
    ("Family\nCentric", "family_centric_care"),
    ("Tobacco\nContext", "tobacco_context"),
    ("Dietary\nContext", "dietary_context"),
]


@dataclass
class DimensionScore:
    dimension: str
    label: str
    score: float = 0.0
    count: int = 0


def compute_dimension_scores(
    eval_results: list[dict],
) -> dict[str, float]:
    dim_totals: dict[str, list[float]] = {}

    for result in eval_results:
        criteria_scores = result.get("criteria_scores", [])
        for cs in criteria_scores:
            dim = cs.get("dimension", "general")
            if dim not in dim_totals:
                dim_totals[dim] = []
            satisfied = 1.0 if cs.get("satisfied", False) else 0.0
            dim_totals[dim].append(satisfied)

    return {
        dim: sum(scores) / len(scores) if scores else 0.0
        for dim, scores in dim_totals.items()
    }


def create_radar_chart(
    dimension_scores: dict[str, float],
    title: str = "Cardio-Sahayak Evaluation",
    output_path: str | Path = "out/radar_chart.png",
    model_name: str = "Cardio-Sahayak v3",
) -> str | None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Skipping radar chart.")
        return None

    labels = [d[0] for d in DIMENSIONS]
    keys = [d[1] for d in DIMENSIONS]
    values = [dimension_scores.get(k, 0.0) for k in keys]

    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

    ax.plot(angles, values, "o-", linewidth=2, label=model_name, color="#2196F3")
    ax.fill(angles, values, alpha=0.15, color="#2196F3")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=7)
    ax.set_title(title, size=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Radar chart saved to {output_path}")
    return str(output_path)


def create_comparison_radar(
    scores_list: list[tuple[str, dict[str, float]]],
    title: str = "Cardio-Sahayak Model Comparison",
    output_path: str | Path = "out/radar_comparison.png",
) -> str | None:
    if not HAS_MATPLOTLIB:
        return None

    labels = [d[0] for d in DIMENSIONS]
    keys = [d[1] for d in DIMENSIONS]
    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

    for i, (name, scores) in enumerate(scores_list):
        values = [scores.get(k, 0.0) for k in keys]
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=7)
    ax.set_title(title, size=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison radar chart saved to {output_path}")
    return str(output_path)
