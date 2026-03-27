from src.knowledge.schemas import (
    ChestPainInput,
    ChestPainResult,
    ChestPainPrecipitant,
    ChestPainLocation,
    ChestPainType,
    ChestPainDuration,
)

PRECIPITANT_SCORES = {
    ChestPainPrecipitant.EXERTION_RELIEVED_BY_REST: 3,
    ChestPainPrecipitant.EMOTIONAL_COLD_MEAL: 1,
    ChestPainPrecipitant.UNPREDICTABLE: 0,
    ChestPainPrecipitant.BREATHING: -1,
}

LOCATION_SCORES = {
    ChestPainLocation.RETROSTERNAL_NECK_SHOULDER_JAW_ARM_EPIGASTRIC: 1,
    ChestPainLocation.RIGHT_SIDE_SUBMAMMARY_LOCALIZED: 0,
}

TYPE_SCORES = {
    ChestPainType.CONSTRICTING_CRAMPING_HEAVY_TIGHT_BURNING_DULL: 1,
    ChestPainType.STABBING_SHARP: 0,
    ChestPainType.REPRODUCIBLE_BY_PALPATION: -1,
}

DURATION_SCORES = {
    ChestPainDuration.LESS_THAN_15_MIN: 1,
    ChestPainDuration.FEW_SECONDS: 0,
    ChestPainDuration.MORE_THAN_15_MIN: -1,
}


def score_chest_pain(pain: ChestPainInput) -> ChestPainResult:
    score = (
        PRECIPITANT_SCORES[pain.precipitant]
        + LOCATION_SCORES[pain.location]
        + TYPE_SCORES[pain.pain_type]
        + DURATION_SCORES[pain.duration]
    )

    if score >= 3:
        probability = "HIGH"
        recommendation = "Expedited cardiology referral. High probability of angina."
    elif score >= 1:
        probability = "INTERMEDIATE"
        recommendation = "Further workup recommended: stress echo or CCTA."
    else:
        probability = "LOW"
        recommendation = "Consider non-cardiac causes. Reassess if symptoms persist."

    return ChestPainResult(
        score=score,
        probability=probability,
        recommendation=recommendation,
    )
