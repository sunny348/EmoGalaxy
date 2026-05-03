"""
emotion_fusion.py — Multimodal Emotion Fusion Module

Combines face-based and voice-based emotion scores using weighted averaging.

Default weights:
    face  = 0.65
    voice = 0.35
"""

from typing import Dict, Tuple, Optional


# Shared emotion classes across both modalities
SHARED_EMOTIONS = ["happy", "sad", "angry", "fear", "neutral", "surprise"]


def combine_emotions(
    face_emotion: Optional[str],
    face_scores: Optional[Dict[str, float]],
    voice_emotion: Optional[str],
    voice_scores: Optional[Dict[str, float]],
    face_weight: float = 0.65,
    voice_weight: float = 0.35,
) -> Tuple[str, Dict[str, float]]:
    """
    Fuse face and voice emotion predictions via weighted averaging.

    If one modality is unavailable (``None``), the other modality is
    returned as-is.

    Args:
        face_emotion:  Dominant face emotion string (e.g. "happy").
        face_scores:   Dict mapping each emotion → confidence %.
        voice_emotion: Dominant voice emotion string.
        voice_scores:  Dict mapping each emotion → confidence %.
        face_weight:   Weight for the face channel  (default 0.65).
        voice_weight:  Weight for the voice channel (default 0.35).

    Returns:
        Tuple of (final_dominant_emotion, fused_scores_dict).
        fused_scores values are percentages that sum to ~100.
    """
    # --- Edge case: neither modality available ---
    if face_scores is None and voice_scores is None:
        default = {e: round(100.0 / len(SHARED_EMOTIONS), 2) for e in SHARED_EMOTIONS}
        return "neutral", default

    # --- Only face available ---
    if voice_scores is None or voice_emotion is None:
        normed = _project_to_shared(face_scores)
        dominant = max(normed, key=lambda k: normed[k])
        return dominant, normed

    # --- Only voice available ---
    if face_scores is None or face_emotion is None:
        normed = _project_to_shared(voice_scores)
        dominant = max(normed, key=lambda k: normed[k])
        return dominant, normed

    # --- Both modalities available → weighted fusion ---
    face_proj = _project_to_shared(face_scores)
    voice_proj = _project_to_shared(voice_scores)

    fused: Dict[str, float] = {}
    for emotion in SHARED_EMOTIONS:
        fused[emotion] = face_weight * face_proj.get(
            emotion, 0.0
        ) + voice_weight * voice_proj.get(emotion, 0.0)

    # Re-normalize to 100%
    total = sum(fused.values())
    if total > 0:
        fused = {e: round((v / total) * 100, 2) for e, v in fused.items()}
    else:
        fused = {e: round(100.0 / len(SHARED_EMOTIONS), 2) for e in SHARED_EMOTIONS}

    dominant = max(fused, key=lambda k: fused[k])
    return dominant, fused


def _project_to_shared(
    scores: Dict[str, float],
) -> Dict[str, float]:
    """
    Project an arbitrary emotion score dict onto the shared 6-class set.

    DeepFace returns 7 emotions (includes "disgust"); this maps "disgust"
    into "angry" (closest valence match) and normalises to 100%.
    """
    projected: Dict[str, float] = {e: 0.0 for e in SHARED_EMOTIONS}

    for emotion, score in scores.items():
        lower = emotion.lower()
        if lower in projected:
            projected[lower] += score
        elif lower == "disgust":
            # Fold disgust into angry (closest negative-arousal match)
            projected["angry"] += score * 0.6
            projected["sad"] += score * 0.4
        # Ignore any other unknown emotion keys

    # Normalize
    total = sum(projected.values())
    if total > 0:
        projected = {e: round((v / total) * 100, 2) for e, v in projected.items()}

    return projected
