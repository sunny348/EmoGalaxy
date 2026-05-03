"""
audio_emotion.py — Voice Emotion Detection Module

Records audio from the microphone, extracts acoustic features (MFCC, pitch,
spectral contrast, energy), and maps them to emotion scores using a
heuristic/feature-based classifier.

Emotion classes: happy, sad, angry, fear, neutral, surprise
"""

import numpy as np
import sounddevice as sd
import librosa
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMOTION_CLASSES = ["happy", "sad", "angry", "fear", "neutral", "surprise"]

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_DURATION = 3  # seconds
N_MFCC = 13


# ---------------------------------------------------------------------------
# 1. Audio Recording
# ---------------------------------------------------------------------------
def record_audio_chunk(
    duration: float = DEFAULT_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Record a short audio clip from the default microphone.

    Args:
        duration:    Length of the recording in seconds.
        sample_rate: Sampling rate in Hz.

    Returns:
        Tuple of (audio_array, sample_rate).
        audio_array is a 1-D float32 numpy array.
    """
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()  # block until recording finishes
        return audio.flatten(), sample_rate
    except Exception as e:
        print(f"[audio_emotion] Recording error: {e}")
        # Return silence so downstream code doesn't crash
        return np.zeros(int(duration * sample_rate), dtype=np.float32), sample_rate


# ---------------------------------------------------------------------------
# 2. Feature Extraction
# ---------------------------------------------------------------------------
def extract_audio_features(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Dict[str, float]:
    """
    Extract acoustic features from a raw audio waveform.

    Extracted features:
        - mfcc_mean / mfcc_std  (13 coefficients each)
        - pitch_mean / pitch_std
        - energy_mean / energy_std  (RMS)
        - spectral_contrast_mean

    Args:
        audio:       1-D float32 numpy array.
        sample_rate: Sample rate of the audio.

    Returns:
        Dictionary of named scalar features.
    """
    features: Dict[str, float] = {}

    # Guard against silent / empty audio
    if audio is None or len(audio) == 0 or np.max(np.abs(audio)) < 1e-6:
        return _empty_features()

    # --- MFCCs ---
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

    # --- Pitch (fundamental frequency) ---
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
    if len(f0_valid) == 0:
        f0_valid = np.array([0.0])
    features["pitch_mean"] = float(np.mean(f0_valid))
    features["pitch_std"] = float(np.std(f0_valid))

    # --- RMS Energy ---
    rms = librosa.feature.rms(y=audio)[0]
    features["energy_mean"] = float(np.mean(rms))
    features["energy_std"] = float(np.std(rms))

    # --- Spectral Contrast ---
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    features["spectral_contrast_mean"] = float(np.mean(contrast))

    # --- Zero Crossing Rate (extra signal for voiced / unvoiced) ---
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features["zcr_mean"] = float(np.mean(zcr))

    return features


def _empty_features() -> Dict[str, float]:
    """Return a zeroed-out feature dict (used when audio is silent)."""
    features: Dict[str, float] = {}
    for i in range(N_MFCC):
        features[f"mfcc_{i}_mean"] = 0.0
        features[f"mfcc_{i}_std"] = 0.0
    features["pitch_mean"] = 0.0
    features["pitch_std"] = 0.0
    features["energy_mean"] = 0.0
    features["energy_std"] = 0.0
    features["spectral_contrast_mean"] = 0.0
    features["zcr_mean"] = 0.0
    return features


# ---------------------------------------------------------------------------
# 3. Emotion Prediction (heuristic / rule-based)
# ---------------------------------------------------------------------------
def predict_voice_emotion(
    features: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    """
    Map extracted audio features to emotion scores.

    Uses empirically-grounded heuristic rules based on acoustic research:
        - High pitch + high energy  → happy / angry
        - Low pitch  + low energy   → sad
        - High pitch + high pitch-variance → surprise / fear
        - Moderate everything       → neutral

    Args:
        features: Dict returned by ``extract_audio_features``.

    Returns:
        Tuple of (dominant_emotion, scores_dict).
        scores_dict values are percentages that sum to ~100.
    """
    # Start with uniform prior
    scores = {e: 1.0 for e in EMOTION_CLASSES}

    pitch = features.get("pitch_mean", 0.0)
    pitch_std = features.get("pitch_std", 0.0)
    energy = features.get("energy_mean", 0.0)
    energy_std = features.get("energy_std", 0.0)
    contrast = features.get("spectral_contrast_mean", 0.0)
    zcr = features.get("zcr_mean", 0.0)

    # If features are all zeros (silence) → neutral
    if pitch == 0.0 and energy == 0.0:
        scores["neutral"] = 10.0
        return _normalize_scores(scores)

    # ---- Pitch-based rules ----
    if pitch > 250:  # high pitch
        scores["happy"] += 3.0
        scores["surprise"] += 2.5
        scores["fear"] += 2.0
        scores["angry"] += 1.5
    elif pitch > 180:  # medium-high pitch
        scores["happy"] += 2.0
        scores["surprise"] += 1.5
        scores["angry"] += 1.0
    elif pitch > 120:  # medium pitch
        scores["neutral"] += 2.0
        scores["happy"] += 0.5
    else:  # low pitch
        scores["sad"] += 3.0
        scores["neutral"] += 1.0
        scores["angry"] += 1.0  # anger can also be low-pitched

    # ---- Pitch variability ----
    if pitch_std > 60:
        scores["surprise"] += 2.5
        scores["fear"] += 2.0
        scores["happy"] += 1.0
    elif pitch_std > 30:
        scores["happy"] += 1.5
        scores["angry"] += 1.0
    else:
        scores["sad"] += 1.0
        scores["neutral"] += 1.5

    # ---- Energy-based rules ----
    if energy > 0.08:  # high energy
        scores["angry"] += 3.5
        scores["happy"] += 2.5
        scores["surprise"] += 1.5
    elif energy > 0.03:  # medium energy
        scores["happy"] += 1.5
        scores["neutral"] += 1.0
        scores["fear"] += 0.5
    else:  # low energy
        scores["sad"] += 3.0
        scores["neutral"] += 2.0
        scores["fear"] += 1.0

    # ---- Energy variability ----
    if energy_std > 0.04:
        scores["surprise"] += 1.5
        scores["angry"] += 1.0
    else:
        scores["neutral"] += 1.0

    # ---- Spectral contrast ----
    if contrast > 25:
        scores["angry"] += 1.5
        scores["happy"] += 1.0
    elif contrast < 15:
        scores["sad"] += 1.0
        scores["neutral"] += 0.5

    # ---- Zero crossing rate ----
    if zcr > 0.1:
        scores["angry"] += 1.0
        scores["fear"] += 0.5
    elif zcr < 0.04:
        scores["sad"] += 0.5
        scores["neutral"] += 0.5

    return _normalize_scores(scores)


def _normalize_scores(scores: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    """Normalize raw scores to percentages summing to 100."""
    total = sum(scores.values())
    if total == 0:
        pct = {e: 100.0 / len(scores) for e in scores}
    else:
        pct = {e: round((v / total) * 100, 2) for e, v in scores.items()}

    dominant = max(pct, key=lambda k: pct[k])
    return dominant, pct
