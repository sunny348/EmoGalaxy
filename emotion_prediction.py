"""
Emotion Prediction Module – Markov Chain Forecasting
=====================================================
Predicts the next likely emotional state from an observed emotion timeline
using a first-order Markov chain (transition-probability matrix).

Public API
----------
build_transition_matrix(emotion_sequence)
predict_next_emotion(current_emotion, transition_matrix)
predict_future_emotions(sequence, steps=5)
create_forecast_chart(predictions)
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import plotly.graph_objects as go

# All emotions recognised by the detection pipeline
EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"]

# Emoji mapping (mirrors appv4)
EMOTION_EMOJIS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😮",
    "neutral": "😐",
    "disgust": "🤢",
}


# ---------------------------------------------------------------------------
# 1. Transition Matrix
# ---------------------------------------------------------------------------


def build_transition_matrix(
    emotion_sequence: list[str],
) -> dict[str, dict[str, float]]:
    """Build a row-normalised Markov transition matrix.

    Parameters
    ----------
    emotion_sequence : list[str]
        Chronologically ordered list of detected emotions.

    Returns
    -------
    dict[str, dict[str, float]]
        ``matrix[from_emotion][to_emotion]`` = probability.
        Every row sums to 1.0 (or 0.0 if the emotion was never observed).
    """
    # Count raw transitions
    counts: dict[str, Counter] = {e: Counter() for e in EMOTIONS}

    for prev, curr in zip(emotion_sequence[:-1], emotion_sequence[1:]):
        prev_lower = prev.lower()
        curr_lower = curr.lower()
        if prev_lower in counts:
            counts[prev_lower][curr_lower] += 1

    # Normalise each row
    matrix: dict[str, dict[str, float]] = {}
    for emotion, transitions in counts.items():
        total = sum(transitions.values())
        if total > 0:
            matrix[emotion] = {e: transitions.get(e, 0) / total for e in EMOTIONS}
        else:
            # Uniform distribution as fallback when emotion was never seen
            matrix[emotion] = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}

    return matrix


# ---------------------------------------------------------------------------
# 2. Single-step Prediction
# ---------------------------------------------------------------------------


def predict_next_emotion(
    current_emotion: str,
    transition_matrix: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Predict probabilities for the next emotion given the current one.

    Parameters
    ----------
    current_emotion : str
        The most recently observed emotion.
    transition_matrix : dict
        Output of :func:`build_transition_matrix`.

    Returns
    -------
    dict[str, float]
        ``{emotion: probability}`` sorted by probability descending.
    """
    current = current_emotion.lower()
    if current in transition_matrix:
        probs = transition_matrix[current]
    else:
        # Unseen emotion → uniform
        probs = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}

    return dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# 3. Multi-step Forecast
# ---------------------------------------------------------------------------


def predict_future_emotions(
    sequence: list[str],
    steps: int = 5,
) -> list[dict[str, Any]]:
    """Forecast the next *steps* emotions using the Markov model.

    At each step the most probable next emotion is selected and fed back
    as input for the following step.

    Parameters
    ----------
    sequence : list[str]
        Full observed emotion timeline (length must be > 0).
    steps : int
        Number of future steps to predict.

    Returns
    -------
    list[dict]
        Each dict contains:
        ``{ "step": int, "predicted_emotion": str, "confidence": float,
            "probabilities": dict[str, float] }``
    """
    if not sequence:
        return []

    matrix = build_transition_matrix(sequence)
    predictions: list[dict[str, Any]] = []
    current = sequence[-1].lower()

    for step_idx in range(1, steps + 1):
        probs = predict_next_emotion(current, matrix)
        top_emotion = next(iter(probs))  # highest probability key
        top_confidence = probs[top_emotion]

        predictions.append(
            {
                "step": step_idx,
                "predicted_emotion": top_emotion,
                "confidence": top_confidence,
                "probabilities": probs,
            }
        )

        # Feed predicted emotion back for next step
        current = top_emotion

    return predictions


# ---------------------------------------------------------------------------
# 4. Forecast Visualisation
# ---------------------------------------------------------------------------

# Colour palette matching the app theme
_EMOTION_COLORS = {
    "happy": "#FBBF24",
    "sad": "#60A5FA",
    "angry": "#F87171",
    "fear": "#A78BFA",
    "surprise": "#34D399",
    "neutral": "#9CA3AF",
    "disgust": "#FB923C",
}


def create_forecast_chart(
    predictions: list[dict[str, Any]],
) -> go.Figure:
    """Create a Plotly grouped-bar chart of future emotion probabilities.

    Parameters
    ----------
    predictions : list[dict]
        Output of :func:`predict_future_emotions`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    step_labels = [f"Step {p['step']}" for p in predictions]

    for emotion in EMOTIONS:
        values = [p["probabilities"].get(emotion, 0) * 100 for p in predictions]
        fig.add_trace(
            go.Bar(
                name=f"{EMOTION_EMOJIS.get(emotion, '')} {emotion.capitalize()}",
                x=step_labels,
                y=values,
                marker_color=_EMOTION_COLORS.get(emotion, "#888"),
            )
        )

    fig.update_layout(
        barmode="group",
        title="Emotion Forecast – Future Step Probabilities",
        xaxis_title="Prediction Step",
        yaxis_title="Probability (%)",
        template="plotly_dark",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="white", size=12),
        height=400,
        margin=dict(t=50, b=50, l=50, r=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
    )

    return fig
