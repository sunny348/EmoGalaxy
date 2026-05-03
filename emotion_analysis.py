"""
Emotion Behavior Analysis Engine
================================
Standalone module for computing behavioral emotion metrics from
an emotion-timeline DataFrame produced by appv4.process_video_for_emotions().

No Streamlit dependency – only pandas, numpy, and plotly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# 1. Dominant Emotion
# ---------------------------------------------------------------------------


def _dominant_emotion(df: pd.DataFrame) -> tuple[str, float]:
    """Return (emotion_name, percentage) for the most frequent dominant_emotion."""
    counts = df["dominant_emotion"].value_counts()
    top = counts.idxmax()
    pct = counts[top] / len(df) * 100
    return top, round(pct, 2)


# ---------------------------------------------------------------------------
# 2. Emotion Volatility Index
# ---------------------------------------------------------------------------


def calculate_volatility(df: pd.DataFrame) -> float:
    """
    volatility = number_of_emotion_switches / total_frames

    A switch occurs whenever consecutive rows have different dominant_emotion.
    Returns a float in [0, 1].
    """
    if len(df) < 2:
        return 0.0
    emotions = df["dominant_emotion"].values
    switches = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1])
    return round(switches / len(emotions), 4)


# ---------------------------------------------------------------------------
# 3. Emotional Stability Score
# ---------------------------------------------------------------------------


def _stability(volatility: float) -> float:
    """stability = 1 - volatility"""
    return round(1 - volatility, 4)


# ---------------------------------------------------------------------------
# 4. Peak Emotion Moments
# ---------------------------------------------------------------------------


def detect_peak_emotions(
    df: pd.DataFrame,
    threshold: float = 80.0,
) -> list[dict[str, Any]]:
    """
    Detect moments where the dominant-emotion confidence exceeds *threshold* (%).

    Returns a list of dicts:
        { "start": float, "end": float, "emotion": str, "max_confidence": float }

    Consecutive frames with the same peak emotion are merged into a single range.
    """
    peaks: list[dict[str, Any]] = []
    current_range: dict[str, Any] | None = None

    for _, row in df.iterrows():
        emotion = row["dominant_emotion"]
        confidence = row.get(emotion, 0.0)
        ts = row["timestamp"]

        if confidence >= threshold:
            if current_range is not None and current_range["emotion"] == emotion:
                # extend the existing range
                current_range["end"] = ts
                current_range["max_confidence"] = max(
                    current_range["max_confidence"], confidence
                )
            else:
                # close previous range if any, start a new one
                if current_range is not None:
                    peaks.append(current_range)
                current_range = {
                    "start": ts,
                    "end": ts,
                    "emotion": emotion,
                    "max_confidence": confidence,
                }
        else:
            if current_range is not None:
                peaks.append(current_range)
                current_range = None

    # flush last range
    if current_range is not None:
        peaks.append(current_range)

    return peaks


# ---------------------------------------------------------------------------
# 5. Stress Indicators
# ---------------------------------------------------------------------------

_STRESS_PATTERNS: list[list[str]] = [
    ["angry", "sad", "neutral"],
    ["angry", "fear", "sad"],
    ["angry", "disgust", "sad"],
    ["fear", "angry", "sad"],
]

_RAPID_SWITCH_WINDOW = 5  # consecutive frames to inspect
_RAPID_SWITCH_THRESHOLD = 3  # minimum switches inside that window


def detect_stress_indicators(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect stress-related patterns in the emotion timeline.

    Two detection strategies:
    1. **Sequence matching** – look for known stress sequences
       (e.g. angry → sad → neutral) in consecutive frames.
    2. **Rapid switching** – flag windows of *_RAPID_SWITCH_WINDOW* frames
       that contain ≥ *_RAPID_SWITCH_THRESHOLD* emotion switches.

    Returns a list of dicts:
        { "timestamp": float, "type": str, "detail": str }
    """
    indicators: list[dict[str, Any]] = []
    emotions = df["dominant_emotion"].values
    timestamps = df["timestamp"].values

    # --- Sequence matching ---
    for pattern in _STRESS_PATTERNS:
        plen = len(pattern)
        for i in range(len(emotions) - plen + 1):
            window = list(emotions[i : i + plen])
            if window == pattern:
                indicators.append(
                    {
                        "timestamp": float(timestamps[i]),
                        "type": "pattern",
                        "detail": f"{'→'.join(p.capitalize() for p in pattern)} detected",
                    }
                )

    # --- Rapid switching ---
    for i in range(len(emotions) - _RAPID_SWITCH_WINDOW + 1):
        window = emotions[i : i + _RAPID_SWITCH_WINDOW]
        switches = sum(1 for j in range(1, len(window)) if window[j] != window[j - 1])
        if switches >= _RAPID_SWITCH_THRESHOLD:
            indicators.append(
                {
                    "timestamp": float(timestamps[i]),
                    "type": "rapid_switch",
                    "detail": f"{switches} emotion switches in {_RAPID_SWITCH_WINDOW} frames",
                }
            )

    # Deduplicate by timestamp (keep first occurrence)
    seen_ts: set[float] = set()
    unique: list[dict[str, Any]] = []
    for ind in indicators:
        if ind["timestamp"] not in seen_ts:
            seen_ts.add(ind["timestamp"])
            unique.append(ind)

    return sorted(unique, key=lambda x: x["timestamp"])


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


def compute_emotion_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Master function – computes every behavioural metric.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``process_video_for_emotions()`` with columns:
        timestamp, dominant_emotion, happy, sad, angry, fear, surprise,
        neutral, disgust.

    Returns
    -------
    dict with keys:
        dominant_emotion, dominant_pct, emotion_counts, emotion_percentages,
        volatility, stability, peak_emotions, stress_indicators
    """
    dom_emo, dom_pct = _dominant_emotion(df)
    vol = calculate_volatility(df)

    counts = df["dominant_emotion"].value_counts().to_dict()
    pcts = {k: round(v / len(df) * 100, 2) for k, v in counts.items()}

    return {
        "dominant_emotion": dom_emo,
        "dominant_pct": dom_pct,
        "emotion_counts": counts,
        "emotion_percentages": pcts,
        "volatility": vol,
        "stability": _stability(vol),
        "peak_emotions": detect_peak_emotions(df),
        "stress_indicators": detect_stress_indicators(df),
    }


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def generate_behavioral_summary(stats: dict[str, Any]) -> str:
    """
    Format the stats dict into a human-readable text block.
    """
    lines: list[str] = []
    lines.append("═══ User Emotional Profile ═══\n")
    lines.append(
        f"Dominant Emotion: {stats['dominant_emotion'].capitalize()} "
        f"({stats['dominant_pct']:.1f}%)"
    )
    lines.append(f"Emotional Stability Score: {stats['stability']:.2f}")
    lines.append(f"Volatility Index: {stats['volatility']:.2f}")

    # Peak emotion moments grouped by emotion
    peaks = stats.get("peak_emotions", [])
    if peaks:
        lines.append("\nPeak Emotion Moments:")
        by_emotion: dict[str, list[str]] = {}
        for p in peaks:
            label = p["emotion"].capitalize()
            rng = (
                f"  {_fmt_time(p['start'])} – {_fmt_time(p['end'])}"
                if p["start"] != p["end"]
                else f"  {_fmt_time(p['start'])}"
            )
            by_emotion.setdefault(label, []).append(rng)
        for emo, ranges in by_emotion.items():
            lines.append(f"  {emo}:")
            for r in ranges:
                lines.append(f"    {r}")

    # Stress indicators
    stress = stats.get("stress_indicators", [])
    if stress:
        lines.append("\nStress Indicators:")
        for s in stress:
            lines.append(f"  Detected at {_fmt_time(s['timestamp'])} – {s['detail']}")
    else:
        lines.append("\nStress Indicators: None detected")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotly Visualisations
# ---------------------------------------------------------------------------


def create_volatility_chart(df: pd.DataFrame, window: int = 5) -> go.Figure:
    """
    Rolling-window emotion volatility over time.

    For each window of *window* consecutive frames, compute the fraction of
    emotion switches.  Returns a Plotly Figure.
    """
    emotions = df["dominant_emotion"].values
    timestamps = df["timestamp"].values
    vol_values: list[float] = []
    vol_times: list[float] = []

    for i in range(len(emotions) - window + 1):
        w = emotions[i : i + window]
        switches = sum(1 for j in range(1, len(w)) if w[j] != w[j - 1])
        vol_values.append(switches / (window - 1))
        # use the midpoint timestamp of the window
        vol_times.append(float(timestamps[i + window // 2]))

    vol_df = pd.DataFrame({"timestamp": vol_times, "volatility": vol_values})

    fig = px.area(
        vol_df,
        x="timestamp",
        y="volatility",
        title="Emotion Volatility Over Time",
        labels={"volatility": "Volatility (0-1)", "timestamp": "Time (seconds)"},
        color_discrete_sequence=["#f43f5e"],
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="white", size=14),
        height=350,
        margin=dict(t=50, b=50, l=50, r=30),
        xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            range=[0, 1],
        ),
    )
    fig.update_traces(
        line=dict(width=2, color="#f43f5e"),
        fillcolor="rgba(244, 63, 94, 0.15)",
    )
    return fig


def create_peak_markers_chart(
    df: pd.DataFrame,
    peaks: list[dict[str, Any]],
) -> go.Figure:
    """
    Emotion confidence timeline with vertical markers at peak moments.
    """
    emotion_cols = ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"]
    available_cols = [c for c in emotion_cols if c in df.columns]

    fig = px.line(
        df,
        x="timestamp",
        y=available_cols,
        title="Emotion Timeline with Peak Markers",
        labels={"value": "Confidence (%)", "timestamp": "Time (seconds)"},
        color_discrete_sequence=px.colors.sequential.Plasma,
    )

    # Add vertical lines for peaks
    for peak in peaks:
        fig.add_vline(
            x=peak["start"],
            line_dash="dash",
            line_color="#10b981",
            line_width=1.5,
            annotation_text=f"⬆ {peak['emotion'].capitalize()}",
            annotation_font_size=10,
            annotation_font_color="#10b981",
        )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="white", size=14),
        height=400,
        margin=dict(t=50, b=50, l=50, r=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
        yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
    )
    fig.update_traces(line=dict(width=3))
    return fig
