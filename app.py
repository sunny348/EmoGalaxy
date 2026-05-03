import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from deepface import DeepFace
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="Emotion AI",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for modern, stylish UI
st.markdown(
    """
<style>
    /* Modern gradient dark theme */
    .main {
        background: linear-gradient(135deg, #0f0c29, #1e1e1e, #24243e);
        color: #f0f0f0;
        background-attachment: fixed;
    }
    
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Hide sidebar */
    .stSidebar {
        display: none;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p {
        color: #f0f0f0 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1 {
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 2.2rem !important;
        font-weight: 700;
        background: linear-gradient(90deg, #ff6b6b, #ff9191, #ff6b6b);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        letter-spacing: 0.5px;
    }
    
    /* Main container padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #e11d48, #f43f5e) !important;
        color: white !important;
        border-radius: 4px;
        padding: 0.4rem 1.5rem;
        font-weight: bold;
        border: none !important;
        box-shadow: 0 4px 12px rgba(225, 29, 72, 0.4);
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(225, 29, 72, 0.5);
    }
    
    /* Video container */
    .video-container {
        background: linear-gradient(135deg, #1a1a1a, #111111);
        border-radius: 12px;
        padding: 15px;
        margin: 0 auto 1.5rem auto;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        max-width: 800px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 1rem;
        font-size: 0.7rem;
        color: #888;
        padding-bottom: 1rem;
    }
    
    /* Brand banner */
    .brand-banner {
        background: linear-gradient(90deg, rgba(225, 29, 72, 0.1), rgba(225, 29, 72, 0.2));
        border-radius: 12px;
        padding: 8px 15px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(225, 29, 72, 0.2);
        margin-top: 30px;
    }
    
    .brand-logo {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .brand-text {
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        background: linear-gradient(90deg, #ff6b6b, #ff8e8e);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    /* Emotion badges */
    .emotion-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 15px;
        justify-content: center;
    }
    
    .emotion-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.05);
        display: flex;
        align-items: center;
        gap: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .emotion-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        background: rgba(255,255,255,0.1);
    }
    
    /* Instructions expander */
    .stExpander {
        border: 1px solid rgba(225, 29, 72, 0.3);
        border-radius: 8px;
        background-color: rgba(39, 39, 39, 0.4);
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Customize scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        background-color: rgba(30, 30, 30, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(to bottom, #e11d48, #f43f5e);
        border-radius: 10px;
    }
    
    /* App title styling */
    .app-title {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.5rem;
    }
    
    .app-title .emoji {
        font-size: 2.2rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    .app-subtitle {
        text-align: center;
        margin-bottom: 1.5rem;
        font-size: 1rem;
        color: #bbbbbb !important;
        letter-spacing: 0.5px;
    }
    
    /* Webcam styling */
    .stVideoContainer {
        width: 100% !important;
        height: auto !important;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.05);
    }
    
    /* Status bar */
    .status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(90deg, rgba(39, 39, 39, 0.7), rgba(30, 30, 30, 0.7));
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 5px #10b981;
        animation: blink 2s infinite;
    }
    
    @keyframes blink {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    /* Features section */
    .features {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin-bottom: 1.5rem;
    }
    
    .feature-card {
        background: rgba(39, 39, 39, 0.4);
        border-radius: 8px;
        padding: 15px;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
        background: rgba(39, 39, 39, 0.6);
    }
    
    .feature-title {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .feature-description {
        font-size: 0.8rem;
        color: #cccccc;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(225, 29, 72, 0), rgba(225, 29, 72, 0.5), rgba(225, 29, 72, 0));
        margin: 1rem 0;
    }
    
    /* Footer links */
    .footer a {
        color: #888;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: #e11d48;
        text-decoration: underline;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Emotion emojis dictionary
emotion_emojis = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😮",
    "neutral": "😐",
    "disgust": "🤢",
}

# Emotion descriptions
emotion_descriptions = {
    "happy": "Happiness is characterized by feelings of joy, contentment, and satisfaction.",
    "sad": "Sadness is associated with feelings of loss, disappointment, and helplessness.",
    "angry": "Anger is a strong feeling of annoyance, displeasure, or hostility.",
    "fear": "Fear is an emotion induced by perceived danger or threat.",
    "surprise": "Surprise is a brief emotional state experienced as the result of an unexpected event.",
    "neutral": "Neutral emotions indicate a calm state without any strong positive or negative feelings.",
    "disgust": "Disgust is an emotional response of revulsion to something considered offensive or unpleasant.",
}


class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.last_emotion = None
        self.emotion_timestamp = time.time()
        self.emotion_scores = {}
        # Emotion history buffer for smoothing
        self.emotion_history = []
        self.history_max_size = 10
        # Minimum confidence threshold
        self.min_confidence = 30.0

    def get_smoothed_emotion(self, new_emotion, new_scores):
        """Apply temporal smoothing to emotions to reduce rapid switching."""
        # Add new emotion to history
        self.emotion_history.append((new_emotion, new_scores))

        # Keep history at max size
        if len(self.emotion_history) > self.history_max_size:
            self.emotion_history.pop(0)

        # Count occurrences of each emotion in history
        emotion_counts = {}

        # Aggregate scores across history
        aggregated_scores = {emotion: 0 for emotion in new_scores.keys()}

        for emotion, scores in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            for e, score in scores.items():
                aggregated_scores[e] += score

        # Average the scores
        for emotion in aggregated_scores:
            aggregated_scores[emotion] /= len(self.emotion_history)

        # Get the dominant emotion based on frequency and confidence
        max_count = 0
        max_emotion = new_emotion

        for emotion, count in emotion_counts.items():
            if count > max_count and aggregated_scores[emotion] >= self.min_confidence:
                max_count = count
                max_emotion = emotion

        return max_emotion, aggregated_scores

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Add a subtle border
        img = cv2.copyMakeBorder(
            img, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=[225, 29, 72]
        )

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for x, y, w, h in faces:
            face = img[y : y + h, x : x + w]

            # Analyze the face for emotions
            try:
                result = DeepFace.analyze(
                    face, actions=["emotion"], enforce_detection=False
                )
                raw_emotion = result[0]["dominant_emotion"]
                raw_scores = result[0]["emotion"]

                # Apply temporal smoothing
                smooth_emotion, smooth_scores = self.get_smoothed_emotion(
                    raw_emotion, raw_scores
                )

                # Update last detected emotion
                self.last_emotion = smooth_emotion
                self.emotion_timestamp = time.time()
                self.emotion_scores = smooth_scores

                # Draw sleek rectangle around face
                cv2.rectangle(img, (x, y), (x + w, y + h), (225, 29, 72), 2)

                # Create a filled rectangle for text background with transparency
                overlay = img.copy()
                cv2.rectangle(
                    overlay,
                    (x, y - 35),
                    (x + len(smooth_emotion) * 11 + 75, y - 5),
                    (225, 29, 72),
                    -1,
                )
                # Apply the overlay with transparency
                alpha = 0.8
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                # Show confidence percentage alongside emotion
                confidence = smooth_scores[smooth_emotion]
                confidence_text = f"{confidence:.0f}%"

                # Put text of dominant emotion with cleaner font
                cv2.putText(
                    img,
                    f"{emotion_emojis.get(smooth_emotion, '😊')} {smooth_emotion.capitalize()} ({confidence_text})",
                    (x + 5, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
            except Exception as e:
                continue

        # Return the processed frame
        return frame.from_ndarray(img, format="bgr24")


# Brand banner
st.markdown(
    """
    <div class="brand-banner">
        <div class="brand-logo">
            <div style="font-size: 1.2rem;">😊</div>
            <div class="brand-text">EMOTION AI</div>
        </div>
        <div style="font-size: 0.8rem; color: #aaa;">v1.0</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# App title
st.markdown(
    '<div class="app-title"><span class="emoji">✨</span><h1>Emotion AI</h1><span class="emoji">✨</span></div>',
    unsafe_allow_html=True,
)

# App subtitle
st.markdown(
    '<div class="app-subtitle">Real-time emotion detection powered by AI</div>',
    unsafe_allow_html=True,
)

# Emotion badges
st.markdown(
    '<div class="emotion-badges">'
    + "".join(
        [
            f'<div class="emotion-badge">{emoji} {emotion.capitalize()}</div>'
            for emotion, emoji in emotion_emojis.items()
        ]
    )
    + "</div>",
    unsafe_allow_html=True,
)

# Status bar
st.markdown(
    """
    <div class="status-bar">
        <div class="status-indicator">
            <div class="status-dot"></div>
            <span>AI Engine Ready</span>
        </div>
        <div>Live Camera Feed</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Instructions
with st.expander("ℹ️ How to Use", expanded=False):
    st.markdown(
        """
    - Allow camera access when prompted
    - Position your face properly in the frame
    - Try to have good lighting for better detection
    - Click Start to begin emotion analysis
    """
    )

# WebRTC component
st.markdown('<div class="video-container">', unsafe_allow_html=True)
emotion_detector = EmotionDetector()
webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=lambda: emotion_detector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Features section
st.markdown(
    """
    <div class="features">
        <div class="feature-card">
            <div class="feature-title">🎯 High Accuracy</div>
            <div class="feature-description">Advanced AI algorithms ensure precise emotion detection in real-time.</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">⚡ Fast Processing</div>
            <div class="feature-description">Optimized for speed with minimal latency for smooth interaction.</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">🔒 Privacy Focused</div>
            <div class="feature-description">All processing happens locally. No data is sent to external servers.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    '<div class="footer">© 2023 Emotion AI | <a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a> | <a href="#">Help Center</a></div>',
    unsafe_allow_html=True,
)

# Keep app running
if webrtc_ctx.state and webrtc_ctx.state.playing:
    st.empty()
