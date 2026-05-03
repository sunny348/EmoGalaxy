import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
from deepface import DeepFace
import numpy as np
import time
import base64
from io import BytesIO
from PIL import Image
import os

# Behavioral emotion analysis
from emotion_analysis import (
    compute_emotion_statistics,
    calculate_volatility,
    detect_peak_emotions,
    detect_stress_indicators,
    generate_behavioral_summary,
    create_volatility_chart,
    create_peak_markers_chart,
)
from datetime import datetime
import av
import tempfile
import matplotlib.pyplot as plt
import matplotlib
import threading

matplotlib.use("Agg")  # Use non-interactive backend
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Multimodal modules
from audio_emotion import (
    record_audio_chunk,
    extract_audio_features,
    predict_voice_emotion,
)
from emotion_fusion import combine_emotions
from face_tracker import CentroidTracker, match_rects_to_ids, record_emotion
from emotion_prediction import (
    build_transition_matrix,
    predict_next_emotion,
    predict_future_emotions,
    create_forecast_chart,
    EMOTION_EMOJIS as PRED_EMOJIS,
)

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

    /* Multimodal emotion cards */
    .multimodal-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 1rem 0;
    }

    .emotion-card {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.8), rgba(40, 40, 40, 0.6));
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .emotion-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    }

    .emotion-card .card-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #aaa;
        margin-bottom: 6px;
    }

    .emotion-card .card-emoji {
        font-size: 2rem;
        margin-bottom: 4px;
    }

    .emotion-card .card-emotion {
        font-size: 1.1rem;
        font-weight: 600;
        background: linear-gradient(90deg, #ff6b6b, #ff9191);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    .emotion-card .card-confidence {
        font-size: 0.8rem;
        color: #bbb;
        margin-top: 4px;
    }

    .emotion-card.fused {
        border-color: rgba(225, 29, 72, 0.4);
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.1), rgba(40, 40, 40, 0.6));
    }

    .voice-toggle-container {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.8rem;
        padding: 8px 12px;
        background: rgba(39, 39, 39, 0.4);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* Emotion Forecast Panel */
    .forecast-panel {
        background: linear-gradient(135deg, rgba(30, 30, 50, 0.85), rgba(20, 20, 40, 0.9));
        border-radius: 12px;
        padding: 20px;
        margin: 1rem 0;
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.15);
    }

    .forecast-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(90deg, #a78bfa, #c4b5fd);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    .forecast-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 10px;
        margin-top: 10px;
    }

    .forecast-bar {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        transition: all 0.3s ease;
    }

    .forecast-bar:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.08);
    }

    .forecast-bar .fb-emotion {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 4px;
    }

    .forecast-bar .fb-pct {
        font-size: 1.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #c4b5fd);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    .forecast-bar .fb-emoji {
        font-size: 1.4rem;
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
        # For video recording
        self.raw_frames = []  # For original frames
        self.processed_frames = []  # For frames with emotion outlines and labels
        self.is_recording = False
        self.frame_count = 0  # Add a frame counter for debugging
        # Multi-person tracking
        self.tracker = CentroidTracker(max_disappeared=15)
        self.person_emotion_history: dict[int, list[tuple[float, str, dict]]] = {}

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

        # Store original frame
        if self.is_recording:
            self.raw_frames.append(img.copy())

        # Add a subtle border
        img = cv2.copyMakeBorder(
            img, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=[225, 29, 72]
        )

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # --- Multi-person tracking ---
        faces_list = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        tracked_ids = self.tracker.update(faces_list)
        id_to_rect = match_rects_to_ids(faces_list, tracked_ids)

        # Analyze each tracked person
        current_timestamp = time.time()
        for person_id, (x, y, w, h) in id_to_rect.items():
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

                # Update last detected emotion (use first/primary person)
                self.last_emotion = smooth_emotion
                self.emotion_timestamp = current_timestamp
                self.emotion_scores = smooth_scores

                # Record per-person emotion history
                record_emotion(
                    self.person_emotion_history,
                    person_id,
                    current_timestamp,
                    smooth_emotion,
                    dict(smooth_scores),
                )

                # Draw sleek rectangle around face
                cv2.rectangle(img, (x, y), (x + w, y + h), (225, 29, 72), 2)

                # Create label with person ID
                label = f"P{person_id + 1}: {smooth_emotion.capitalize()}"
                confidence = smooth_scores[smooth_emotion]
                confidence_text = f"{confidence:.0f}%"
                full_label = f"{label} ({confidence_text})"

                # Create a filled rectangle for text background with transparency
                overlay = img.copy()
                cv2.rectangle(
                    overlay,
                    (x, y - 35),
                    (x + len(full_label) * 10 + 20, y - 5),
                    (225, 29, 72),
                    -1,
                )
                # Apply the overlay with transparency
                alpha = 0.8
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                # Put text with person ID and emotion
                cv2.putText(
                    img,
                    full_label,
                    (x + 5, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
            except Exception as e:
                continue

        # Store processed frame with emotion outlines and labels
        if self.is_recording:
            self.processed_frames.append(img.copy())
            self.frame_count += 1

            # Limit the number of frames to avoid memory issues
            if len(self.processed_frames) > 12000:  # 10 mins at 20fps
                self.processed_frames.pop(0)
                self.raw_frames.pop(0)

            # Log periodically for debugging
            if self.frame_count % 100 == 0:
                print(f"Recording... Frames captured: {len(self.processed_frames)}")

        # Return the processed frame for display
        return frame.from_ndarray(img, format="bgr24")

    def start_recording(self):
        self.is_recording = True
        print("Recording started")
        # Only clear frames if there are none already
        if not self.processed_frames:
            self.raw_frames = []
            self.processed_frames = []
            self.frame_count = 0

    def stop_recording(self):
        self.is_recording = False
        print(f"Recording stopped. Total frames: {len(self.processed_frames)}")

    def get_processed_frames(self):
        return self.processed_frames

    def get_raw_frames(self):
        return self.raw_frames


def generate_pdf_from_html(html_content, filename="emotion_report.pdf"):
    """
    Convert HTML content to a PDF file.

    Args:
        html_content: String containing HTML to convert
        filename: Name for the output PDF file

    Returns:
        Path to the generated PDF file or None if conversion fails
    """
    try:
        # Create a temporary directory for the PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, filename)

            # Generate PDF from HTML
            HTML(string=html_content).write_pdf(pdf_path)

            # Return the PDF data
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()

            return pdf_data
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None


# Function to analyze image and detect emotion
def analyze_image(uploaded_image):
    # Convert to OpenCV format
    image = np.array(uploaded_image)
    if image.shape[2] == 4:  # If RGBA, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Create a copy for drawing on
    display_image = image.copy()

    # Add a subtle border
    display_image = cv2.copyMakeBorder(
        display_image, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=[225, 29, 72]
    )

    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []

    for x, y, w, h in faces:
        face = image[y : y + h, x : x + w]

        # Analyze the face for emotions
        try:
            result = DeepFace.analyze(
                face, actions=["emotion"], enforce_detection=False
            )
            emotion = result[0]["dominant_emotion"]
            emotion_scores = result[0]["emotion"]

            # Draw rectangle around face
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (225, 29, 72), 2)

            # Create a filled rectangle for text background with transparency
            overlay = display_image.copy()
            cv2.rectangle(
                overlay,
                (x, y - 35),
                (x + len(emotion) * 11 + 75, y - 5),
                (225, 29, 72),
                -1,
            )
            # Apply the overlay with transparency
            alpha = 0.8
            cv2.addWeighted(overlay, alpha, display_image, 1 - alpha, 0, display_image)

            # Show confidence percentage alongside emotion
            confidence = emotion_scores[emotion]
            confidence_text = f"{confidence:.0f}%"

            # Put text of dominant emotion
            cv2.putText(
                display_image,
                f"{emotion_emojis.get(emotion, '😊')} {emotion.capitalize()} ({confidence_text})",
                (x + 5, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            results.append(
                {
                    "emotion": emotion,
                    "confidence": confidence,
                    "all_scores": emotion_scores,
                }
            )

        except Exception as e:
            print(f"Error analyzing face: {e}")
            continue

    # Convert back to PIL for displaying in Streamlit
    rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(rgb_image)

    return result_image, results


# Function to save video from frames
def save_video(frames, filename, fps=20):
    if not frames:
        return None

    try:
        # Create directory if it doesn't exist
        os.makedirs("saved_videos", exist_ok=True)

        # Define the output path
        output_path = os.path.join("saved_videos", filename)

        # Get frame dimensions
        height, width, _ = frames[0].shape

        # Create VideoWriter object - using mp4v codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4v instead of XVID
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write frames to video
        for frame in frames:
            out.write(frame)

        # Release VideoWriter
        out.release()

        return output_path
    except Exception as e:
        st.error(f"Error saving video: {str(e)}")
        return None


# Function to create download link
def get_binary_file_downloader_html(file_path, file_label="File"):
    try:
        with open(file_path, "rb") as f:
            data = f.read()

        # Use a more direct way to create download links
        b64 = base64.b64encode(data).decode()
        download_button_str = f"""
            <a href="data:video/mp4;base64,{b64}" download="{os.path.basename(file_path)}">
                <button style="
                    background: linear-gradient(90deg, #e11d48, #f43f5e);
                    color: white;
                    padding: 0.5rem 1rem;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-weight: bold;
                    margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    {file_label} 📥
                </button>
            </a>
        """
        return download_button_str
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return ""


# Function to process video for emotion analysis
def process_video_for_emotions(video_path, sampling_rate=1):
    """
    Process a video file and detect emotions in frames sampled at the given rate.

    Args:
        video_path: Path to the video file
        sampling_rate: How many frames to sample per second

    Returns:
        DataFrame with timestamped emotion data, key frames, and summary stats
    """
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Open the video
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    # Progress bar for video processing
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Data storage
    emotion_data = []
    key_frames = []
    previous_emotion = None
    frame_skip = max(1, int(fps / sampling_rate))

    # Multi-person tracking
    tracker = CentroidTracker(max_disappeared=15)
    per_person_data: dict[int, list[dict]] = (
        {}
    )  # person_id -> [{timestamp, emotion, scores}]

    # Process the video
    for frame_idx in range(0, frame_count, frame_skip):
        # Update progress
        progress_percentage = frame_idx / frame_count
        progress_bar.progress(progress_percentage)
        status_text.text(
            f"Processing frame {frame_idx}/{frame_count} ({progress_percentage:.1%})"
        )

        # Set frame position and read frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if not ret:
            break

        # Calculate timestamp
        timestamp = frame_idx / fps

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # If no faces detected, continue to next frame
        if len(faces) == 0:
            continue

        # --- Multi-person tracking for this frame ---
        faces_list = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        tracked_ids = tracker.update(faces_list)
        id_to_rect = match_rects_to_ids(faces_list, tracked_ids)

        # For each tracked person, detect emotions
        frame_emotions = []
        for person_id, (x, y, w, h) in id_to_rect.items():
            face = frame[y : y + h, x : x + w]

            # Skip if face is too small
            if w < 30 or h < 30:
                continue

            try:
                # Analyze emotions
                result = DeepFace.analyze(
                    face, actions=["emotion"], enforce_detection=False
                )
                emotion = result[0]["dominant_emotion"]
                scores = result[0]["emotion"]

                # Store emotion data
                frame_emotions.append({"emotion": emotion, "scores": scores})

                # Record per-person data
                if person_id not in per_person_data:
                    per_person_data[person_id] = []
                per_person_data[person_id].append(
                    {"timestamp": timestamp, "emotion": emotion, "scores": scores}
                )

                # Draw on frame (for key frames) with person ID
                cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 29, 72), 2)

                # Create label with person ID
                confidence = scores[emotion]
                confidence_text = f"{confidence:.0f}%"
                full_label = (
                    f"P{person_id + 1}: {emotion.capitalize()} ({confidence_text})"
                )

                # Create a filled rectangle for text background with transparency
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (x, y - 35),
                    (x + len(full_label) * 10 + 20, y - 5),
                    (225, 29, 72),
                    -1,
                )
                # Apply the overlay with transparency
                alpha = 0.8
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Put text with person ID and emotion
                cv2.putText(
                    frame,
                    full_label,
                    (x + 5, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

                # Save key frames when emotion changes
                if emotion != previous_emotion and len(key_frames) < 8:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    key_frames.append(
                        {"timestamp": timestamp, "frame": frame_rgb, "emotion": emotion}
                    )
                    previous_emotion = emotion

            except Exception as e:
                continue

        # If emotions were detected in this frame
        if frame_emotions:
            # Get the most frequent emotion in this frame
            emotions_list = [item["emotion"] for item in frame_emotions]
            dominant_emotion = max(set(emotions_list), key=emotions_list.count)

            # Get average scores for this emotion
            avg_scores = {}
            for emotion_key in frame_emotions[0]["scores"].keys():
                avg_scores[emotion_key] = sum(
                    item["scores"].get(emotion_key, 0) for item in frame_emotions
                ) / len(frame_emotions)

            # Add to time series data
            emotion_data.append(
                {
                    "timestamp": timestamp,
                    "emotion": dominant_emotion,
                    "scores": avg_scores,
                }
            )

    # Clean up
    video.release()
    progress_bar.empty()
    status_text.empty()

    # Convert to DataFrame for easier analysis
    if not emotion_data:
        return None, None, None, {}

    # Create a DataFrame with emotion scores for each timestamp
    df_rows = []
    for entry in emotion_data:
        row = {"timestamp": entry["timestamp"], "dominant_emotion": entry["emotion"]}
        # Add individual emotion scores
        for emotion, score in entry["scores"].items():
            row[emotion] = score
        df_rows.append(row)

    df = pd.DataFrame(df_rows)

    # Calculate summary statistics
    emotion_counts = df["dominant_emotion"].value_counts()
    emotion_percentages = emotion_counts / len(df) * 100

    summary = {
        "duration": duration,
        "frame_count": frame_count,
        "emotion_counts": emotion_counts.to_dict(),
        "emotion_percentages": emotion_percentages.to_dict(),
    }

    return df, key_frames, summary, per_person_data


# Function to generate report as HTML
def generate_emotion_report(df, key_frames, summary, per_person_data=None):
    """Generate an HTML report for the video emotion analysis"""
    if df is None or df.empty:
        return None

    # Create emotion timeline chart with improved size and visibility
    timeline_fig = px.line(
        df,
        x="timestamp",
        y=["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"],
        title="Emotion Confidence Timeline",
        labels={"value": "Confidence (%)", "timestamp": "Time (seconds)"},
        color_discrete_sequence=px.colors.sequential.Plasma,  # Use the same color sequence as the pie chart
    )
    timeline_fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="white", size=14),
        height=400,  # Increased height
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
    # Improve line visibility
    timeline_fig.update_traces(line=dict(width=3))  # Make lines thicker

    # Create emotion distribution pie chart with improved visibility
    pie_data = pd.DataFrame(
        {
            "emotion": summary["emotion_percentages"].keys(),
            "percentage": summary["emotion_percentages"].values(),
        }
    )
    pie_fig = px.pie(
        pie_data,
        values="percentage",
        names="emotion",
        title="Emotion Distribution",
        color_discrete_sequence=px.colors.sequential.Plasma,
    )
    pie_fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="white", size=14),
        height=400,  # Increased height
        margin=dict(t=50, b=30, l=30, r=30),
        legend=dict(font=dict(size=12)),
    )
    pie_fig.update_traces(textinfo="percent+label", textfont_size=12)

    # Convert key frames to base64 for embedding in HTML
    key_frame_images = []
    if key_frames:
        for i, frame_data in enumerate(key_frames):
            # Convert image to PIL then to base64
            pil_img = Image.fromarray(frame_data["frame"])
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=90)  # Increased quality
            img_str = base64.b64encode(buffer.getvalue()).decode()

            key_frame_images.append(
                {
                    "image": img_str,
                    "timestamp": frame_data["timestamp"],
                    "emotion": frame_data["emotion"],
                }
            )

    # Generate HTML content with improved styling
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Emotion Analysis Report</title>
        <style>
            /* Base styles */
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #f0f0f0;
                background: linear-gradient(135deg, #0f0c29, #1e1e1e, #24243e);
                margin: 0;
                padding: 0;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(20, 20, 20, 0.85);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            }}
            
            /* Typography */
            h1, h2, h3 {{
                color: #f0f0f0;
                text-align: center;
                margin-top: 1.5em;
                margin-bottom: 0.8em;
            }}
            
            h1 {{
                font-size: 2.5em;
                margin-top: 0.5em;
                background: linear-gradient(90deg, #ff6b6b, #ff9191, #ff6b6b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: inline-block;
            }}
            
            /* Header section */
            .header {{
                text-align: center;
                margin-bottom: 2em;
                padding-bottom: 1.5em;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}
            
            /* Summary cards */
            .summary {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin-bottom: 2.5em;
                gap: 15px;
            }}
            
            .stat-card {{
                flex: 1 1 30%;
                min-width: 250px;
                background: linear-gradient(135deg, rgba(30, 30, 30, 0.7), rgba(40, 40, 40, 0.7));
                border-radius: 10px;
                padding: 20px;
                margin: 0;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                border: 1px solid rgba(225, 29, 72, 0.3);
            }}
            
            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
                background: linear-gradient(90deg, #f43f5e, #ff9191);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            
            /* Chart containers */
            .charts {{
                display: flex;
                flex-direction: column;
                margin-bottom: 3em;
                gap: 30px;
            }}
            
            .chart {{
                width: 100%;
                background: rgba(20, 20, 20, 0.7);
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                border: 1px solid rgba(255,255,255,0.05);
            }}
            
            /* Key frames section */
            .key-frames-title {{
                margin-top: 2em;
                margin-bottom: 1em;
                text-align: center;
            }}
            
            .key-frames {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 20px;
                margin: 1.5em 0;
            }}
            
            .key-frame {{
                background: rgba(25, 25, 25, 0.8);
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.05);
                transition: transform 0.2s ease;
            }}
            
            .key-frame:hover {{
                transform: translateY(-5px);
                box-shadow: 0 6px 15px rgba(0,0,0,0.4);
            }}
            
            .key-frame img {{
                width: 100%;
                border-radius: 8px;
                margin-bottom: 10px;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            
            .key-frame-info {{
                text-align: center;
                padding: 5px 0;
            }}
            
            .emotion-label {{
                font-weight: bold;
                font-size: 1.1em;
                background: linear-gradient(90deg, #f43f5e, #ff9191);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 5px 0;
                display: inline-block;
            }}
            
            .timeline-label {{
                color: #bbb;
                font-size: 0.9em;
            }}
            
            /* Logo and branding */
            .logo {{
                text-align: center;
                margin-bottom: 15px;
            }}
            
            .logo span {{
                font-size: 3em;
                filter: drop-shadow(0 0 5px rgba(255, 107, 107, 0.5));
            }}
            
            /* Footer */
            footer {{
                text-align: center;
                margin-top: 3em;
                padding-top: 1.5em;
                border-top: 1px solid rgba(255,255,255,0.1);
                color: #888;
                font-size: 0.9em;
            }}
            
            /* Iframe adjustments */
            iframe {{
                border: none;
                width: 100%;
                height: 100%;
                min-height: 400px;
            }}
            
            /* Responsive adjustments */
            @media (max-width: 768px) {{
                .container {{
                    padding: 15px;
                }}
                
                .charts {{
                    flex-direction: column;
                }}
                
                .stat-card {{
                    min-width: 100%;
                }}
                
                .key-frames {{
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <span>😊</span>
                </div>
                <h1>Video Emotion Analysis Report</h1>
                <p>Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <div class="stat-card">
                    <h3>Duration</h3>
                    <div class="stat-value">{summary['duration']:.1f}s</div>
                </div>
                <div class="stat-card">
                    <h3>Frames Analyzed</h3>
                    <div class="stat-value">{len(df)}</div>
                </div>
                <div class="stat-card">
                    <h3>Dominant Emotion</h3>
                    <div class="stat-value">
                        {max(summary['emotion_percentages'].items(), key=lambda x: x[1])[0].capitalize()}
                    </div>
                </div>
            </div>
            
            <h2>Emotional Analysis</h2>
            
            <div class="charts">
                <div class="chart">
                    <h3>Emotion Timeline</h3>
                    {timeline_fig.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})}
                </div>
                
                <div class="chart">
                    <h3>Emotion Distribution</h3>
                    {pie_fig.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})}
                </div>
            </div>
            
            <!-- ========== Behavioral Analysis Section ========== -->
    """

    # --- Compute behavioral stats and inject into report ---
    try:
        behavior_stats = compute_emotion_statistics(df)
        vol_chart = create_volatility_chart(df)
        peak_chart = create_peak_markers_chart(df, behavior_stats["peak_emotions"])

        vol_chart_html = vol_chart.to_html(
            include_plotlyjs="cdn", full_html=False, config={"responsive": True}
        )
        peak_chart_html = peak_chart.to_html(
            include_plotlyjs="cdn", full_html=False, config={"responsive": True}
        )

        # Helper for timestamp formatting
        def _fmt(sec):
            m, s = divmod(int(sec), 60)
            return f"{m:02d}:{s:02d}"

        # Build peak moments HTML list
        peaks_html = ""
        if behavior_stats["peak_emotions"]:
            by_emotion = {}
            for p in behavior_stats["peak_emotions"]:
                label = p["emotion"].capitalize()
                rng = (
                    f"{_fmt(p['start'])} – {_fmt(p['end'])}"
                    if p["start"] != p["end"]
                    else f"{_fmt(p['start'])}"
                )
                by_emotion.setdefault(label, []).append(rng)
            for emo, ranges in by_emotion.items():
                peaks_html += f'<div style="margin-bottom:8px;"><strong>Peak {emo} Moments:</strong><br>'
                for r in ranges:
                    peaks_html += f'<span style="color:#10b981;margin-left:12px;">● {r}</span><br>'
                peaks_html += "</div>"
        else:
            peaks_html = (
                '<p style="color:#888;">No peak emotions detected above threshold.</p>'
            )

        # Build stress indicators HTML list
        stress_html = ""
        if behavior_stats["stress_indicators"]:
            for s in behavior_stats["stress_indicators"]:
                stress_html += (
                    f'<div style="margin-bottom:6px;">'
                    f'<span style="color:#f43f5e;">⚠</span> '
                    f'Detected at <strong>{_fmt(s["timestamp"])}</strong> – {s["detail"]}'
                    f"</div>"
                )
        else:
            stress_html = '<p style="color:#888;">No stress indicators detected.</p>'

        html_content += f"""
            <h2>🧠 Behavioral Analysis</h2>

            <div class="summary">
                <div class="stat-card">
                    <h3>Dominant Emotion</h3>
                    <div class="stat-value">{behavior_stats['dominant_emotion'].capitalize()}</div>
                    <p style="color:#bbb;margin:0;">{behavior_stats['dominant_pct']:.1f}% of all frames</p>
                </div>
                <div class="stat-card">
                    <h3>Stability Score</h3>
                    <div class="stat-value">{behavior_stats['stability']:.2f}</div>
                    <p style="color:#bbb;margin:0;">Higher is calmer</p>
                </div>
                <div class="stat-card">
                    <h3>Volatility Index</h3>
                    <div class="stat-value">{behavior_stats['volatility']:.2f}</div>
                    <p style="color:#bbb;margin:0;">Emotion switch rate</p>
                </div>
            </div>

            <div class="charts">
                <div class="chart">
                    <h3>Emotion Volatility Over Time</h3>
                    {vol_chart_html}
                </div>
                <div class="chart">
                    <h3>Emotion Timeline with Peak Markers</h3>
                    {peak_chart_html}
                </div>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:2em;">
                <div class="stat-card" style="flex:unset;min-width:unset;">
                    <h3>Peak Emotion Moments</h3>
                    {peaks_html}
                </div>
                <div class="stat-card" style="flex:unset;min-width:unset;">
                    <h3>Stress Indicators</h3>
                    {stress_html}
                </div>
            </div>
        """
    except Exception as e:
        html_content += f"""
            <h2>🧠 Behavioral Analysis</h2>
            <p style="color:#f43f5e;">Could not compute behavioral analysis: {str(e)}</p>
        """

    # ── Emotion Prediction Section in Report ──
    try:
        report_emo_seq = df["dominant_emotion"].tolist()
        if len(report_emo_seq) > 20:
            rpt_matrix = build_transition_matrix(report_emo_seq)
            rpt_next = predict_next_emotion(report_emo_seq[-1], rpt_matrix)
            rpt_future = predict_future_emotions(report_emo_seq, steps=5)
            rpt_forecast_fig = create_forecast_chart(rpt_future)
            rpt_forecast_html = rpt_forecast_fig.to_html(
                include_plotlyjs="cdn", full_html=False, config={"responsive": True}
            )

            # Build probability bars
            prob_cards = ""
            for emo, prob in list(rpt_next.items())[:5]:
                emoji = PRED_EMOJIS.get(emo, "❓")
                prob_cards += (
                    f'<div class="stat-card" style="flex:1 1 15%;min-width:120px;">'
                    f"<h3>{emoji}</h3>"
                    f'<div class="stat-value">{prob * 100:.0f}%</div>'
                    f'<p style="color:#bbb;margin:0;">{emo.capitalize()}</p>'
                    f"</div>"
                )

            rpt_top = next(iter(rpt_next))
            rpt_top_conf = rpt_next[rpt_top] * 100

            html_content += f"""
                <h2>🔮 Emotion Prediction Analysis</h2>
                <p style="text-align:center;color:#bbb;">
                    Markov chain forecast based on the observed emotion timeline.
                </p>

                <div class="summary" style="margin-bottom:1.5em;">
                    <div class="stat-card">
                        <h3>Predicted Next Emotion</h3>
                        <div class="stat-value">{PRED_EMOJIS.get(rpt_top, '')} {rpt_top.capitalize()}</div>
                        <p style="color:#bbb;margin:0;">Confidence: {rpt_top_conf:.0f}%</p>
                    </div>
                </div>

                <div class="summary" style="margin-bottom:1.5em;">
                    {prob_cards}
                </div>

                <div class="charts">
                    <div class="chart">
                        <h3>Multi-Step Emotion Forecast</h3>
                        {rpt_forecast_html}
                    </div>
                </div>
            """
    except Exception as e:
        html_content += f"""
            <h2>🔮 Emotion Prediction Analysis</h2>
            <p style="color:#f43f5e;">Could not generate prediction: {str(e)}</p>
        """

    # ── Multi-Person Analysis Section ──
    if per_person_data and len(per_person_data) > 0:
        html_content += """
            <h2>👥 Multi-Person Analysis</h2>
            <p style="color:#bbb;text-align:center;">Emotion tracking for each individual detected in the video.</p>
        """
        person_colors = [
            "#f43f5e",
            "#3b82f6",
            "#10b981",
            "#f59e0b",
            "#8b5cf6",
            "#ec4899",
            "#06b6d4",
            "#84cc16",
            "#ef4444",
            "#6366f1",
        ]
        for p_idx, (person_id, records) in enumerate(sorted(per_person_data.items())):
            color = person_colors[p_idx % len(person_colors)]
            # Build emotion sequence
            emotions_seq = [r["emotion"] for r in records]
            # Deduplicate consecutive
            deduped = [emotions_seq[0]] if emotions_seq else []
            for e in emotions_seq[1:]:
                if e != deduped[-1]:
                    deduped.append(e)
            seq_str = " → ".join(e.capitalize() for e in deduped)

            # Count emotions for this person
            from collections import Counter

            emo_counts = Counter(emotions_seq)
            total = len(emotions_seq)
            dominant = emo_counts.most_common(1)[0] if emo_counts else ("unknown", 0)

            # Build per-person emotion timeline chart
            person_rows = []
            for r in records:
                row = {"timestamp": r["timestamp"], "dominant_emotion": r["emotion"]}
                for emo_key, score in r["scores"].items():
                    row[emo_key] = score
                person_rows.append(row)
            person_df = pd.DataFrame(person_rows)

            emotion_cols = [
                "happy",
                "sad",
                "angry",
                "fear",
                "surprise",
                "neutral",
                "disgust",
            ]
            available = [c for c in emotion_cols if c in person_df.columns]

            if len(person_df) > 1 and available:
                person_fig = px.line(
                    person_df,
                    x="timestamp",
                    y=available,
                    title=f"Person {person_id + 1} – Emotion Timeline",
                    labels={"value": "Confidence (%)", "timestamp": "Time (s)"},
                    color_discrete_sequence=px.colors.sequential.Plasma,
                )
                person_fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#1a1a1a",
                    paper_bgcolor="#1a1a1a",
                    font=dict(color="white", size=12),
                    height=300,
                    margin=dict(t=40, b=40, l=40, r=20),
                )
                person_fig.update_traces(line=dict(width=2))
                person_chart = person_fig.to_html(
                    include_plotlyjs="cdn", full_html=False, config={"responsive": True}
                )
            else:
                person_chart = '<p style="color:#888;">Not enough data for a chart.</p>'

            # Emotion distribution for this person
            dist_html = ""
            for emo, cnt in emo_counts.most_common():
                pct = cnt / total * 100
                dist_html += (
                    f'<span style="display:inline-block;margin:3px 6px;padding:3px 10px;'
                    f"border-radius:12px;font-size:0.8em;"
                    f'background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.1);">'
                    f"{emo.capitalize()} {pct:.0f}%</span>"
                )

            html_content += f"""
            <div style="background:rgba(25,25,25,0.7);border-radius:12px;padding:20px;
                        margin-bottom:20px;border-left:4px solid {color};
                        border:1px solid rgba(255,255,255,0.05);">
                <h3 style="margin-top:0;">👤 Person {person_id + 1}</h3>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-bottom:15px;">
                    <div class="stat-card" style="flex:unset;min-width:unset;">
                        <h3>Dominant Emotion</h3>
                        <div class="stat-value">{dominant[0].capitalize()}</div>
                        <p style="color:#bbb;margin:0;">{dominant[1] / total * 100:.1f}% of frames</p>
                    </div>
                    <div class="stat-card" style="flex:unset;min-width:unset;">
                        <h3>Frames Tracked</h3>
                        <div class="stat-value">{total}</div>
                    </div>
                </div>
                <div style="margin-bottom:10px;">
                    <strong style="color:#ccc;">Emotion Sequence:</strong>
                    <p style="color:#aaa;font-size:0.9em;">{seq_str}</p>
                </div>
                <div style="margin-bottom:15px;">{dist_html}</div>
                <div class="chart">{person_chart}</div>
            </div>
            """

    html_content += """
            <h2 class="key-frames-title">Key Emotional Moments</h2>
            <div class="key-frames">
    """

    # Add key frames to the report
    for frame in key_frame_images:
        html_content += f"""
                <div class="key-frame">
                    <img src="data:image/jpeg;base64,{frame['image']}" alt="Key frame">
                    <div class="key-frame-info">
                        <div class="emotion-label">{emotion_emojis.get(frame['emotion'], '😊')} {frame['emotion'].capitalize()}</div>
                        <div class="timeline-label">at {frame['timestamp']:.2f} seconds</div>
                    </div>
                </div>
        """

    html_content += """
            </div>
            
            <footer>
                <p>© 2023 Emotion AI | This report was automatically generated</p>
            </footer>
        </div>
        
        <script>
            // Add script to ensure Plotly charts are fully visible
            window.onload = function() {
                setTimeout(function() {
                    if (window.Plotly) {
                        var graphs = document.querySelectorAll('.plotly-graph-div');
                        graphs.forEach(function(graph) {
                            Plotly.relayout(graph, {});
                        });
                    }
                }, 500);
            }
        </script>
    </body>
    </html>
    """

    return html_content


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

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["Live Camera", "Upload Image", "Video Analysis"])

with tab1:
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
        - When you stop the video, you'll be asked if you want to save it
        - Enable **Voice Emotion Detection** to combine facial + voice analysis
        """
        )

    # --- Voice emotion toggle ---
    voice_enabled = st.toggle("🎙️ Enable Voice Emotion Detection", value=False)

    # Initialize session state for video handling
    if "was_playing" not in st.session_state:
        st.session_state.was_playing = False
    if "recorder" not in st.session_state:
        st.session_state.recorder = None
    if "video_saved" not in st.session_state:
        st.session_state.video_saved = False
    if "frames_buffer" not in st.session_state:
        st.session_state.frames_buffer = []

    # Initialize voice/fusion session state
    if "voice_emotion" not in st.session_state:
        st.session_state.voice_emotion = None
    if "voice_scores" not in st.session_state:
        st.session_state.voice_scores = None
    if "voice_thread_running" not in st.session_state:
        st.session_state.voice_thread_running = False

    # WebRTC component
    st.markdown('<div class="video-container">', unsafe_allow_html=True)

    # Create a new detector only if one doesn't exist or if we're starting fresh
    if "emotion_detector" not in st.session_state:
        st.session_state.emotion_detector = EmotionDetector()

    emotion_detector = st.session_state.emotion_detector

    # Setting up webrtc with recording capability
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        video_transformer_factory=lambda: emotion_detector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Handle recording state - with better logging
    if webrtc_ctx.state and webrtc_ctx.state.playing:
        if not st.session_state.was_playing:  # Only start if we were not playing before
            st.info("Starting video recording...")
            emotion_detector.start_recording()
            st.session_state.recorder = emotion_detector

    # Check if the stream was just stopped
    if st.session_state.was_playing and not (
        webrtc_ctx.state and webrtc_ctx.state.playing
    ):
        # Stream was stopped
        st.info("Video stream stopped.")

        if "recorder" in st.session_state and st.session_state.recorder:
            recorder = st.session_state.recorder

            # Stop recording explicitly
            if recorder.is_recording:
                recorder.stop_recording()

            # Get frames
            processed_frames = recorder.get_processed_frames()
            raw_frames = recorder.get_raw_frames()

            processed_frames_count = len(processed_frames)
            raw_frames_count = len(raw_frames)

            st.write(f"Frames captured: {processed_frames_count}")

            if processed_frames_count > 0:
                # First, ask if they want to save the video
                save_option = st.radio(
                    "Would you like to save the recorded video?", ("Yes", "No")
                )

                if save_option == "Yes":
                    # Then, ask which version they want to save
                    video_type = st.radio(
                        "Which version would you like to save?",
                        (
                            "Processed video (with emotion labels)",
                            "Raw video (without labels)",
                        ),
                    )

                    # Generate a timestamp for the filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    if video_type == "Processed video (with emotion labels)":
                        # Save the processed video
                        filename = f"emotion_detection_{timestamp}.mp4"
                        frames_to_save = processed_frames
                        st.session_state.frames_buffer = frames_to_save.copy()

                        with st.spinner(
                            f"Saving processed video with {processed_frames_count} frames..."
                        ):
                            try:
                                video_path = save_video(frames_to_save, filename)
                                if video_path and os.path.exists(video_path):
                                    st.success(
                                        f"Video saved successfully as {filename}"
                                    )
                                    st.markdown(
                                        get_binary_file_downloader_html(
                                            video_path, "Download Processed Video"
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                    st.session_state.video_saved = True
                                else:
                                    st.error("Failed to save video. Please try again.")
                            except Exception as e:
                                st.error(f"Error during video saving: {str(e)}")
                    else:
                        # Save the raw video
                        filename = f"raw_video_{timestamp}.mp4"
                        frames_to_save = raw_frames
                        st.session_state.frames_buffer = frames_to_save.copy()

                        with st.spinner(
                            f"Saving raw video with {raw_frames_count} frames..."
                        ):
                            try:
                                video_path = save_video(frames_to_save, filename)
                                if video_path and os.path.exists(video_path):
                                    st.success(
                                        f"Raw video saved successfully as {filename}"
                                    )
                                    st.markdown(
                                        get_binary_file_downloader_html(
                                            video_path, "Download Raw Video"
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                    st.session_state.video_saved = True
                                else:
                                    st.error("Failed to save video. Please try again.")
                            except Exception as e:
                                st.error(f"Error during video saving: {str(e)}")
            else:
                st.warning(
                    "No frames were recorded. This might be due to browser or permission issues."
                )

                # Offer a test video option
                if st.button("Generate a test video instead"):
                    with st.spinner("Generating test video..."):
                        test_frames = create_test_video(100)
                        filename = (
                            f"test_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                        )
                        video_path = save_video(test_frames, filename)
                        if video_path:
                            st.success(f"Test video created as {filename}")
                            st.markdown(
                                get_binary_file_downloader_html(
                                    video_path, "Download Test Video"
                                ),
                                unsafe_allow_html=True,
                            )

    # Update the was_playing state
    if webrtc_ctx.state:
        st.session_state.was_playing = webrtc_ctx.state.playing
    else:
        st.session_state.was_playing = False

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # MULTIMODAL EMOTION DISPLAY
    # ------------------------------------------------------------------
    # Run one voice analysis cycle when toggle is on and camera is playing
    if voice_enabled and webrtc_ctx.state and webrtc_ctx.state.playing:
        if st.button("🎤 Capture Voice Emotion (3 s)"):
            with st.spinner("Recording audio… speak now!"):
                try:
                    audio, sr = record_audio_chunk(duration=3)
                    features = extract_audio_features(audio, sr)
                    v_emotion, v_scores = predict_voice_emotion(features)
                    st.session_state.voice_emotion = v_emotion
                    st.session_state.voice_scores = v_scores
                except Exception as e:
                    st.error(f"Voice capture error: {e}")

    # Show multimodal results when we have data
    if webrtc_ctx.state and webrtc_ctx.state.playing:
        face_emo = emotion_detector.last_emotion
        face_scr = (
            emotion_detector.emotion_scores if emotion_detector.emotion_scores else None
        )
        voice_emo = st.session_state.voice_emotion if voice_enabled else None
        voice_scr = st.session_state.voice_scores if voice_enabled else None

        # Compute fusion
        if face_emo or voice_emo:
            final_emo, final_scr = combine_emotions(
                face_emo, face_scr, voice_emo, voice_scr
            )

            # Build display cards
            face_emoji = emotion_emojis.get(face_emo, "❓") if face_emo else "❓"
            face_label = face_emo.capitalize() if face_emo else "Waiting…"
            face_conf = (
                f"{face_scr.get(face_emo, 0):.1f}%" if face_scr and face_emo else "—"
            )

            voice_emoji = emotion_emojis.get(voice_emo, "❓") if voice_emo else "🔇"
            voice_label = (
                voice_emo.capitalize()
                if voice_emo
                else ("Disabled" if not voice_enabled else "Press Capture")
            )
            voice_conf = (
                f"{voice_scr.get(voice_emo, 0):.1f}%"
                if voice_scr and voice_emo
                else "—"
            )

            final_emoji = emotion_emojis.get(final_emo, "❓")
            final_label = final_emo.capitalize()
            final_conf = f"{final_scr.get(final_emo, 0):.1f}%"

            st.markdown(
                f"""
                <div class="multimodal-container">
                    <div class="emotion-card">
                        <div class="card-label">👁️ Face</div>
                        <div class="card-emoji">{face_emoji}</div>
                        <div class="card-emotion">{face_label}</div>
                        <div class="card-confidence">{face_conf}</div>
                    </div>
                    <div class="emotion-card">
                        <div class="card-label">🎙️ Voice</div>
                        <div class="card-emoji">{voice_emoji}</div>
                        <div class="card-emotion">{voice_label}</div>
                        <div class="card-confidence">{voice_conf}</div>
                    </div>
                    <div class="emotion-card fused">
                        <div class="card-label">✨ Final</div>
                        <div class="card-emoji">{final_emoji}</div>
                        <div class="card-emotion">{final_label}</div>
                        <div class="card-confidence">{final_conf}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Multi-Person Tracking Display (Live Camera) ---
    if webrtc_ctx.state and webrtc_ctx.state.playing:
        person_history = emotion_detector.person_emotion_history
        if person_history and len(person_history) > 0:
            with st.expander(
                f"👥 Multi-Person Tracking ({len(person_history)} people)",
                expanded=True,
            ):
                for person_id in sorted(person_history.keys()):
                    records = person_history[person_id]
                    if not records:
                        continue
                    # Show last 10 emotions, deduplicated consecutive
                    recent = records[-20:]
                    emotions_seq = [
                        r[1] for r in recent
                    ]  # (timestamp, emotion, scores)
                    deduped = [emotions_seq[0]]
                    for e in emotions_seq[1:]:
                        if e != deduped[-1]:
                            deduped.append(e)
                    seq_str = " → ".join(e.capitalize() for e in deduped[-8:])
                    last_emo = records[-1][1]
                    emoji = emotion_emojis.get(last_emo, "❓")

                    st.markdown(
                        f'<div style="background:rgba(39,39,39,0.4);border-radius:8px;'
                        f'padding:10px 15px;margin-bottom:8px;border:1px solid rgba(255,255,255,0.05);">'
                        f'<span style="font-weight:600;">👤 Person {person_id + 1}</span> '
                        f'<span style="margin-left:8px;">{emoji} {last_emo.capitalize()}</span>'
                        f'<br><span style="color:#aaa;font-size:0.85em;">{seq_str}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # --- Emotion Forecast Panel (Live Camera) ---
    if webrtc_ctx.state and webrtc_ctx.state.playing:
        # Gather all emotions from every tracked person into a flat sequence
        person_history = emotion_detector.person_emotion_history
        all_emotions_seq: list[str] = []
        if person_history:
            for pid in sorted(person_history.keys()):
                for _ts, emo, _scores in person_history[pid]:
                    all_emotions_seq.append(emo)

        if len(all_emotions_seq) > 20:
            pred_matrix = build_transition_matrix(all_emotions_seq)
            next_probs = predict_next_emotion(all_emotions_seq[-1], pred_matrix)
            top_emo = next(iter(next_probs))
            top_conf = next_probs[top_emo] * 100

            # Build probability bars HTML
            bars_html = ""
            for emo, prob in list(next_probs.items())[:5]:
                emoji = emotion_emojis.get(emo, "❓")
                bars_html += (
                    f'<div class="forecast-bar">'
                    f'<div class="fb-emoji">{emoji}</div>'
                    f'<div class="fb-pct">{prob * 100:.0f}%</div>'
                    f'<div class="fb-emotion">{emo.capitalize()}</div>'
                    f"</div>"
                )

            st.markdown(
                f"""
                <div class="forecast-panel">
                    <div class="forecast-title">🔮 Emotion Forecast</div>
                    <div style="margin-bottom:8px;color:#ccc;font-size:0.9rem;">
                        Predicted next emotion:
                        <strong style="color:#c4b5fd;">{emotion_emojis.get(top_emo, '❓')} {top_emo.capitalize()}</strong>
                        &nbsp;·&nbsp; Confidence <strong style="color:#c4b5fd;">{top_conf:.0f}%</strong>
                    </div>
                    <div class="forecast-grid">{bars_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab2:
    # Upload image section
    st.markdown(
        """
        <div class="status-bar">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>Image Analysis Ready</span>
            </div>
            <div>Upload & Analyze</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ How to Use", expanded=False):
        st.markdown(
            """
        - Upload an image containing one or more faces
        - The AI will detect and analyze emotions in the image
        - Results will be displayed below the image
        """
        )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        with st.spinner("Analyzing image..."):
            result_image, emotion_results = analyze_image(image)

        # Display the image with emotion detection
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.image(result_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Display results
        if emotion_results:
            st.markdown("<h3>Detected Emotions</h3>", unsafe_allow_html=True)

            for i, result in enumerate(emotion_results):
                st.markdown(
                    f"""
                    <div style="background: rgba(39, 39, 39, 0.4); padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.05);">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <div style="font-size: 1.5rem; margin-right: 10px;">{emotion_emojis.get(result['emotion'], '😊')}</div>
                            <div>
                                <div style="font-weight: bold;">{result['emotion'].capitalize()}</div>
                                <div>Confidence: {result['confidence']:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No faces detected in the image. Please try another image.")

with tab3:
    # Video Analysis section
    st.markdown(
        """
        <div class="status-bar">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>Video Analysis Ready</span>
            </div>
            <div>Upload & Analyze Video</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ How to Use", expanded=False):
        st.markdown(
            """
        - Upload a video file containing one or more faces
        - Click 'Analyze Video' to process the emotions throughout the video
        - The AI will sample frames at regular intervals
        - Results will be displayed as charts and key moments
        - You can download a detailed report of the analysis
        """
        )

    # File uploader for video
    uploaded_video = st.file_uploader(
        "Upload a video for emotion analysis...", type=["mp4", "mov", "avi"]
    )

    if uploaded_video is not None:
        # Create a temporary file to store the uploaded video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name
        temp_file.close()

        # Display the uploaded video
        st.video(uploaded_video)

        # Settings for video analysis
        st.markdown("### Analysis Settings")
        col1, col2 = st.columns(2)
        with col1:
            sampling_rate = st.slider(
                "Sampling rate (frames per second)",
                min_value=0.5,
                max_value=10.0,
                value=1.0,
                step=0.5,
                help="Higher values process more frames but take longer",
            )
        with col2:
            st.markdown("")  # Placeholder for alignment
            include_report = st.checkbox("Generate detailed report", value=True)

        # Analyze Video button
        if st.button("Analyze Video"):
            with st.spinner(
                "Analyzing video emotions... This may take several minutes depending on video length."
            ):
                # Process the video
                emotion_df, key_frames, summary, per_person_data = (
                    process_video_for_emotions(video_path, sampling_rate)
                )

                if emotion_df is not None and not emotion_df.empty:
                    # Store results in session state
                    st.session_state.emotion_df = emotion_df
                    st.session_state.key_frames = key_frames
                    st.session_state.summary = summary
                    st.session_state.per_person_data = per_person_data
                    st.session_state.has_video_results = True
                else:
                    st.error(
                        "No emotions could be detected in the video. Please try a different video with visible faces."
                    )
                    # Clear any previous results
                    st.session_state.has_video_results = False

        # Display results if available
        if st.session_state.get("has_video_results", False):
            emotion_df = st.session_state.emotion_df
            key_frames = st.session_state.key_frames
            summary = st.session_state.summary

            st.markdown("## Analysis Results")

            # Summary statistics
            st.markdown("### Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Video Duration", f"{summary['duration']:.1f}s")
            with col2:
                st.metric("Frames Analyzed", f"{len(emotion_df)}")
            with col3:
                dominant_emotion = max(
                    summary["emotion_percentages"].items(), key=lambda x: x[1]
                )[0]
                dominant_pct = summary["emotion_percentages"][dominant_emotion]
                st.metric(
                    "Dominant Emotion",
                    f"{dominant_emotion.capitalize()} ({dominant_pct:.1f}%)",
                )

            # Emotion distribution
            st.markdown("### Emotion Distribution")
            fig = px.pie(
                values=list(summary["emotion_percentages"].values()),
                names=list(summary["emotion_percentages"].keys()),
                title="Proportion of Emotions",
                color_discrete_sequence=px.colors.sequential.Plasma,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Emotion timeline
            st.markdown("### Emotion Timeline")
            timeline_fig = px.line(
                emotion_df,
                x="timestamp",
                y=["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"],
                title="Emotion Confidence Over Time",
                labels={"value": "Confidence (%)", "timestamp": "Time (seconds)"},
            )
            st.plotly_chart(timeline_fig, use_container_width=True)

            # Key frames
            if key_frames:
                st.markdown("### Key Emotional Moments")
                columns = st.columns(4)
                for i, frame_data in enumerate(key_frames):
                    col_idx = i % 4
                    with columns[col_idx]:
                        st.image(frame_data["frame"], use_container_width=True)
                        st.markdown(
                            f"**{frame_data['emotion'].capitalize()}** at {frame_data['timestamp']:.2f}s"
                        )

            # ── Behavioral analysis section in Streamlit UI ──
            st.markdown("### 🧠 Behavioral Analysis")
            try:
                behavior_stats = compute_emotion_statistics(emotion_df)

                # Profile metrics row
                bcol1, bcol2, bcol3 = st.columns(3)
                with bcol1:
                    st.metric(
                        "Dominant Emotion",
                        f"{behavior_stats['dominant_emotion'].capitalize()} ({behavior_stats['dominant_pct']:.1f}%)",
                    )
                with bcol2:
                    st.metric("Stability Score", f"{behavior_stats['stability']:.2f}")
                with bcol3:
                    st.metric("Volatility Index", f"{behavior_stats['volatility']:.2f}")

                # Volatility chart
                vol_fig = create_volatility_chart(emotion_df)
                st.plotly_chart(vol_fig, use_container_width=True)

                # Peak emotion markers chart
                peak_fig = create_peak_markers_chart(
                    emotion_df, behavior_stats["peak_emotions"]
                )
                st.plotly_chart(peak_fig, use_container_width=True)

                # Peak moments expander
                with st.expander("📈 Peak Emotion Moments", expanded=False):
                    peaks = behavior_stats["peak_emotions"]
                    if peaks:
                        for p in peaks:
                            m1, s1 = divmod(int(p["start"]), 60)
                            m2, s2 = divmod(int(p["end"]), 60)
                            ts_str = (
                                f"{m1:02d}:{s1:02d} – {m2:02d}:{s2:02d}"
                                if p["start"] != p["end"]
                                else f"{m1:02d}:{s1:02d}"
                            )
                            st.markdown(
                                f"**{p['emotion'].capitalize()}** at {ts_str} "
                                f"(max confidence {p['max_confidence']:.1f}%)"
                            )
                    else:
                        st.info("No peak emotions detected above the 80% threshold.")

                # Stress indicators expander
                with st.expander("⚠️ Stress Indicators", expanded=False):
                    stress = behavior_stats["stress_indicators"]
                    if stress:
                        for s in stress:
                            m, sec = divmod(int(s["timestamp"]), 60)
                            st.markdown(
                                f"Detected at **{m:02d}:{sec:02d}** – {s['detail']}"
                            )
                    else:
                        st.success("No stress indicators detected.")

            except Exception as e:
                st.warning(f"Behavioral analysis unavailable: {str(e)}")

            # ── Emotion Prediction / Forecast Section ──
            emo_seq = emotion_df["dominant_emotion"].tolist()
            if len(emo_seq) > 20:
                st.markdown("### 🔮 Emotion Forecast")
                try:
                    pred_matrix = build_transition_matrix(emo_seq)
                    next_probs = predict_next_emotion(emo_seq[-1], pred_matrix)
                    future_preds = predict_future_emotions(emo_seq, steps=5)

                    top_emo = next(iter(next_probs))
                    top_conf = next_probs[top_emo] * 100

                    fcol1, fcol2, fcol3 = st.columns(3)
                    with fcol1:
                        st.metric("Predicted Next Emotion", f"{top_emo.capitalize()}")
                    with fcol2:
                        st.metric("Confidence", f"{top_conf:.1f}%")
                    with fcol3:
                        second_emo = (
                            list(next_probs.keys())[1] if len(next_probs) > 1 else "—"
                        )
                        second_conf = (
                            list(next_probs.values())[1] * 100
                            if len(next_probs) > 1
                            else 0
                        )
                        st.metric(
                            "Runner-up",
                            f"{second_emo.capitalize()} ({second_conf:.0f}%)",
                        )

                    # Probability breakdown
                    with st.expander("📊 Next-Emotion Probabilities", expanded=True):
                        for emo, prob in next_probs.items():
                            emoji = emotion_emojis.get(emo, "❓")
                            st.markdown(
                                f"{emoji} **{emo.capitalize()}** — {prob * 100:.1f}%"
                            )

                    # Forecast chart
                    forecast_fig = create_forecast_chart(future_preds)
                    st.plotly_chart(forecast_fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Emotion forecast unavailable: {str(e)}")
            else:
                st.info(
                    "Emotion forecast requires more than 20 data frames. Not enough data yet."
                )

            # ── Multi-Person Analysis Section in Streamlit UI ──
            per_person_data = st.session_state.get("per_person_data", {})
            if per_person_data and len(per_person_data) > 0:
                st.markdown("### 👥 Multi-Person Analysis")
                st.markdown(
                    f'<p style="color:#bbb;">{len(per_person_data)} individual(s) tracked across the video.</p>',
                    unsafe_allow_html=True,
                )

                person_colors = [
                    "#f43f5e",
                    "#3b82f6",
                    "#10b981",
                    "#f59e0b",
                    "#8b5cf6",
                    "#ec4899",
                    "#06b6d4",
                    "#84cc16",
                    "#ef4444",
                    "#6366f1",
                ]

                for p_idx, (person_id, records) in enumerate(
                    sorted(per_person_data.items())
                ):
                    color = person_colors[p_idx % len(person_colors)]
                    with st.expander(
                        f"👤 Person {person_id + 1} ({len(records)} frames)",
                        expanded=(p_idx < 3),
                    ):
                        # Emotion sequence (deduplicated consecutive)
                        emotions_seq = [r["emotion"] for r in records]
                        deduped = [emotions_seq[0]] if emotions_seq else []
                        for e in emotions_seq[1:]:
                            if e != deduped[-1]:
                                deduped.append(e)
                        seq_str = " → ".join(e.capitalize() for e in deduped)
                        st.markdown(f"**Emotion Flow:** {seq_str}")

                        # Per-person stats
                        from collections import Counter

                        emo_counts = Counter(emotions_seq)
                        total = len(emotions_seq)
                        dominant = (
                            emo_counts.most_common(1)[0]
                            if emo_counts
                            else ("unknown", 0)
                        )

                        pcol1, pcol2 = st.columns(2)
                        with pcol1:
                            st.metric(
                                "Dominant Emotion",
                                f"{dominant[0].capitalize()} ({dominant[1] / total * 100:.1f}%)",
                            )
                        with pcol2:
                            st.metric("Frames Tracked", f"{total}")

                        # Per-person timeline chart
                        if len(records) > 1:
                            person_rows = []
                            for r in records:
                                row = {
                                    "timestamp": r["timestamp"],
                                    "dominant_emotion": r["emotion"],
                                }
                                for emo_key, score in r["scores"].items():
                                    row[emo_key] = score
                                person_rows.append(row)
                            person_df = pd.DataFrame(person_rows)

                            emotion_cols = [
                                "happy",
                                "sad",
                                "angry",
                                "fear",
                                "surprise",
                                "neutral",
                                "disgust",
                            ]
                            available = [
                                c for c in emotion_cols if c in person_df.columns
                            ]
                            if available:
                                pfig = px.line(
                                    person_df,
                                    x="timestamp",
                                    y=available,
                                    title=f"Person {person_id + 1} – Emotion Timeline",
                                    labels={
                                        "value": "Confidence (%)",
                                        "timestamp": "Time (s)",
                                    },
                                    color_discrete_sequence=px.colors.sequential.Plasma,
                                )
                                pfig.update_traces(line=dict(width=2))
                                st.plotly_chart(pfig, use_container_width=True)

            # Generate and offer report download
            if include_report:
                with st.spinner("Generating detailed report..."):
                    report_html = generate_emotion_report(
                        emotion_df,
                        key_frames,
                        summary,
                        per_person_data=st.session_state.get("per_person_data"),
                    )

                    if report_html:
                        # Create a download button for the report
                        report_filename = f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        st.download_button(
                            label="📊 Download Detailed Report",
                            data=report_html,
                            file_name=report_filename,
                            mime="text/html",
                            help="Download a detailed HTML report of the emotion analysis",
                        )

            # Clean up the temp file
            try:
                os.unlink(video_path)
            except:
                pass

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


# Add a simple function to make a fake video for testing
def create_test_video(num_frames=100, width=640, height=480):
    frames = []
    for i in range(num_frames):
        # Create a colored frame with frame number
        frame = np.ones((height, width, 3), dtype=np.uint8) * (i % 255, 100, 200)
        # Add text with frame number
        cv2.putText(
            frame,
            f"Frame {i}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        frames.append(frame)
    return frames
