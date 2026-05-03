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
from datetime import datetime
import av

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
        # For video recording
        self.raw_frames = []  # For original frames
        self.processed_frames = []  # For frames with emotion outlines and labels
        self.is_recording = False
        self.frame_count = 0  # Add a frame counter for debugging

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
tab1, tab2 = st.tabs(["Live Camera", "Upload Image"])

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
        """
        )

    # Initialize session state for video handling
    if "was_playing" not in st.session_state:
        st.session_state.was_playing = False
    if "recorder" not in st.session_state:
        st.session_state.recorder = None
    if "video_saved" not in st.session_state:
        st.session_state.video_saved = False
    if "frames_buffer" not in st.session_state:
        st.session_state.frames_buffer = []

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

    # Debugging buttons
    # with st.expander("Debug Options", expanded=False):
    #     if st.button("Check Frame Count"):
    #         if "emotion_detector" in st.session_state:
    #             detector = st.session_state.emotion_detector
    #             st.write(f"Current frame count: {len(detector.get_processed_frames())}")
    #             st.write(f"Is recording: {detector.is_recording}")
    #             st.write(f"Frames in session: {len(st.session_state.frames_buffer)}")

    #     if st.button("Generate Test Video"):
    #         with st.spinner("Generating test video..."):
    #             test_frames = create_test_video(100)
    #             filename = f"test_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    #             video_path = save_video(test_frames, filename)
    #             if video_path:
    #                 st.success(f"Test video created as {filename}")
    #                 st.markdown(
    #                     get_binary_file_downloader_html(
    #                         video_path, "Download Test Video"
    #                     ),
    #                     unsafe_allow_html=True,
    #                 )

    # Update the was_playing state
    if webrtc_ctx.state:
        st.session_state.was_playing = webrtc_ctx.state.playing
    else:
        st.session_state.was_playing = False

    st.markdown("</div>", unsafe_allow_html=True)

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
