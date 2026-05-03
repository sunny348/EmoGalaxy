# Emotion AI

A real-time emotion detection application that analyzes facial expressions using webcam feed, image uploads, and video analysis.

## 👨‍💻 Project Information

### Project Type
**Copyright Project**

### Team Members

| Name | Roll Number |
|---|---|
| Sanpreet Singh | 2210990790 |
| Sanidhya Chauhan | 2210990786 |
| Sai Madhav Bhalla | 2210990760 |
| Sambhav Yadav | 2210990774 |

---

## 📚 Table of Contents

- [Emotion AI](#emotion-ai)
  - [📚 Table of Contents](#-table-of-contents)
  - [🚀 Features](#-features)
    - [Version 1 (app.py)](#version-1-apppy)
    - [Version 2 (appv2.py)](#version-2-appv2py)
    - [Version 3 (appv3.py)](#version-3-appv3py)
    - [Version 4 (appv4.py) — Latest](#version-4-appv4py--latest)
  - [🗂️ Project Structure](#️-project-structure)
  - [🔧 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [💻 Usage](#-usage)
  - [🔬 Technologies](#-technologies)
  - [📊 Architecture Diagrams](#-architecture-diagrams)
  - [📜 License](#-license)

## 🚀 Features

### Version 1 (app.py)

The initial version provides real-time emotion detection through webcam:

- **Live Webcam Analysis**: Detects and analyzes emotions in real-time
- **Temporal Smoothing**: Reduces rapid emotion switching with a history buffer
- **Confidence Display**: Shows confidence percentage for detected emotions
- **Modern UI**: Dark-themed gradient interface with responsive design
- **Emotion Visualization**: Labels faces with detected emotions and confidence scores

### Version 2 (appv2.py)

Building on version 1, adds:

- **Tabbed Interface**: Separates Live Camera and Image Upload functionality
- **Image Upload Analysis**: Upload and analyze images for emotion detection
- **Video Recording**: Records the webcam feed for later use
- **Video Saving Options**: Save processed videos (with emotion labels) or raw videos
- **Download Functionality**: Download recorded videos in MP4 format
- **Error Handling**: Improved error management and user feedback
- **Test Video Generation**: Create test videos for debugging purposes

### Version 3 (appv3.py)

The most advanced version with comprehensive features:

- **Video Analysis Tab**: Upload and analyze videos for emotion detection
- **Sampling Rate Control**: Adjust the frame sampling rate for video analysis
- **Emotion Timeline**: Visual representation of emotions over time
- **Emotion Distribution**: Pie chart showing the proportion of different emotions
- **Key Emotional Moments**: Captures and displays significant emotional changes
- **Detailed Reports**: Generate and download HTML reports with visualizations
- **Summary Statistics**: Duration, frames analyzed, and dominant emotions
- **Interactive Charts**: Colorful, responsive charts powered by Plotly
- **Improved Styling**: Enhanced report design with responsive layout

### Version 4 (appv4.py) — Latest

Extends v3 with **multimodal emotion detection**, **behavioral analysis**, **multi-person face tracking**, and **emotion prediction**:

#### 🎙️ Voice Emotion Detection (`audio_emotion.py`)
- Toggleable microphone input in the Live Camera tab
- Records 3-second audio clips on demand via the **🎤 Capture Voice Emotion** button
- Extracts acoustic features using `librosa`:
  - **MFCCs** (13 coefficients) — timbral texture
  - **Pitch** (mean & variance via `pyin`) — fundamental frequency
  - **RMS Energy** — loudness
  - **Spectral Contrast** — harmonic structure
  - **Zero Crossing Rate** — voiced/unvoiced signal
- Heuristic classifier maps features to 6 emotions: `happy`, `sad`, `angry`, `fear`, `neutral`, `surprise`

#### 🤝 Multimodal Emotion Fusion (`emotion_fusion.py`)
- Combines face and voice detections via **weighted averaging**:
  ```
  Final = Face × 0.65 + Voice × 0.35
  ```
- Maps DeepFace's 7-class output (including "disgust") onto the shared 6-class set
- Falls back gracefully when one modality is unavailable
- **Three-card display** in the Live Camera tab: Face Emotion | Voice Emotion | Final Emotion

#### 🧠 Behavioral Analysis (`emotion_analysis.py`)
- **Volatility Index**: Emotion switch rate — how often the dominant emotion changes
- **Stability Score**: `1 - volatility` — higher means calmer
- **Peak Emotion Detection**: Identifies moments where confidence exceeds 80%
- **Stress Indicator Detection**: Flags rapid-switching and known stress patterns (e.g. angry → fear → sad)
- **Behavioral Summary**: Human-readable text report of all metrics
- **Two new Plotly charts** in the Video Analysis tab and HTML report:
  - Rolling emotion volatility over time
  - Emotion confidence timeline with peak moment markers

#### 👥 Multi-Person Face Tracking (`face_tracker.py`)
- Assigns **persistent IDs** to detected faces across frames using centroid-based matching
- Uses `scipy.spatial.distance.cdist` for efficient O(n²) Hungarian-style assignment
- Configurable `max_disappeared` threshold (default: 15 frames) before an ID is dropped
- **Live Camera tab**: each face is labeled `P1: Happy (85%)`, `P2: Neutral (72%)`, etc.
  - 👥 Multi-Person Tracking expander shows per-person emotion flow in real time
- **Video Analysis tab**: new 👥 Multi-Person Analysis section with:
  - Per-person emotion timeline charts (interactive Plotly)
  - Dominant emotion & frame count per person
  - Deduplicated emotion sequence (e.g., `Happy → Neutral → Angry`)
- **HTML Report**: 👥 Multi-Person Analysis section with per-person cards, charts, and emotion distribution badges
- Maintains `person_emotion_history` dictionary: `{person_id: [(timestamp, emotion, scores), ...]}`

#### 🔮 Emotion Prediction (`emotion_prediction.py`)
- **Markov Chain Model**: Builds a first-order transition-probability matrix from observed emotion sequences
- `build_transition_matrix(sequence)` — counts consecutive emotion pairs and row-normalises
- `predict_next_emotion(current, matrix)` — returns `{emotion: probability}` sorted by confidence
- `predict_future_emotions(sequence, steps=5)` — multi-step iterative forecast
- `create_forecast_chart(predictions)` — Plotly grouped-bar visualisation of future probabilities
- **Activation threshold**: Prediction only runs when the emotion timeline exceeds **20 frames**
- **Live Camera tab**: "🔮 Emotion Forecast" panel with predicted next emotion, confidence %, and top-5 probability bars
- **Video Analysis tab**: Forecast metrics (predicted emotion, confidence, runner-up), expandable probability breakdown, and multi-step forecast chart
- **HTML Report**: "🔮 Emotion Prediction Analysis" section with stat cards and embedded Plotly chart

---

## 🗂️ Project Structure

```
emotion-python/
├── app.py                # Version 1 — live webcam detection
├── appv2.py              # Version 2 — + image upload & recording
├── appv3.py              # Version 3 — + video analysis & reports
├── appv4.py              # Version 4 — + voice, behavioral, tracking & prediction (← current)
│
├── audio_emotion.py      # Voice recording & emotion prediction via librosa
├── emotion_fusion.py     # Weighted face + voice emotion fusion
├── emotion_analysis.py   # Behavioral metrics, volatility, stress detection
├── face_tracker.py       # Centroid-based multi-person face tracker
├── emotion_prediction.py # Markov chain emotion forecasting
│
├── requirements.txt      # Pinned dependency list
└── README.md
```

---

## 🔧 Installation

### Prerequisites

- Python 3.11
- UV package manager

### Setup

1. **Install UV Package Manager**:

   For Windows:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   For macOS/Linux:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create and Activate Virtual Environment**:

   For Windows:

   ```powershell
   uv venv myenv --python=3.11
   myenv\Scripts\activate
   ```

   For macOS/Linux:

   ```bash
   uv venv myenv --python=3.11
   source myenv/bin/activate
   ```

3. **Install Core Dependencies**:
   ```
   uv pip install streamlit opencv-python-headless streamlit-webrtc deepface
   uv pip install numpy==1.26.3
   uv pip install matplotlib plotly pandas
   ```

4. **Install v4 Additional Dependencies** (for voice + behavioral analysis):
   ```
   uv pip install librosa sounddevice scikit-learn tf-keras
   ```

## 💻 Usage

Run any version of the application using Streamlit:

```
streamlit run app.py      # For version 1
streamlit run appv2.py    # For version 2
streamlit run appv3.py    # For version 3
streamlit run appv4.py    # For version 4 (recommended)
```

### Version 1: Live Webcam Emotion Detection

- Grant camera access when prompted
- Position your face in the frame
- Ensure good lighting for better detection
- See real-time emotion analysis with confidence scores

### Version 2: Webcam + Image Upload Analysis

- **Live Camera Tab**: Same as Version 1, plus video recording
- **Upload Image Tab**: Upload images for emotion analysis
- When stopping the webcam feed, choose to save the video with emotion labels

### Version 3: Complete Emotion Analysis Suite

- **Live Camera Tab**: Same as Version 2
- **Upload Image Tab**: Same as Version 2
- **Video Analysis Tab**: Upload videos for comprehensive emotion analysis
  - Adjust sampling rate for processing speed vs. accuracy
  - View emotion distribution, timeline, and key moments
  - Download detailed HTML reports with interactive visualizations

### Version 4: Multimodal Emotion Analysis Suite

#### Live Camera Tab
- Grant camera access when prompted
- Position your face in the frame with good lighting
- Toggle **🎙️ Enable Voice Emotion Detection** to activate microphone input
- Click **🎤 Capture Voice Emotion (3 s)** while speaking to record a sample
- Three emotion cards appear below the feed: **Face** | **Voice** | **Final (Fused)**

#### Upload Image Tab
- Upload a JPG/PNG image with one or more faces
- AI detects and labels emotions per face with confidence scores

#### Video Analysis Tab
- Upload an MP4/MOV/AVI video and set the sampling rate
- Click **Analyze Video** to process
- View: emotion distribution pie chart, confidence timeline, key moments
- Behavioral Analysis section shows:
  - Stability Score & Volatility Index metrics
  - Rolling volatility chart
  - Peak emotion markers chart
  - Peak emotion moments (timestamps where confidence > 80%)
  - Stress indicators (rapid-switching and stress-pattern detection)
- **🔮 Emotion Forecast** section (when > 20 frames of data):
  - Predicted next emotion with confidence score and runner-up
  - Expandable probability breakdown for all emotions
  - Multi-step forecast chart (grouped-bar visualisation)
- **👥 Multi-Person Analysis** section (when multiple faces are detected):
  - Per-person expanders showing emotion flow and timeline charts
  - Dominant emotion and frame count per tracked individual
- Download a detailed **HTML report** with all charts, behavioral data, prediction analysis, and per-person analysis embedded

---

## 🔬 Technologies

| Library | Purpose |
|---|---|
| **Streamlit** | Web application framework |
| **Streamlit-WebRTC** | Real-time webcam streaming |
| **OpenCV** | Computer vision — face detection, video I/O |
| **DeepFace** | Facial emotion analysis |
| **librosa** | Audio feature extraction (MFCC, pitch, spectral) |
| **sounddevice** | Microphone audio recording |
| **scipy** | Centroid distance matching for multi-person tracking |
| **Plotly** | Interactive charts and visualizations |
| **Matplotlib** | Static chart rendering |
| **NumPy** | Numerical computing |
| **Pandas** | Data analysis and time-series handling |
| **UV** | Fast Python package installer |

## 📊 Architecture Diagrams

### Version 1 (app.py) Architecture

```mermaid
%%{init: { 'theme': 'dark', 'flowchart': { 'curve': 'basis' } } }%%
flowchart TD
    classDef default fill:#2D2B55,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef userNode fill:#3B9CFF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef processNode fill:#7F40BF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef dataNode fill:#4E8B3D,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold

    A[User] -->|Access Browser| B[Streamlit Web App]
    B -->|Start Webcam| C[WebRTC Component]
    C -->|Video Stream| D[EmotionDetector Class]
    D -->|Face Image| E[DeepFace Analysis]
    D -->|Process Frame| F[OpenCV]
    F -->|Face Detection| D
    E -->|Emotion Result| G[Temporal Smoothing]
    G -->|Stable Emotion| H[Display Result]
    H -->|Annotated Video| B
    B -->|UI Rendering| A

    A:::userNode
    B:::userNode
    C:::processNode
    D:::processNode
    E:::processNode
    F:::processNode
    G:::dataNode
    H:::dataNode

    linkStyle default stroke-width:2px,stroke:#FFFFFF,color:#FFFFFF
```

### Version 2 (appv2.py) Architecture

```mermaid
%%{init: { 'theme': 'dark', 'flowchart': { 'curve': 'basis' } } }%%
flowchart TD
    classDef default fill:#2D2B55,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef userNode fill:#3B9CFF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef processNode fill:#7F40BF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef dataNode fill:#4E8B3D,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef choiceNode fill:#E64747,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold

    A[User] -->|Access Browser| B[Streamlit Web App]
    B -->|Tab Selection| C{Tab Selection}

    %% Live Camera Tab
    C -->|Live Camera| D[WebRTC Component]
    D -->|Video Stream| E[EmotionDetector]
    E -->|Face Image| F[DeepFace Analysis]
    E -->|Process Frame| G[OpenCV]
    G -->|Face Detection| E
    F -->|Emotion Result| H[Temporal Smoothing]
    H -->|Stable Emotion| I[Display Result]
    I -->|Annotated Video| B

    %% Video Recording Feature
    E -->|Store Frames| J[Frame Buffer]
    E -->|Store Processed Frames| K[Processed Frame Buffer]
    D -->|Stop Recording| L[Save Video Option]
    L -->|Yes| M{Video Type}
    M -->|Processed| N[Save Processed Video]
    M -->|Raw| O[Save Raw Video]
    N --> P[Download Link]
    O --> P
    P -->|Download| A

    %% Image Upload Tab
    C -->|Upload Image| Q[Image Uploader]
    Q -->|Image| R[Analyze Image]
    R -->|Face Detection| S[OpenCV]
    R -->|Emotion Analysis| T[DeepFace]
    R -->|Display Results| U[Results UI]
    U -->|Analyzed Image| B

    A:::userNode
    B:::userNode
    C:::choiceNode
    D:::processNode
    E:::processNode
    F:::processNode
    G:::processNode
    H:::dataNode
    I:::dataNode
    J:::dataNode
    K:::dataNode
    L:::choiceNode
    M:::choiceNode
    N:::processNode
    O:::processNode
    P:::dataNode
    Q:::processNode
    R:::processNode
    S:::processNode
    T:::processNode
    U:::dataNode

    linkStyle default stroke-width:2px,stroke:#FFFFFF,color:#FFFFFF
```

### Version 3 (appv3.py) Architecture

```mermaid
%%{init: { 'theme': 'dark', 'flowchart': { 'curve': 'basis' } } }%%
flowchart TD
    classDef default fill:#2D2B55,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef userNode fill:#3B9CFF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef processNode fill:#7F40BF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef dataNode fill:#4E8B3D,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef choiceNode fill:#E64747,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold

    A[User] -->|Access Browser| B[Streamlit Web App]
    B -->|Tab Selection| C{Tab Selection}

    %% Live Camera Tab (Same as v2)
    C -->|Live Camera| D[WebRTC Component]
    D -->|Video Stream| E[EmotionDetector]
    E --> F[Record & Process]
    F --> G[Save Options]

    %% Image Upload Tab (Same as v2)
    C -->|Upload Image| H[Image Analysis]
    H --> I[Results Display]

    %% Video Analysis Tab (New)
    C -->|Video Analysis| J[Video Uploader]
    J -->|Video File| K[Temp Storage]
    K -->|User Settings| L[Sampling Rate]
    L -->|Process Video| M[process_video_for_emotions]
    M -->|Frame Extraction| N[OpenCV]
    N -->|Sampled Frames| O[Face Detection]
    O -->|Face Images| P[DeepFace Analysis]
    P -->|Emotion Data| Q[Data Processing]
    Q -->|DataFrame| R[Results]
    R -->|Visualization| S[Charts & Key Frames]
    R -->|Generate Report| T[HTML Report]
    T -->|Download| U[Download Button]
    U -->|Save Report| A
    S -->|Display| B

    A:::userNode
    B:::userNode
    C:::choiceNode
    D:::processNode
    E:::processNode
    F:::processNode
    G:::processNode
    H:::processNode
    I:::dataNode
    J:::processNode
    K:::dataNode
    L:::processNode
    M:::processNode
    N:::processNode
    O:::processNode
    P:::processNode
    Q:::processNode
    R:::dataNode
    S:::dataNode
    T:::dataNode
    U:::processNode

    linkStyle default stroke-width:2px,stroke:#FFFFFF,color:#FFFFFF
```

### Version 4 (appv4.py) Architecture

```mermaid
%%{init: { 'theme': 'dark', 'flowchart': { 'curve': 'basis' } } }%%
flowchart TD
    classDef default fill:#2D2B55,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef userNode fill:#3B9CFF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef processNode fill:#7F40BF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef dataNode fill:#4E8B3D,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef choiceNode fill:#E64747,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef newNode fill:#D97706,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold

    A[User] --> B[Streamlit App]
    B --> C{Tab}

    %% Live Camera Tab
    C -->|Live Camera| D[WebRTC Webcam]
    D --> E[EmotionDetector]
    E --> E1[CentroidTracker]
    E1 -->|Assign P1 P2 ...| E2[Per-Person History]
    E --> F[DeepFace]
    F --> G[Temporal Smoothing]
    G --> H[Face Emotion]
    E2 --> H2[Multi-Person Display]
    H2 --> B

    %% Voice branch (NEW)
    B -->|Toggle On| I[audio_emotion.py]
    I --> J[sounddevice Mic]
    J --> K[librosa Features]
    K --> L[Voice Emotion]

    %% Fusion
    H --> M[emotion_fusion.py]
    L --> M
    M -->|"face×0.65 + voice×0.35"| N[Final Fused Emotion]
    N -->|3-Card Display| B

    %% Video Analysis Tab
    C -->|Video Analysis| O[Video Uploader]
    O --> P[process_video_for_emotions]
    P --> P1[CentroidTracker]
    P1 -->|per_person_data| P2[Per-Person Charts]
    P --> Q[Emotion DataFrame]
    Q --> R[Charts & Key Frames]
    Q --> S[emotion_analysis.py]
    S --> T["Volatility / Stability / Peaks / Stress"]
    T --> B
    Q --> V[emotion_prediction.py]
    V --> W["Markov Forecast"]
    W --> B
    Q --> U[HTML Report]
    P2 --> U
    W --> U

    A:::userNode
    B:::userNode
    C:::choiceNode
    D:::processNode
    E:::processNode
    E1:::newNode
    E2:::newNode
    F:::processNode
    G:::dataNode
    H:::dataNode
    H2:::newNode
    I:::newNode
    J:::newNode
    K:::newNode
    L:::newNode
    M:::newNode
    N:::newNode
    O:::processNode
    P:::processNode
    P1:::newNode
    P2:::newNode
    Q:::dataNode
    R:::dataNode
    S:::newNode
    T:::newNode
    U:::dataNode
    V:::newNode
    W:::newNode

    linkStyle default stroke-width:2px,stroke:#FFFFFF,color:#FFFFFF
```

> **Legend** — 🟠 Orange nodes are new in v4.

### Multimodal Fusion Flow

```mermaid
%%{init: { 'theme': 'dark', 'flowchart': { 'curve': 'natural' } } }%%
flowchart LR
    classDef default fill:#2D2B55,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef dataNode fill:#4E8B3D,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef processNode fill:#7F40BF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef choiceNode fill:#E64747,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold

    A[Webcam Frame] --> B[Face Detection]
    B --> C[DeepFace Analysis]
    C --> D[Temporal Smoothing]
    D --> E["Face Scores (7 classes)"]

    F[Microphone] --> G[sounddevice Record]
    G --> H[librosa Feature Extraction]
    H --> I[Heuristic Classifier]
    I --> J["Voice Scores (6 classes)"]

    E --> K[emotion_fusion.py]
    J --> K
    K -->|"face×0.65 + voice×0.35"| L[Fused Scores]
    L --> M[Final Dominant Emotion]

    A:::dataNode
    B:::processNode
    C:::processNode
    D:::processNode
    E:::dataNode
    F:::dataNode
    G:::processNode
    H:::processNode
    I:::processNode
    J:::dataNode
    K:::processNode
    L:::dataNode
    M:::dataNode

    linkStyle default stroke-width:2px,stroke:#FFFFFF,color:#FFFFFF
```

### Multi-Person Tracking Flow

```mermaid
%%{init: { 'theme': 'dark', 'flowchart': { 'curve': 'basis' } } }%%
flowchart TD
    classDef default fill:#2D2B55,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef dataNode fill:#4E8B3D,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef processNode fill:#7F40BF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef choiceNode fill:#E64747,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold

    A[Video Frame] --> B[OpenCV Face Detection]
    B --> C["Detected Rects (x,y,w,h)"]
    C --> D[CentroidTracker.update]
    D --> E["Compute Centroids"]
    E --> F["scipy cdist — Distance Matrix"]
    F --> G[Greedy ID Assignment]
    G --> H{"New face?"}
    H -->|Yes| I[Register new ID]
    H -->|No| J[Carry existing ID]
    I --> K["person_id → rect mapping"]
    J --> K
    K --> L[DeepFace per person]
    L --> M["person_emotion_history\n{id: [(ts, emotion, scores)]}"]  
    M --> N[Draw P1 / P2 labels on frame]
    M --> O[Per-person timeline charts]
    M --> P[HTML Report section]

    A:::dataNode
    B:::processNode
    C:::dataNode
    D:::processNode
    E:::processNode
    F:::processNode
    G:::processNode
    H:::choiceNode
    I:::processNode
    J:::processNode
    K:::dataNode
    L:::processNode
    M:::dataNode
    N:::dataNode
    O:::dataNode
    P:::dataNode

    linkStyle default stroke-width:2px,stroke:#FFFFFF,color:#FFFFFF
```

### Behavioral Analysis Flow

```mermaid
%%{init: { 'theme': 'dark', 'flowchart': { 'curve': 'basis' } } }%%
flowchart TD
    classDef default fill:#2D2B55,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef dataNode fill:#4E8B3D,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef processNode fill:#7F40BF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold

    A[Emotion DataFrame] --> B[compute_emotion_statistics]
    B --> C[Dominant Emotion]
    B --> D[calculate_volatility]
    D --> E[Volatility Index]
    D --> F[Stability Score]
    B --> G[detect_peak_emotions]
    G --> H["Peak Moments (>80% confidence)"]
    B --> I[detect_stress_indicators]
    I --> J[Rapid Switches and Stress Patterns]
    C & E & F & H & J --> K[Streamlit UI Display]
    C & E & F & H & J --> L[HTML Report Section]

    A:::dataNode
    B:::processNode
    C:::dataNode
    D:::processNode
    E:::dataNode
    F:::dataNode
    G:::processNode
    H:::dataNode
    I:::processNode
    J:::dataNode
    K:::dataNode
    L:::dataNode

    linkStyle default stroke-width:2px,stroke:#FFFFFF,color:#FFFFFF
```

### Emotion Prediction Flow

```mermaid
%%{init: { 'theme': 'dark', 'flowchart': { 'curve': 'basis' } } }%%
flowchart TD
    classDef default fill:#2D2B55,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef dataNode fill:#4E8B3D,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef processNode fill:#7F40BF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef choiceNode fill:#E64747,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold

    A[Emotion Timeline] --> B{"Length > 20?"}
    B -->|No| C[Skip Prediction]
    B -->|Yes| D[build_transition_matrix]
    D --> E[Transition Probability Matrix]
    E --> F[predict_next_emotion]
    F --> G["Next-Emotion Probabilities"]
    G --> H[Forecast Panel UI]
    E --> I[predict_future_emotions]
    I --> J["Multi-Step Predictions (5 steps)"]
    J --> K[create_forecast_chart]
    K --> L[Plotly Grouped-Bar Chart]
    L --> H
    H --> M[Streamlit Display]
    H --> N[HTML Report Section]

    A:::dataNode
    B:::choiceNode
    C:::dataNode
    D:::processNode
    E:::dataNode
    F:::processNode
    G:::dataNode
    H:::dataNode
    I:::processNode
    J:::dataNode
    K:::processNode
    L:::dataNode
    M:::dataNode
    N:::dataNode

    linkStyle default stroke-width:2px,stroke:#FFFFFF,color:#FFFFFF
```

### Emotion Detection Flow

```mermaid
%%{init: { 'theme': 'dark', 'flowchart': { 'curve': 'natural' } } }%%
flowchart LR
    classDef default fill:#2D2B55,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef dataNode fill:#4E8B3D,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef processNode fill:#7F40BF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef choiceNode fill:#E64747,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold

    A[Input Source] --> B{Face Detected?}
    B -->|Yes| C[Extract Face]
    B -->|No| A
    C --> D[DeepFace Analysis]
    D --> E{Apply Smoothing}
    E --> F[Emotions History]
    F --> G[Calculate Dominant Emotion]
    G --> H[Display Result]
    G --> I[Confidence Score]
    I --> H

    A:::dataNode
    B:::choiceNode
    C:::processNode
    D:::processNode
    E:::choiceNode
    F:::dataNode
    G:::processNode
    H:::dataNode
    I:::dataNode

    linkStyle default stroke-width:2px,stroke:#FFFFFF,color:#FFFFFF
```

### Video Analysis Flow (Version 3)

```mermaid
%%{init: { 'theme': 'dark', 'flowchart': { 'curve': 'basis' } } }%%
flowchart TD
    classDef default fill:#2D2B55,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef dataNode fill:#4E8B3D,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef processNode fill:#7F40BF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef choiceNode fill:#E64747,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold
    classDef userNode fill:#3B9CFF,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF,font-weight:bold

    A[Upload Video] --> B[Set Sampling Rate]
    B --> C[Process Video]
    C --> D{For Each Sample Frame}
    D --> E{Face Detected?}
    E -->|Yes| F[Analyze Emotions]
    E -->|No| D
    F --> G[Store Data]
    G --> H{Emotion Changed?}
    H -->|Yes| I[Save as Key Frame]
    H -->|No| D
    D -->|All Frames Processed| J[Compile Data]
    J --> K[Generate Statistics]
    K --> L[Create Visualizations]
    L --> M[Display Results]
    K --> N[Generate Report]
    N --> O[Download Report]

    A:::userNode
    B:::processNode
    C:::processNode
    D:::choiceNode
    E:::choiceNode
    F:::processNode
    G:::dataNode
    H:::choiceNode
    I:::dataNode
    J:::processNode
    K:::processNode
    L:::processNode
    M:::dataNode
    N:::processNode
    O:::dataNode

    linkStyle default stroke-width:2px,stroke:#FFFFFF,color:#FFFFFF
```

## 📜 License

© 2026 Emotion AI | All Rights Reserved

---

Built with ❤️ for emotion analysis and computer vision