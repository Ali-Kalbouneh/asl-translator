# ASL Translator

A real-time computer-vision application that recognizes American Sign Language letter gestures from webcam input using hand-landmark tracking and a machine-learning classification pipeline.

The project captures hand landmarks with **MediaPipe**, normalizes them into model features, trains a **Random Forest** classifier, and performs real-time inference through an **OpenCV** webcam interface.

## How It Works

```text
Webcam Frame
     |
     v
MediaPipe Hand Detection
     |
     v
21 Hand Landmarks
     |
     v
Feature Extraction + Normalization
     |
     v
Random Forest Classifier
     |
     v
Prediction Smoothing
     |
     v
Real-Time OpenCV Display
```

## Features

- Real-time webcam hand tracking
- MediaPipe landmark extraction
- Normalized landmark-based feature representation
- Custom training-data collection workflow
- Random Forest gesture classification
- Prediction smoothing to reduce frame-to-frame flicker
- On-screen hand skeleton and predicted letter

## Tech Stack

- Python
- OpenCV
- MediaPipe
- scikit-learn
- NumPy
- joblib

## Project Structure

```text
asl-translator/
|
├── data/
│   └── letters.csv             # Collected training samples
|
├── models/
│   └── asl_letters_rf.pkl      # Trained Random Forest model
|
├── src/
│   ├── data_collection.py      # Training-data collection
│   ├── train_model.py          # Model training and evaluation
│   ├── interpreter.py          # Real-time inference application
│   ├── hand_tracker.py         # MediaPipe hand-detection wrapper
│   ├── features.py             # Feature extraction and normalization
│   └── show_landmarks.py       # Landmark visualization/debug utility
|
└── README.md
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Ali-Kalbouneh/asl-translator.git
cd asl-translator
pip install opencv-python mediapipe numpy scikit-learn joblib
```

## Collect Training Data

Run:

```bash
python3 src/data_collection.py
```

Controls:

- **A**, **B**, **C** select the letter being recorded
- **3** saves the current hand sample
- **ESC** exits

Samples are written to:

```text
data/letters.csv
```

For a more robust dataset, collect samples with variations in hand angle, distance, and lighting while keeping the complete hand visible.

## Train the Model

```bash
python3 src/train_model.py
```

The training pipeline:

1. Loads the collected landmark dataset
2. Builds the feature representation
3. Trains a Random Forest classifier
4. Reports model accuracy and classification metrics
5. Serializes the trained model to `models/asl_letters_rf.pkl`

## Run Real-Time Inference

```bash
python3 src/interpreter.py
```

The application opens the webcam, detects the user's hand, extracts normalized landmark features, predicts the gesture, smooths predictions across frames, and overlays the result on the video feed.

## Current Scope

The current model recognizes the static letters **A, B, and C**. The project focuses on the complete computer-vision pipeline rather than serving as a production ASL translation system.

Important limitations include:

- Limited gesture vocabulary
- Sensitivity to hand orientation and lighting
- Training data collected from a limited set of examples
- Dynamic gestures such as J and Z require temporal information rather than single-frame classification
- Full ASL translation involves grammar, motion, facial expression, and context beyond isolated hand poses

## Engineering Decisions

### Landmark features instead of raw images

Using MediaPipe landmarks reduces the input from full image pixels to structured hand geometry. This makes the classifier substantially smaller and allows experimentation without training a large image model.

### Feature normalization

Normalizing landmark coordinates reduces sensitivity to where the hand appears in the camera frame and helps the model focus on relative hand shape.

### Prediction smoothing

Individual video frames can produce unstable predictions. Smoothing predictions across frames improves the stability of the displayed result during real-time use.

## Future Improvements

- Expand the dataset to the full static ASL alphabet
- Add temporal gesture recognition for J and Z
- Evaluate models across multiple users
- Add confidence thresholds and an unknown-gesture state
- Compare Random Forest performance with neural-network approaches
- Add two-hand gesture support
- Build sentence-level output from sequences of recognized signs
