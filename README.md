✋ ASL Translator

This project is not meant to be a serious ASL translation tool.
It’s simply a personal project where I experiment with:

MediaPipe (hand tracking)

OpenCV (webcam + visualization)

Basic machine learning (RandomForest)

The goal is to learn and explore — not to build a production-grade ASL interpreter.

📁 Project Structure
asl-translator/
│
├── data/
│   └── letters.csv             # Collected samples (A, B, C)
│
├── models/
│   └── asl_letters_rf.pkl      # Trained RandomForest model
│
├── src/
│   ├── data_collection.py      # Script for collecting training data
│   ├── train_model.py          # Trains the machine learning model
│   ├── interpreter.py          # Real-time ASL interpreter
│   ├── hand_tracker.py         # MediaPipe hand detection wrapper
│   ├── features.py             # Normalization / feature extraction
│   └── show_landmarks.py       # Debug landmark visualization tool
│
└── README.md

⚙️ Installation

Clone the repository:

git clone https://github.com/<your-username>/asl-translator.git
cd asl-translator


Install dependencies:

pip install opencv-python mediapipe numpy scikit-learn joblib

🏗 Collecting Training Data (A, B, C)

Before training the model, collect your own hand-shape samples.

Run:

python3 src/data_collection.py

Controls

A, B, or C → choose a letter

3 → save a sample

ESC → exit

Samples are saved automatically to:

data/letters.csv


Tips for better data:

Collect 60–120 samples per letter

Use slightly different hand positions

Vary angles, lighting, and distance

Keep your hand steady before saving each sample

🎓 Training the Model

After collecting enough samples:

python3 src/train_model.py


This script will:

Load the dataset

Train a RandomForest classifier

Print accuracy and a classification report

Save the trained model to:

models/asl_letters_rf.pkl


You only need to retrain when you add new samples.

🤖 Running the Real-Time Interpreter

After training:

python3 src/interpreter.py


The interpreter will:

Open your webcam

Detect hand landmarks using MediaPipe

Normalize the coordinates

Predict A/B/C in real time

Display the prediction on the screen

Draw the landmark skeleton

Smooth predictions across frames

Hold up one of the ASL letters and the model will attempt to classify it.

🧪 Notes & Limitations

This project is just for fun and has limitations:

Recognizes only A, B, and C

Sensitive to hand rotation and lighting

Sometimes flickers between predictions

No “rest hand” or “none” class yet

Not intended for real communication

Still, it’s a great way to learn:

Computer vision

MediaPipe

Feature extraction

Machine learning

Real-time interactive apps

🚧 Future Ideas

Add a NONE / rest-hand class

Collect samples for more letters (A–Z)

Recognize dynamic signs (J, Z)

Add word-level prediction

Implement a deep learning model

Add UI elements or sentence-builder
