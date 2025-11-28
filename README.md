✋ ASL Translator
This is not meant to be a research-grade ASL interpreter or anything serious.
It's simply a fun personal project where I experiment with computer vision, MediaPipe, and basic machine learning to recognize a few static ASL hand shapes.

The goal is to learn, explore, and gradually improve — not to create a full ASL translation system.

For now, I have only trained the model on (A, B, C), and will train it further as time passes.

📁 Project Structure:
asl-translator/
│
├── data/
│   └── letters.csv             # All collected samples (A, B, C)
│
├── models/
│   └── asl_letters_rf.pkl      # Trained RandomForest model
│
├── src/
│   ├── data_collection.py      # Script for collecting training data
│   ├── train_model.py          # Trains the ML model
│   ├── interpreter.py          # Real-time ASL interpreter
│   ├── hand_tracker.py         # Mediapipe hand detection wrapper
│   ├── features.py             # Normalization / feature extraction
│   └── show_landmarks.py       # Debugging tool to visualize landmarks
│
└── README.md


⚙️ Setup & Installation:
Clone the repo:
git clone https://github.com/<Ali-Kalbouneh>/asl-translator.git
cd asl-translator

Install Dependencies:
pip install opencv-python mediapipe numpy scikit-learn joblib


🏗 Collecting Training Data:

Before training the model(if you want to), you need to collect examples of your hand forming 

ASL letters.

Run:

python3 src/data_collection.py

Controls:

Press A, B, or C → choose a letter to record

Press 3 → save a sample of your hand

Press ESC → exit the tool

All samples are stored in: data/letters.csv


🎓 Training the Model:

After collecting enough data run:

python3 src/train_model.py

This will:

Load your dataset

Train a RandomForest classifier

Print training and test accuracy

Save the trained model to: models/asl_letters_rf.pkl


🤖 Running the Real-Time Interpreter

Once the model is trained, run: 

python3 src/interpreter.py

The interpreter:

Opens your webcam

Detects your hand using MediaPipe

Extracts normalized landmark features

Predicts A/B/C in real time

Displays the predicted letter on-screen

Draws hand landmarks for debugging

Uses smoothing to reduce flicker

Hold up the ASL letters and watch it attempt to classify them.



🧪 Notes & Limitations

This is just a fun experiment, not an actual ASL translation tool.

Current limitations:

Only supports A, B, and C

Sensitive to hand rotation, lighting, and camera distance

Occasional misclassifications

No dynamic signs (J, Z)

Not meant for real communication

But it’s a great learning project for:

Computer vision

Hand-tracking

Feature engineering

Machine learning

Real-time interactive systems


🚧 Future Ideas

Add a NONE / rest class

Expand dataset to full alphabet (A–Z)

Add dynamic gestures (J, Z)

Use a deep learning model (CNN/MLP)

Add sentence builder / UI overlay

Support two-hand signs
