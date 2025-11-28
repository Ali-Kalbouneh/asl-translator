# ✋ ASL Translator

This project is **not meant to be a serious ASL translation tool**.  
It's simply a **personal project** where I experiment with:

- MediaPipe (hand tracking)  
- OpenCV (webcam + visualization)  
- Basic machine learning (RandomForest)  

The goal is to learn and explore — not to build a production-grade ASL interpreter.

---

## 📁 Project Structure

```
asl-translator/
│
├── data/
│   └── letters.csv             # Collected samples (A, B, C)
│
├── models/
│   └── asl_letters_rf.pkl      # Trained RandomForest model
│
├── src/
│   ├── data_collection.py      # Collects training data
│   ├── train_model.py          # Trains the model
│   ├── interpreter.py          # Real-time ASL interpreter
│   ├── hand_tracker.py         # MediaPipe hand detection wrapper
│   ├── features.py             # Feature extraction + normalization
│   └── show_landmarks.py       # Debugging utility
│
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/asl-translator.git
cd asl-translator
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy scikit-learn joblib
```

---

## 🏗 Collecting Training Data (A, B, C)

Run the data collection script:

```bash
python3 src/data_collection.py
```

### Controls

- **A**, **B**, **C** → select which letter to record  
- **3** → save the current frame as a sample  
- **ESC** → exit  

Samples are saved to:

```
data/letters.csv
```

### Tips for best results

- Collect **60–120 samples per letter**  
- Use small variations in angle, distance, lighting  
- Keep your hand steady when saving  
- Ensure your hand is fully inside the frame  

---

## 🎓 Training the Model

Once you have collected enough samples, train the model:

```bash
python3 src/train_model.py
```

This will:

- Load all data  
- Train a RandomForest classifier  
- Print accuracy and a classification report  
- Save the trained model to:

```
models/asl_letters_rf.pkl
```

---

## 🤖 Running the Real-Time Interpreter

After training the model:

```bash
python3 src/interpreter.py
```

The interpreter:

- Opens your webcam  
- Uses MediaPipe to detect hand landmarks  
- Extracts normalized features  
- Predicts A/B/C in real time  
- Displays the prediction on screen  
- Draws the hand skeleton  
- Smooths predictions to reduce flicker  

---

## 🧪 Notes & Limitations

This is just a fun experiment. Current limitations:

- Only supports **A, B, C**  
- Sensitive to hand rotation and lighting  
- No “rest hand / none” class yet  
- Flickers occasionally  
- Not meant for actual ASL communication  

Still, it’s a great way to learn:

- Computer vision basics  
- MediaPipe landmark extraction  
- Feature engineering  
- Machine learning model training  
- Real-time CV application design  

---

## 🚧 Future Ideas

- Add a **NONE** / rest-hand class  
- Add more letters (A–Z)  
- Recognize dynamic gestures (J, Z)  
- Add a sentence builder  
- Use a neural-network-based model  
- Support two-hand signs  

---

