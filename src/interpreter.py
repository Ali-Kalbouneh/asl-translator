import cv2
import joblib
import numpy as np
from hand_tracker import HandTracker
from features import normalize_landmarks
from collections import deque


# Load Model & Initialize 

model = joblib.load("../models/asl_letters_rf.pkl")
tracker = HandTracker()
webcam = cv2.VideoCapture(0)

if not webcam.isOpened:
    print("could not access webcam")
    exit()

prediction_buffer = deque(maxlen = 5) ## Smoothing buffer (to prevent flickering), Keeps last 5 predictions

print("=== ASL Interpreter Running ===")
print("Press ESC to exit.")


# Main Loop

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)

    coords, landmarks = tracker.get_landmarks(frame)

    if landmarks:
        tracker.draw(frame, landmarks)


    # Prediction

    if coords is not None:
        features = normalize_landmarks(coords)

        # Get probabilistic predictions
        proba = model.predict_proba([features])[0]
        best_prob = np.max(proba)
        best_letter = model.classes_[np.argmax(proba)]

        # Confidence threshold
        # If model is not at least 70% confident → treat as "no sign"
        if best_letter == "NONE" or best_prob < 0.70:
            prediction_buffer.append("NONE")
        else:
            prediction_buffer.append(best_letter)

        # Stable prediction (majority in last 5 frames) 
        if prediction_buffer.count("NONE") >= 3:  # appears 3+ times within buffer
            smoothed_pred = "NONE"
        else:
            non_none = [p for p in prediction_buffer if p != "NONE"]
            smoothed_pred = max(set(non_none), key = non_none.count) if non_none else "NONE"

        # Display result
        if smoothed_pred == "NONE":
            cv2.putText(frame, "Pred: ...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, f"Pred: {smoothed_pred}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "No hand detected", (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Interpreter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # exit on esc
        break

webcam.release()
cv2.destroyAllWindows()