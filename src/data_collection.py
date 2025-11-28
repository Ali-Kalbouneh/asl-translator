import cv2
from hand_tracker import HandTracker
from features import normalize_landmarks
import csv
import os

# Initialize Tracker and Webcame
tracker = HandTracker()
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Make CSV File
os.makedirs("../data", exist_ok=True)
csv_path = "../data/letters.csv"

# Write data to file
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(63)])

current_label = None
typing_label = False
typed_text = ""

save_message = ""
save_message_counter = 0

print("=== ASL DATA COLLECTION ===")
print("Press a letter (A-Z) to set a label.")
print("Press '3' to save a sample.")
print("Press 'esc' to exit.")

# Main Loop

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Mirroring Frame
    frame = cv2.flip(frame, 1)

    coords, landmarks = tracker.get_landmarks(frame)

    # Drawing Landmarks
    if landmarks:
        tracker.draw(frame, landmarks)

    # Text on window
    if current_label:
        cv2.putText(frame, f"Label: {current_label}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if save_message_counter > 0:
        cv2.putText(frame, save_message, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        save_message_counter -= 1

    if typing_label:
        cv2.putText(frame, f"Label: {typed_text}_", (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ASL Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Select which letter data is about to be collected ona
    if not typing_label and ord("a") <= key <= ord("z"):
        current_label = chr(key).upper()
        print(f"Current label set to: {current_label}")

    elif key == ord("0") and not typing_label:
        typing_label = True
        typed_text = ""
        print("Enter your own label, then press Enter")

    elif typing_label:
        if key == 13: #Enter
            current_label = typed_text.strip()
            typing_label = False
            print(f"Custom label set to {current_label}")

        elif key in (8, 127): # Backspace
            typed_text = typed_text[:-1]

        elif 32 <= key <= 126:
            typed_text += chr(key)

    # save data to csv
    elif key == ord("3"):
        if coords is not None and current_label:
            features = normalize_landmarks(coords)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([current_label] + features.tolist())

            print(f"Saved sample for {current_label}")

            save_message = f"Saved sample for {current_label}"
            save_message_counter = 20
        
        else:
            print("Cannot save: No hand detected OR no label chosen.")
    
    # exit on "esc"
    elif key == 27:
        print("Exiting data collection...")
        break

webcam.release()
cv2.destroyAllWindows()