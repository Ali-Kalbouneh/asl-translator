import cv2
import mediapipe as mp


webcam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#initializing Hand detector
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.6,
    min_tracking_confidence = 0.7,
)

while True:
    ret,frame = webcam.read()

    if ret:
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        #if detected, draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    connections = mp_hands.HAND_CONNECTIONS
                )

        # Display the Frame
        cv2.imshow("Hand Tracker", frame)

        # Exit when "E" is pressed
        key = cv2.waitKey(1)
        if key == ord("e"):
            break
    else:
        print("Failed to Capture Frame")
        break

webcam.release()
cv2.destroyAllWindows()