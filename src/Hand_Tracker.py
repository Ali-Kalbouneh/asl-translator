import cv2
import mediapipe as mp

class HandTracker:
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode = False,
            max_num_hands = 2,
            min_detection_confidence = 0.6,
            min_tracking_confidence = 0.7,
        )

    def get_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None, None

        hand_lm_obj = results.multi_hand_landmarks[0]
 
        h, w, _ = frame.shape
        coords = []
        for lm in hand_lm_obj.landmark:
            coords.append((lm.x * w, lm.y * h, lm.z))

        return coords, hand_lm_obj

    def draw(self, frame, landmarks_obj):
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks_obj,
            self.mp_hands.HAND_CONNECTIONS
        )

