import cv2
import mediapipe as mp

class HandTracker:
    
    def __init__(self):
        # MediaPipe modules
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize the MediaPipe Hands model
        self.hands = self.mp_hands.Hands(
            static_image_mode = False,   # For video: track across frames instead of re-detecting
            max_num_hands = 2,   # We only really use 1 hand, but this prevents missed detections
            min_detection_confidence = 0.6,
            min_tracking_confidence = 0.7,
        )

    
    """
        Detect hand landmarks in a BGR frame.

        Returns:
            coords (list of [x, y, z]):
                The pixel coordinates of the 21 hand landmarks.
                Used for feature extraction.
            
            hand_lm_obj (MediaPipe landmark object):
                The original MediaPipe landmark object, used for drawing.
                return None, None if no hand is detected.
    """
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

    
    """
        Draws the 21 landmarks + connections onto the frame.
        Uses MediaPipe's built-in drawing utilities.
    """
    def draw(self, frame, landmarks_obj):
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks_obj,
            self.mp_hands.HAND_CONNECTIONS
        )

