import cv2
from hand_tracker import HandTracker

tracker = HandTracker()
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    coords, landmarks = tracker.get_landmarks(frame)

    if landmarks:
        tracker.draw(frame, landmarks)


    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) == ord("e"):
        break

webcam.release()
cv2.destroyAllWindows()
