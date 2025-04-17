import cv2
import mediapipe as mp
import os
from vision.object_detector import crop_object_from_point

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def get_pointing_target():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)

                # Draw marker on fingertip
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                # Save cropped region
                crop_path = crop_object_from_point(frame, (cx, cy))
                cap.release()
                cv2.destroyAllWindows()
                return crop_path

        cv2.imshow("Point at an object", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None
