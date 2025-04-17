import cv2
import mediapipe as mp
from vision.object_detector import detect_object_from_pointing

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def get_pointing_target():
    cap = cv2.VideoCapture(0)
    object_name = "nothing"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)

                # Draw circle on fingertip
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                # Use object detector
                object_name = detect_object_from_pointing((cx, cy), w, h)
                cv2.putText(frame, f"Pointing at: {object_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Vision - Press Q to exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return object_name
