import cv2
import mediapipe as mp
import time
from vision.depth_estimator import estimate_depth, find_pointed_object_in_depth
import os

def crop_from_depth_target(frame, target_point, box_size=100, save_path="data/samples/depth_crop.jpg"):
    h, w, _ = frame.shape
    x, y = target_point
    x1 = max(0, x - box_size)
    y1 = max(0, y - box_size)
    x2 = min(w, x + box_size)
    y2 = min(h, y + box_size)
    cropped = frame[y1:y2, x1:x2]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cropped)
    return "depth-based crop", save_path

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def is_pointing(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

    # Index extended (tip above pip), others curled (tip not much above pip)
    index_extended = index_tip.y < index_pip.y - 0.02
    middle_curled = middle_tip.y > middle_pip.y - 0.005
    ring_curled = ring_tip.y > ring_pip.y - 0.005

    return index_extended and middle_curled and ring_curled

def get_pointing_target(cooldown_secs=1.0):
    cap = cv2.VideoCapture(0)
    label = "nothing"
    crop_path = None
    last_label = None
    last_detect_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            score = result.multi_handedness[0].classification[0].score
            if score < 0.9:
                continue

            for hand_landmarks in result.multi_hand_landmarks:
                # if not is_pointing(hand_landmarks):
                  #  continue

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                h, w, _ = frame.shape
                cx_tip, cy_tip = int(index_tip.x * w), int(index_tip.y * h)
                cx_wrist, cy_wrist = int(wrist.x * w), int(wrist.y * h)

                current_time = time.time()
                if current_time - last_detect_time > cooldown_secs:
                    depth_map = estimate_depth(frame)
                    target_point = find_pointed_object_in_depth(
                        depth_map,
                        wrist=(cx_wrist, cy_wrist),
                        fingertip=(cx_tip, cy_tip)
                    )

                    if target_point:
                        candidate_label, candidate_path = crop_from_depth_target(
                            frame,
                            target_point=target_point
                        )

                        if candidate_label != last_label:
                            label = candidate_label
                            crop_path = candidate_path
                            last_label = label
                            last_detect_time = current_time

                cv2.circle(frame, (cx_tip, cy_tip), 10, (0, 255, 0), -1)

        cv2.putText(frame, f"Locked on: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Vision - Press Q to exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return label, crop_path
