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

    # index extended (tip above pip), others curled (tip not much above pip)
    index_extended = index_tip.y < index_pip.y - 0.02
    middle_curled = middle_tip.y > middle_pip.y - 0.005
    ring_curled = ring_tip.y > ring_pip.y - 0.005

    return index_extended and middle_curled and ring_curled

def get_pointing_target(cooldown_secs=1.0):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # hand detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            # high-confidence detections only
            if result.multi_handedness[0].classification[0].score < 0.9:
                continue

            lm = result.multi_hand_landmarks[0]
            h, w, _ = frame.shape

            # compute wrist and index fingertip coords
            cx_wrist = int(lm.landmark[mp_hands.HandLandmark.WRIST].x * w)
            cy_wrist = int(lm.landmark[mp_hands.HandLandmark.WRIST].y * h)
            cx_tip = int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            cy_tip = int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

            # estimate depth and target point
            depth_map    = estimate_depth(frame)
            target_point = find_pointed_object_in_depth(depth_map, wrist=(cx_wrist, cy_wrist), fingertip=(cx_tip, cy_tip))

            if target_point:
                # skip if too close to finger
                dx = target_point[0] - cx_tip
                dy = target_point[1] - cy_tip
                if (dx*dx + dy*dy) < (100**2): # 100px radius disallowed
                    continue
                
                # return information
                label, path = crop_from_depth_target(frame, target_point)
                cap.release()
                cv2.destroyAllWindows()
                return label, path

            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (cx_tip, cy_tip), 8, (0,255,0), -1)

        # show live feed so user can align
        cv2.imshow("Vision - Press Q to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup if no crop
    cap.release()
    cv2.destroyAllWindows()
    return "nothing", None

