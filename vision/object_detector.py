from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load YOLO model
model = YOLO('yolov8m.pt')

def smart_crop_pointed_object(frame, wrist, fingertip, save_path="data/samples/smart_crop.jpg"):
    x0, y0 = wrist
    x1, y1 = fingertip

    # Normalize pointing direction
    dx = x1 - x0
    dy = y1 - y0
    mag = np.sqrt(dx**2 + dy**2)
    if mag == 0:
        return "invalid vector", None
    dx /= mag
    dy /= mag

    # Detect objects
    results = model(frame)[0]
    best_box = None
    best_score = float('inf')
    label = "unknown object"

    # Compare center of each box to the pointing vector
    for box in results.boxes:
        x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])
        box_cx = (x1_box + x2_box) // 2
        box_cy = (y1_box + y2_box) // 2

        # Vector from wrist to box center
        bx = box_cx - x0
        by = box_cy - y0

        # Cosine similarity with pointing direction
        b_mag = np.sqrt(bx**2 + by**2)
        if b_mag == 0:
            continue
        bx /= b_mag
        by /= b_mag
        dot = dx * bx + dy * by  # cosine similarity

        # Angle difference (lower = better)
        angle_diff = np.arccos(np.clip(dot, -1.0, 1.0))

        # Small angle = better alignment
        if angle_diff < best_score:
            best_score = angle_diff
            best_box = (x1_box, y1_box, x2_box, y2_box)
            label = results.names[int(box.cls[0])]

    # Crop that object
    if best_box:
        x1_crop, y1_crop, x2_crop, y2_crop = best_box
        cropped = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cropped)
        print(f"[SmartCrop] Cropped {label} saved to {save_path}")
        return label, save_path
    else:
        print("[SmartCrop] No aligned object found.")
        return "nothing", None