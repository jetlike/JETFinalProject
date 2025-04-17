import cv2
import os

def crop_object_from_point(frame, point, size=100):
    x, y = point
    h, w, _ = frame.shape

    # Define bounding box around point
    x1 = max(0, x - size)
    y1 = max(0, y - size)
    x2 = min(w, x + size)
    y2 = min(h, y + size)

    # Crop image
    crop = frame[y1:y2, x1:x2]

    # Save to file
    output_path = "data/samples/cropped_object.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, crop)
    print(f"Cropped image saved to {output_path}")

    return output_path

def detect_object_from_pointing(point_coords, frame_width, frame_height):
    x, y = point_coords

    # Divide the screen into a 3x3 grid
    col = x // (frame_width // 3)
    row = y // (frame_height // 3)

    region_label = f"Zone [{int(row)}, {int(col)}]"

    # Placeholder: Map each grid to a dummy object
    object_map = {
        (0, 0): "a lamp",
        (0, 1): "a monitor",
        (0, 2): "a window",
        (1, 0): "a book",
        (1, 1): "a coffee mug",
        (1, 2): "a phone",
        (2, 0): "a pen",
        (2, 1): "a notebook",
        (2, 2): "a chair"
    }

    object_name = object_map.get((int(row), int(col)), "an unknown object")

    return object_name
