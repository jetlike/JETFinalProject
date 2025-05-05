from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import torch
import cv2
import numpy as np

# load MiDaS model and transform
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

def estimate_depth(frame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)

    # resize to fixed size (square) to avoid transformer mismatch
    input_resized = cv2.resize(frame, (384, 384))
    input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)
    input_pil = Image.fromarray(input_rgb)

    transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    input_tensor = transform(input_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=frame.shape[:2], mode="bicubic", align_corners=False).squeeze()

    return prediction.cpu().numpy()

def find_pointed_object_in_depth(depth_map, wrist, fingertip, max_steps=150, step_size=4, depth_jump_thresh=0.03):
    h, w = depth_map.shape
    x0, y0 = wrist
    x1, y1 = fingertip

    # normalize direction vector
    dx = x1 - x0
    dy = y1 - y0
    mag = np.sqrt(dx**2 + dy**2)
    if mag == 0: return None

    dx /= mag
    dy /= mag

    # clamp wrist position to frame bounds
    x0 = int(min(max(0, x0), w - 1))
    y0 = int(min(max(0, y0), h - 1))
    wrist_depth = depth_map[y0][x0]

    for step in range(1, max_steps):
        px = int(x1 + dx * step * step_size)
        py = int(y1 + dy * step * step_size)

        if px < 0 or py < 0 or px >= w or py >= h:
            break  # out of bounds

        current_depth = depth_map[py][px]
        depth_delta = current_depth - wrist_depth

        # If object is significantly deeper than the wrist
        if depth_delta > depth_jump_thresh:
            return (px, py)

    return None  # No target found along ray

