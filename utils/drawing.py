import cv2

def draw_pointing_line(frame, start, end):
    cv2.line(frame, start, end, (0, 255, 0), 2)
    return frame
