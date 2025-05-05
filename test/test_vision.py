from vision.hand_tracker import get_pointing_target

if __name__ == "__main__":
    obj = get_pointing_target()
    print(f"Detected: {obj}")