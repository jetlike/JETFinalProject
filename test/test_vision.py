from vision.hand_tracker import get_pointing_target

if __name__ == "__main__":
    print("Starting test... Point your finger at the screen.")
    obj = get_pointing_target()
    print(f"Detected: {obj}")