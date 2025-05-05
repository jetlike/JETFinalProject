from dotenv import load_dotenv
load_dotenv()

import os, time, cv2, pvporcupine
from face.facial_recog import FaceRecognitionSystem
from audio.listener import WakeWordListener
from vision.hand_tracker import get_pointing_target
from llm.query_engine import QueryEngine
from face.facial_recog import FaceRecognitionSystem

project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(project_root, "face"))

def authenticate_user(system: FaceRecognitionSystem, timeout: float = 30.0):
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera; check macOS Camera permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start = time.time()
    while True:
        if time.time() - start > timeout:
            cap.release()
            raise RuntimeError("Authentication timed out.")
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        name = system.recognize_once(frame)
        if name not in ("Unknown", "NoFace"):
            cap.release()
            cv2.destroyAllWindows()
            return name
        cv2.putText(frame, "Authenticating…", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Auth", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    raise RuntimeError("Authentication aborted.")

def main():
    face_sys = FaceRecognitionSystem()
    user = authenticate_user(face_sys)

    engine = QueryEngine(model="gpt-4.1-nano-2025-04-14", temperature=0.3)

    use_manual = False
    listener = None

    try:
        listener = WakeWordListener(
            keyword_paths=[pvporcupine.KEYWORD_PATHS["porcupine"]],
            sensitivities=[0.6],
            callback=lambda: None  # we won't actually use callback here
        )
        listener.porcupine  # force initialization
    except pvporcupine.PorcupineActivationLimitError:
        use_manual = True
    except Exception as e:
        use_manual = True

    if not use_manual:
        # real wake-word mode
        def on_wake():
            wav = listener._record_with_threshold()
            txt = listener.transcriber.transcribe(wav)
            label, img_path = get_pointing_target()
            ans = engine.answer(
                question=question,
                context_text=f"User pointed at {label}",
                image=img_path
            )
            print(f"Bot: {ans}\n")

        listener.callback = on_wake
        listener.start()
        print(f"\nSystem live for {user}. Say the wake-word now… (Ctrl+C to quit)\n")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down…")
        finally:
            listener.stop()

    else:
        # manual text‐entry mode
        print(f"\nSystem live for {user}. (Manual mode: type your question, or Ctrl+C to quit)\n")
        try:
            while True:
                question = input("Please type your question: ")
                if not question.strip():
                    continue
                _, img_path = get_pointing_target()
                ans = engine.answer(
                    question=question,
                    image=img_path
                )          
                print(f"Bot: {ans}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")

if __name__ == "__main__":
    main()