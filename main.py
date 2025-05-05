from dotenv import load_dotenv
load_dotenv()

import os, time, cv2, pvporcupine
from face.facial_recog   import FaceRecognitionSystem
from audio.listener      import WakeWordListener
from vision.hand_tracker import get_pointing_target
from llm.query_engine    import QueryEngine
import os

project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(project_root, "face"))
from face.facial_recog import FaceRecognitionSystem

print("▶️ PICOVOICE_KEY =", os.getenv("PICOVOICE_KEY"))

def authenticate_user(system: FaceRecognitionSystem, timeout: float = 30.0):
    print(f"🔍 Known users: {system.known_names}")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera; check macOS Camera permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start = time.time()
    print("👤 Position yourself for authentication…")
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
            print(f"✅ Authenticated as {name}")
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
    # 1️⃣ Face-auth
    face_sys = FaceRecognitionSystem()
    user = authenticate_user(face_sys)

    # 2️⃣ LLM engine
    engine = QueryEngine(model="gpt-3.5-turbo", temperature=0.3)

    # 3️⃣ Try wake-word first
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
        print("⚠️  Picovoice key invalid/limit hit. Skipping wake-word → manual text entry.")
        use_manual = True
    except Exception as e:
        print("⚠️  Porcupine init failed:", e, "\n→ Skipping wake-word.")
        use_manual = True

    if not use_manual:
        # real wake-word mode
        def on_wake():
            print("\n🔔 Wake word detected!")
            wav = listener._record_with_threshold()
            print(f"📝 Recorded to {wav}")
            txt = listener.transcriber.transcribe(wav)
            print(f"💬 You said: {txt}")
            label, img_path = get_pointing_target()
            ans = engine.answer(
                question=question,
                context_text=f"User pointed at {label}",
                image=img_path
            )
            print(f"🤖 Bot: {ans}\n")

        listener.callback = on_wake
        listener.start()
        print(f"\nSystem live for {user}. Say the wake-word now… (Ctrl+C to quit)\n")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down…")
        finally:
            listener.stop()

    else:
        # manual text‐entry mode
        print(f"\nSystem live for {user}. (Manual mode: type your question, or Ctrl+C to quit)\n")
        try:
            while True:
                question = input("📢 Your question: ")
                if not question.strip():
                    continue
                _, img_path = get_pointing_target()
                ans = engine.answer(
                    question=question,
                    image=img_path
                )          
                print(f"🤖 Bot: {ans}\n")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()