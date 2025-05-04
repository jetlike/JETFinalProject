# main.py
import time
import cv2
from pathlib import Path

# ← now that face/ is a package, import from it
from face.facial_recog     import FaceRecognitionSystem
from audio.listener        import WakeWordListener
from vision.hand_tracker   import get_pointing_target
from llm.query_engine      import QueryEngine


# adjust this path to wherever your .ppn lives:
KEYWORD_MODEL = Path(__file__).parent / "models" / "hey_bot.ppn"

def authenticate_user(system: FaceRecognitionSystem, timeout: float = 30.0):
    """
    Open the camera and keep grabbing frames until someone
    registered is recognized, or until timeout. Returns the name.
    """
    cap = cv2.VideoCapture(0)
    start = time.time()
    print("👤 Please position yourself in front of the camera…")
    while True:
        if time.time() - start > timeout:
            cap.release()
            raise RuntimeError("Authentication timed out")
        ret, frame = cap.read()
        if not ret:
            continue
        name = system.recognize_once(frame)
        if name not in ("Unknown", "NoFace"):
            cap.release()
            print(f"✅ Authenticated as {name}")
            return name
        # you can show the frame for feedback if you like:
        cv2.putText(frame, "Authenticating…", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Auth", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    raise RuntimeError("Authentication aborted")

def main():
    # ——— 1. Face-based authentication ——————————————————————————————
    face_sys = FaceRecognitionSystem()
    user = authenticate_user(face_sys)

    # ——— 2. Spin up the LLM engine ————————————————————————————————
    engine = QueryEngine(model="gpt-3.5-turbo", temperature=0.3)

    # ——— 3. Define what happens on wake-word —————————————————————————
    def on_wake():
        print("\n🔔 Wake word detected!")
        wav = listener._record_with_threshold()   # record question
        print(f"📝 Saved query to {wav}")
        # transcribe
        transcript = listener.transcriber.transcribe(wav)
        print(f"💬 You said: {transcript}")
        # vision
        obj = get_pointing_target()
        print(f"👆 Pointed at: {obj}")
        # query LLM
        answer = engine.answer(question=transcript, context_text=f"User pointed at {obj}")
        print(f"🤖 Bot: {answer}\n")

    # ——— 4. Start wake-word listener ————————————————————————————————
    listener = WakeWordListener(
        keyword_paths=[str(KEYWORD_MODEL)],
        sensitivities=[0.6],
        callback=on_wake
    )
    listener.start()
    print(f"\nSystem is live for {user}. Say your wake-word now… (Ctrl+C to quit)\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down…")
    finally:
        listener.stop()

if __name__ == "__main__":
    main()
