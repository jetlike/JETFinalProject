# 🤖 Robot Vision & Voice MVP

This project enables a robot to recognize pointing gestures and respond to spoken questions like “Hey Bot, what is this?”

---

### 🧠 How It Works

1. **👂 Voice Activation** — Listens for "Hey Bot" and records your question.
2. **📝 Transcription** — Converts your voice to text using Whisper.
3. **🖐️ Pointing Detection** — Detects your pointing gesture using MediaPipe.
4. **🎯 Object Detection** — Determines what you're pointing at.
5. **🤖 LLM Response** — Sends context to GPT-4 and gets a natural answer.
6. **🗣️ Output** — Responds by text or voice.

---

### 📁 Project Structure

robot-vision-voice-mvp/
├── main.py               # Runs the full system
├── requirements.txt      # Python dependencies
├── README.md             # Overview and instructions
│
├── vision/
│   ├── hand_tracker.py       # Hand + pointing gesture detection
│   └── object_detector.py    # Object detection from pointing
│
├── audio/
│   ├── listener.py           # Wake word + audio recording
│   └── transcriber.py        # Speech-to-text with Whisper
│
├── llm/
│   └── query_engine.py       # Formats + sends LLM queries
│
├── utils/
│   └── drawing.py            # Drawing helper functions
│
└── data/samples/         # Sample images/audio for testing
