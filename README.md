# Robot Vision & Voice MVP

This project enables a robot to recognize pointing gestures and respond to spoken questions like “Hey Bot, what is this?”

---

### How It Works

1. ** Voice Activation** — Listens for "Hey Bot" and records your question.
2. ** Transcription** — Converts your voice to text using Whisper.
3. ** Pointing Detection** — Detects your pointing gesture using MediaPipe.
4. ** Object Detection** — Determines what you're pointing at.
5. ** LLM Response** — Sends context to GPT-4 and gets a natural answer.
6. ** Output** — Responds by text or voice.