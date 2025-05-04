import sys
import os
import struct
import threading
import math
import numpy as np
import traceback
import time
from pathlib import Path
import wave

import pyaudio
import pvporcupine

# Ensure project root is on sys.path so we can import llm.query_engine
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Transcription and LLM modules
from transcriber import Transcriber
from llm.query_engine import QueryEngine

class WakeWordListener:
    """
    Wake-word listener that uses Porcupine for detection, records follow-up audio
    based on volume threshold, transcribes with Whisper, and queries GPT for answers.
    """

    def __init__(
        self,
        keyword_paths,
        sensitivities=None,
        library_path=None,
        model_path=None,
        callback=None,
    ):
        # Load wake-word access key
        access_key = os.getenv("PICOVOICE_KEY")
        if not access_key:
            raise RuntimeError("PICOVOICE_KEY not set in environment")

        # Porcupine configuration
        self.keyword_paths = keyword_paths
        self.sensitivities = sensitivities or [0.5] * len(keyword_paths)
        self.callback = callback or self._on_wake

        # Initialize Porcupine
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            library_path=library_path,
            model_path=model_path,
            keyword_paths=self.keyword_paths,
            sensitivities=self.sensitivities,
        )

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self._running = False
        self._thread = None

        # Load models once at startup
        print("DEBUG: Loading transcriber model...", flush=True)
        self.transcriber = Transcriber(model_size="tiny")
        print("DEBUG: Transcriber loaded.", flush=True)
        print("DEBUG: Loading query engine...", flush=True)
        self.query_engine = QueryEngine(model="gpt-3.5-turbo", temperature=0.3)
        print("DEBUG: Query engine ready.", flush=True)

    def _record_with_threshold(
        self,
        max_duration: float = 6.0,
        threshold_db: float = -40.0,
        silence_timeout: float = 0.8,
        filename: str = "query.wav",
    ) -> str:
        """Record audio segments above a dB threshold, stop after silence."""
        frames = []
        sr = self.porcupine.sample_rate
        fl = self.porcupine.frame_length
        max_chunks = int(sr / fl * max_duration)
        silence_chunks = int(sr / fl * silence_timeout)
        stream = self.audio.open(
            rate=sr,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=fl,
        )

        recording = False
        silent_count = 0
        for _ in range(max_chunks):
            data = stream.read(fl, exception_on_overflow=False)
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(pcm**2))
            db = 20 * math.log10(rms + 1e-6)
            if db >= threshold_db:
                recording = True
                silent_count = 0
                frames.append(data)
            elif recording:
                silent_count += 1
                if silent_count > silence_chunks:
                    break
        stream.stop_stream()
        stream.close()

        # Save WAV
        wf = wave.open(filename, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sr)
        wf.writeframes(b"".join(frames))
        wf.close()
        return filename

    def _on_wake(self):
        print("🔔 Wake word detected! Recording question…", flush=True)
        try:
            wav = self._record_with_threshold()
            print(f"📝 Recorded query to {wav}", flush=True)

            print("DEBUG: Starting transcription…", flush=True)
            t0 = time.time()
            transcript = self.transcriber.transcribe(wav)
            dt = time.time() - t0
            print(f"DEBUG: Transcription finished in {dt:.2f}s", flush=True)
            print(f"DEBUG: Transcription result: {transcript}", flush=True)
            print("💬 You said:", transcript, flush=True)

            # Save transcript
            text_file = Path(wav).with_suffix('.txt')
            text_file.write_text(transcript)
            print(f"🗒️ Saved transcription to {text_file}", flush=True)

            # Query LLM
            print("DEBUG: Sending to query engine…", flush=True)
            answer = self.query_engine.answer(question=transcript)
            print("🤖 Bot says:", answer, flush=True)
        except Exception:
            print("❌ Error during follow-up processing:", flush=True)
            traceback.print_exc()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        pcm = struct.unpack_from('h' * self.porcupine.frame_length, in_data)
        if self.porcupine.process(pcm) >= 0:
            threading.Thread(target=self.callback, daemon=True).start()
        return None, pyaudio.paContinue

    def start(self, device_index=None):
        """Start the wake-word detection loop."""
        if self._running:
            return
        self._running = True
        self.stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
            input_device_index=device_index,
            stream_callback=self._audio_callback,
        )
        self.stream.start_stream()
        self._thread = threading.Thread(target=lambda: None, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening and clean up resources."""
        if not self._running:
            return
        self._running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.porcupine.delete()
        self.audio.terminate()
        if self._thread:
            self._thread.join()


if __name__ == '__main__':
    KEYWORD_MODEL = PROJECT_ROOT / 'models' / 'hey_bot.ppn'
    if not KEYWORD_MODEL.exists():
        raise FileNotFoundError(f"Keyword model not found: {KEYWORD_MODEL}")
    listener = WakeWordListener(keyword_paths=[str(KEYWORD_MODEL)], sensitivities=[0.6])
    listener.start()
    print("👂 Listening for wake word. Press Ctrl+C to exit.", flush=True)
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        listener.stop()
        print("👋 Stopped.", flush=True)
