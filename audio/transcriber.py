import whisper
import wave
import numpy as np
import os
import traceback
from whisper import audio as whisper_audio

try:
    from faster_whisper import WhisperModel
    _HAS_FASTER_WHISPER = True
except ImportError:
    _HAS_FASTER_WHISPER = False

class Transcriber:
    def __init__(self, model_size: str = "tiny", use_faster: bool = True, device: str = "cpu", compute_type: str = "int8"):
        self.use_faster = use_faster and _HAS_FASTER_WHISPER
        self.sample_rate = whisper_audio.SAMPLE_RATE
        if self.use_faster:
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            except Exception as e:
                print("Failed to load faster-whisper model, falling back to whisper:", e)
                self.use_faster = False
        if not self.use_faster:
            try: 
                self.model = whisper.load_model(model_size)
            except AttributeError:
                raise RuntimeError(
                    "Whisper API not found. Ensure 'openai-whisper' is installed."
                )

    def transcribe(self, audio_path: str) -> str:
        if self.use_faster:
            segments, _ = self.model.transcribe(audio_path, beam_size=2, vad_filter=True, language="en")
            return " ".join([seg.text for seg in segments])

        try:
            wf = wave.open(audio_path, 'rb')
        except Exception:
            raise RuntimeError(f"Failed to open audio file: {audio_path}")
        channels = wf.getnchannels()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        wf.close()

        # validate format
        if channels != 1 or rate != self.sample_rate:
            raise RuntimeError(
                f"audio must be mono at {self.sample_rate} Hz. Got {channels} channels at {rate} Hz."
            )

        # convert PCM to float32 numpy array
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # perform transcription on raw audio
        try:
            result = self.model.transcribe(audio, language="en")
            return result.get("text", "")
        except Exception as e:
            print("Whisper transcription failed:", e)
            traceback.print_exc()
            return ""