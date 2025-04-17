import struct
import threading
import os

import pyaudio
import pvporcupine


class WakeWordListener:
    """
    A wake-word listener using Picovoice Porcupine.

    Attributes:
        keyword_paths (list[str]): Paths to Porcupine keyword files (.ppn).
        sensitivities (list[float]): Sensitivities for each keyword (0 to 1).
        callback (callable): Function to call when wake word is detected.
        audio (pyaudio.PyAudio): PyAudio interface.
        porcupine (pvporcupine.Porcupine): Porcupine engine instance.
        stream (pyaudio.Stream): Audio input stream.
        _thread (threading.Thread): Background listening thread.
        _running (bool): Flag to control the listening loop.
    """

    def __init__(
        self,
        keyword_paths,
        sensitivities=None,
        library_path=None,
        model_path=None,
        callback=None,
    ):
        # Retrieve access key from environment
        access_key = os.getenv("PICOVOICE_KEY")
        if not access_key:
            raise RuntimeError("PICOVOICE_KEY not set in environment")

        self.keyword_paths = keyword_paths
        self.sensitivities = (
            sensitivities
            if sensitivities is not None
            else [0.5] * len(keyword_paths)
        )
        self.callback = callback or (lambda: print("Wake word detected!"))

        # Initialize Porcupine with access key and keyword models
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            library_path=library_path,
            model_path=model_path,
            keyword_paths=self.keyword_paths,
            sensitivities=self.sensitivities,
        )

        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self._running = False
        self._thread = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        pcm = struct.unpack_from("h" * self.porcupine.frame_length, in_data)
        result = self.porcupine.process(pcm)
        if result >= 0:
            # Detected one of the keywords
            self.callback()
        return (None, pyaudio.paContinue)

    def start(self, device_index=None):
        """
        Start the wake-word listening loop in a background thread.

        Args:
            device_index (int, optional): Specific audio input device index.
        """
        if self._running:
            return
        self._running = True

        # Open audio stream
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

        # Keep thread alive
        def _run():
            while self._running:
                pass

        self._thread = threading.Thread(target=_run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """
        Stop the wake-word listener and release resources.
        """
        if not self._running:
            return
        self._running = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.porcupine.delete()
        self.audio.terminate()
        if self._thread is not None:
            self._thread.join()


if __name__ == "__main__":
    # Example usage
    # Make sure PICOVOICE_KEY is exported in your environment
    listener = WakeWordListener(
        keyword_paths=["/path/to/hey_bot.ppn"],  # Replace with your .ppn file path
        sensitivities=[0.6],
        callback=lambda: print("🔔 Hey Bot activated! Listening for your question...🔔"),
    )
    listener.start()
    print("👂 Listening for wake word. Press Ctrl+C to exit.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        listener.stop()
        print("👋 Stopped.")
