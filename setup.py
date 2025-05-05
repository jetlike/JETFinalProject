from setuptools import setup, find_packages

setup(
    name="JETFinalProject",
    version="0.1.0",
    packages=find_packages(),  # discovers vision/, audio/, llm/, etc.
    install_requires=[
        "opencv-python",
        "mediapipe",
        "openai",
        "openai-whisper",
        "torch",
        "torchaudio",
        "numpy",
        "pyttsx3",
        "face_recognition",
        "matplotlib",
        "pandas",
        "PyYAML",
        "seaborn",
        "tqdm",
        "ultralytics",
        "wheel",
        "timm",
        "pyaudio",
        "python-dotenv",
    ],
)
