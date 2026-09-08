import librosa

from audio.fragmenter import split_audio
from ai.voice_detector import PretrainedDetector


AUDIO_PATH = r"C:\Apps\kumku\Documents\Sound Recordings\Recording.m4a"


# Load audio
audio, sample_rate = librosa.load(
    AUDIO_PATH,
    sr=16000,
    mono=True,
)

# Split into 3-second chunks
chunks = split_audio(
    audio,
    sample_rate,
)

print(f"Total chunks: {len(chunks)}")

# Load model ONCE
detector = PretrainedDetector()

print("\n--- AASIST RESULTS ---")

for i, chunk in enumerate(chunks, start=1):

    score = detector.predict(
        chunk["audio"],
        sample_rate,
    )

    print(
        f"Chunk {i}: "
        f"{chunk['start']:.2f}s - "
        f"{chunk['end']:.2f}s "
        f"→ synthetic = {score:.4f}"
    )