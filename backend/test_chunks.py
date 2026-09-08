import librosa
import torch

from audio.fragmenter import split_audio
from ai.voice_detector import PretrainedDetector


AUDIO_PATH =  r"C:\Apps\kumku\Documents\Sound Recordings\Recording.m4a"

audio, sample_rate = librosa.load(
    AUDIO_PATH,
    sr=16000,
    mono=True,
)

chunks = split_audio(audio, sample_rate)

detector = PretrainedDetector()

print("\n--- RAW AASIST OUTPUT ---")

for i, chunk in enumerate(chunks, start=1):

    waveform = torch.from_numpy(
        chunk["audio"]
    ).float().unsqueeze(0)

    with torch.no_grad():
        _, logits = detector.model(waveform)

    probabilities = torch.softmax(logits, dim=1)

    print(
        f"Chunk {i}: "
        f"{chunk['start']:.2f}s - "
        f"{chunk['end']:.2f}s | "
        f"logit0={logits[0, 0].item():.4f} | "
        f"logit1={logits[0, 1].item():.4f} | "
        f"prob0={probabilities[0, 0].item():.4f} | "
        f"prob1={probabilities[0, 1].item():.4f}"
    )