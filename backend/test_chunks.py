import numpy as np
import soundfile as sf
import torch

from ai.voice_detector import PretrainedDetector

AUDIO_PATH = r"D:\sih2k26\backend\test_audio\clean_real_16k.wav"
AASIST_INPUT_SAMPLES = 64600

from pathlib import Path

print("TESTING FILE:", Path(AUDIO_PATH).resolve())


# Load exactly like official AASIST
audio, sample_rate = sf.read(AUDIO_PATH)

print("Sample rate:", sample_rate)
print("Shape:", audio.shape)
print("Samples:", len(audio))


# Official AASIST padding behavior
if len(audio) >= AASIST_INPUT_SAMPLES:
    audio = audio[:AASIST_INPUT_SAMPLES]
else:
    num_repeats = int(AASIST_INPUT_SAMPLES / len(audio)) + 1
    audio = np.tile(audio, num_repeats)[:AASIST_INPUT_SAMPLES]


waveform = torch.from_numpy(
    audio
).float().unsqueeze(0)


detector = PretrainedDetector()

with torch.no_grad():
    _, logits = detector.model(waveform)

probabilities = torch.softmax(logits, dim=1)

print("\n--- OFFICIAL-STYLE AASIST TEST ---")

print("logit0:", logits[0, 0].item())
print("logit1:", logits[0, 1].item())

print("prob0:", probabilities[0, 0].item())
print("prob1:", probabilities[0, 1].item())

print("\nOfficial AASIST score (class 1):",
      logits[0, 1].item())