import librosa

from ai.voice_detector import PretrainedDetector

AUDIO_PATH = r"D:\sih2k26\backend\test_audio\sample_16k.wav"



# Load audio at AASIST's required sample rate
audio, sample_rate = librosa.load(
    AUDIO_PATH,
    sr=16000,
    mono=True,
)

print("Sample rate:", sample_rate)
print("Total samples:", len(audio))
print("Duration:", len(audio) / sample_rate, "seconds")


# Take first 3 seconds
chunk = audio[:3 * 16000]

detector = PretrainedDetector()

score = detector.predict(
    chunk,
    sample_rate=16000,
)

print("\nSynthetic probability:", score)