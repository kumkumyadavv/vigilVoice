import numpy as np

AASIST_INPUT_SAMPLES = 64_600
SAMPLE_RATE = 16_000


def split_audio(audio: np.ndarray, sample_rate: int):
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz."
        )

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    chunks = []

    for start_sample in range(0, len(audio), AASIST_INPUT_SAMPLES):
        end_sample = start_sample + AASIST_INPUT_SAMPLES
        chunk = audio[start_sample:end_sample]

        if len(chunk) < AASIST_INPUT_SAMPLES:
            num_repeats = AASIST_INPUT_SAMPLES // len(chunk) + 1
            chunk = np.tile(chunk, num_repeats)
            chunk = chunk[:AASIST_INPUT_SAMPLES]

        chunks.append({
            "start": start_sample / sample_rate,
            "end": min(end_sample, len(audio)) / sample_rate,
            "audio": chunk,
        })

    return chunks