import numpy as np


CHUNK_DURATION = 3  # seconds


def split_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration: int = CHUNK_DURATION,
):
    chunk_size = sample_rate * chunk_duration

    chunks = []

    for start_sample in range(0, len(audio), chunk_size):
        end_sample = min(
            start_sample + chunk_size,
            len(audio),
        )

        chunk = audio[start_sample:end_sample]

        chunks.append({
            "start": start_sample / sample_rate,
            "end": end_sample / sample_rate,
            "audio": chunk,
        })

    return chunks