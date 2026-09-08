import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16_000


class AudioProcessingError(Exception):
    """Raised when audio preprocessing fails."""
    pass


def convert_to_wav(input_path: str) -> str:
    """
    Convert any FFmpeg-supported audio format to:
    - WAV
    - mono
    - 16-bit PCM
    - TARGET_SAMPLE_RATE
    """

    input_file = Path(input_path)

    if not input_file.exists():
        raise AudioProcessingError(
            f"Audio file not found: {input_path}"
        )

    output_file = input_file.with_suffix(".wav")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(output_file),
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    except FileNotFoundError:
        raise AudioProcessingError(
            "FFmpeg is not installed or not available in PATH."
        )

    except subprocess.CalledProcessError as e:
        raise AudioProcessingError(
            f"FFmpeg conversion failed:\n{e.stderr}"
        )

    return str(output_file)


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Load audio as a normalized mono waveform.

    Returns:
        waveform: float32 NumPy array in range approximately [-1, 1]
        sample_rate: audio sample rate
    """

    try:
        waveform, sample_rate = librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
        )

    except Exception as e:
        raise AudioProcessingError(
            f"Unable to load audio: {e}"
        )

    if waveform.size == 0:
        raise AudioProcessingError(
            "Audio file contains no samples."
        )

    waveform = waveform.astype(np.float32)

    # Peak normalization
    peak = np.max(np.abs(waveform))

    if peak > 0:
        waveform = waveform / peak

    return waveform, sample_rate


def preprocess_audio(input_path: str) -> dict:
    """
    Complete preprocessing pipeline.

    Input:
        MP3 / WAV / M4A / other FFmpeg-supported formats

    Output:
        {
            "wav_path": "...",
            "waveform": np.ndarray,
            "sample_rate": 16000,
            "duration": float
        }
    """

    wav_path = convert_to_wav(input_path)

    waveform, sample_rate = load_audio(wav_path)

    duration = len(waveform) / sample_rate

    return {
        "wav_path": wav_path,
        "waveform": waveform,
        "sample_rate": sample_rate,
        "duration": duration,
    }