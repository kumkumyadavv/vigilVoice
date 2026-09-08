import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AASIST_ROOT = PROJECT_ROOT / "aasist"

if not AASIST_ROOT.exists():
    raise FileNotFoundError(
        f"AASIST repository not found at: {AASIST_ROOT}"
    )

sys.path.insert(0, str(AASIST_ROOT))

from models.AASIST import Model


AASIST_INPUT_SAMPLES = 64_600
AASIST_SAMPLE_RATE = 16_000


class VoiceDetector:

    def predict(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> float:
        raise NotImplementedError


class PretrainedDetector(VoiceDetector):

    def __init__(self):

        self.device = torch.device("cpu")

        model_config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [
                70,
                [1, 32],
                [32, 32],
                [32, 64],
                [64, 64],
            ],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
        }

        self.model = Model(model_config).to(self.device)

        checkpoint_path = (
            AASIST_ROOT
            / "models"
            / "weights"
            / "AASIST.pth"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(state_dict)

        # IMPORTANT
        self.model.eval()

    @staticmethod
    def _prepare_audio(
        audio_chunk: np.ndarray,
        sample_rate: int,
    ):

        if audio_chunk is None or len(audio_chunk) == 0:
            raise ValueError("Audio chunk is empty.")

        if sample_rate != AASIST_SAMPLE_RATE:
            raise ValueError(
                f"AASIST expects 16 kHz audio, got {sample_rate} Hz."
            )

        audio = np.asarray(
            audio_chunk,
            dtype=np.float32,
        ).reshape(-1)

        # Official AASIST-style padding/repetition
        if len(audio) < AASIST_INPUT_SAMPLES:

            num_repeats = (
                AASIST_INPUT_SAMPLES // len(audio)
            ) + 1

            audio = np.tile(
                audio,
                num_repeats,
            )

        audio = audio[:AASIST_INPUT_SAMPLES]

        return audio.astype(np.float32)

@torch.no_grad()
def predict(self, audio_chunk: np.ndarray, sample_rate: int) -> float:
    audio = self._prepare_audio(audio_chunk, sample_rate)

    waveform = torch.from_numpy(audio)
    waveform = waveform.unsqueeze(0).to(self.device)

    _, logits = self.model(waveform)

    probabilities = torch.softmax(logits, dim=1)

    # AASIST official label mapping:
    # class 0 = spoof / fake
    # class 1 = bona fide / real
    spoof_probability = probabilities[0, 0].item()

    return float(spoof_probability)