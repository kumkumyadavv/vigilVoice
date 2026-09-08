import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------
# Locate the cloned official AASIST repository
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AASIST_ROOT = PROJECT_ROOT / "aasist"

if not AASIST_ROOT.exists():
    raise FileNotFoundError(
        f"AASIST repository not found at: {AASIST_ROOT}"
    )

# Add AASIST repo to Python import path
sys.path.insert(0, str(AASIST_ROOT))


from models.AASIST import Model


# AASIST expects exactly 64,600 samples
AASIST_INPUT_SAMPLES = 64_600


class VoiceDetector:
    """
    Base interface for any voice anti-spoofing detector.
    """

    def predict(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> float:
        raise NotImplementedError


class PretrainedDetector(VoiceDetector):
    """
    Pretrained AASIST anti-spoofing detector.

    Input:
        16 kHz mono waveform

    Output:
        synthetic probability in [0, 1]
    """

    def __init__(self):
        self.device = torch.device("cpu")

        # Same architecture configuration as official AASIST.conf
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
                f"AASIST checkpoint not found: {checkpoint_path}"
            )

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def _prepare_audio(
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:

        if audio_chunk is None or len(audio_chunk) == 0:
            raise ValueError("Audio chunk is empty.")

        if sample_rate != 16_000:
            raise ValueError(
                f"AASIST expects 16 kHz audio, got {sample_rate} Hz."
            )

        audio = np.asarray(
            audio_chunk,
            dtype=np.float32,
        )

        # Flatten in case shape is (N, 1)
        audio = audio.reshape(-1)

        # AASIST expects exactly 64,600 samples.
        if len(audio) >= AASIST_INPUT_SAMPLES:
            audio = audio[:AASIST_INPUT_SAMPLES]

        else:
            # Repeat audio instead of zero-padding.
            repeats = (
                AASIST_INPUT_SAMPLES // len(audio)
            ) + 1

            audio = np.tile(
                audio,
                repeats,
            )[:AASIST_INPUT_SAMPLES]

        return audio.astype(np.float32)

    @torch.no_grad()
    def predict(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> float:

        audio = self._prepare_audio(
            audio_chunk,
            sample_rate,
        )

        waveform = torch.from_numpy(audio)

        waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        # Official AASIST forward pass
        _, logits = self.model(waveform)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        # AASIST class convention:
        # index 0 = spoof
        # index 1 = bona fide
        #
        # Therefore:
        # synthetic probability = P(spoof)
        synthetic_probability = probabilities[
            0, 0
        ].item()

        return float(synthetic_probability)