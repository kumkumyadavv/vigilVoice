import numpy as np

from ai.voice_detector import PretrainedDetector


def test_voice_detector_loads():

    detector = PretrainedDetector()

    sample_rate = 16_000

    # 3 seconds of test audio
    audio = np.zeros(
        sample_rate * 3,
        dtype=np.float32,
    )

    score = detector.predict(
        audio,
        sample_rate,
    )

    assert 0.0 <= score <= 1.0