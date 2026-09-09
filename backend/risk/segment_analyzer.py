import numpy as np


class SegmentDetector:

    def __init__(
        self,
        voice_detector,
        chunk_duration=4.0,
        hop_duration=2.0,
        threshold=0.70
    ):
        self.voice_detector = voice_detector
        self.chunk_duration = chunk_duration
        self.hop_duration = hop_duration
        self.threshold = threshold

    def detect(
        self,
        waveform: np.ndarray,
        sample_rate: int
    ) -> list:

        chunk_size = int(
            self.chunk_duration * sample_rate
        )

        hop_size = int(
            self.hop_duration * sample_rate
        )

        suspicious_segments = []

        start = 0

        while start < len(waveform):

            end = start + chunk_size

            chunk = waveform[start:end]

            if len(chunk) < 1000:
                break

            spoof_probability = self.voice_detector.predict(
                chunk,
                sample_rate
            )

            start_time = start / sample_rate
            end_time = min(
                end / sample_rate,
                len(waveform) / sample_rate
            )

            if spoof_probability >= self.threshold:

                suspicious_segments.append({
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "spoof_probability": round(
                        spoof_probability,
                        4
                    ),
                    "voice_risk": round(
                        spoof_probability * 100,
                        2
                    )
                })

            start += hop_size

        return suspicious_segments