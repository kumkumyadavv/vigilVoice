from faster_whisper import WhisperModel


class Transcriber:
    """
    Language-agnostic speech-to-text.

    Optimized for CPU / low-RAM systems.
    Whisper automatically detects the spoken language.
    """

    def __init__(self):
        # Lazy loading prevents FastAPI startup from blocking.
        self.model = None

        self.model_size = "tiny"
        self.device = "cpu"
        self.compute_type = "int8"

    def _load_model(self):
        if self.model is None:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
                num_workers=1,
            )

        return self.model

    def transcribe(self, audio_path: str) -> dict:

        model = self._load_model()

        segments, info = model.transcribe(
            audio_path,

            # Let Whisper detect the language automatically
            language=None,

            # Reasonable CPU accuracy/speed trade-off
            beam_size=5,

            # Ignore silence / non-speech
            vad_filter=True,

            # Helps reduce repetitive hallucinations
            condition_on_previous_text=False,

            # Normal transcription
            without_timestamps=False,

            # Deterministic decoding
            temperature=0.0,

            # Hallucination controls
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        results = []

        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue

            results.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text
            })

        full_text = " ".join(
            segment["text"]
            for segment in results
        )

        return {
            "language": info.language,
            "language_probability": round(
                info.language_probability,
                4
            ),
            "text": full_text,
            "segments": results
        }