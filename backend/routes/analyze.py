from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tempfile
import os

from audio.processor import preprocess_audio
from ai.voice_detector import PretrainedDetector
from ai.transcriber import Transcriber
from ai.fraud_analyzer import FraudAnalyzer
from risk.risk_engine import RiskEngine
from risk.segment_analyzer import SegmentDetector


router = APIRouter()


# Load models/analyzers once
voice_detector = PretrainedDetector()
transcriber = Transcriber()
fraud_analyzer = FraudAnalyzer()
risk_engine = RiskEngine()

segment_detector = SegmentDetector(
    voice_detector
)

 

@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):

    # -----------------------------
    # 1. Validate file
    # -----------------------------
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )

    suffix = Path(file.filename).suffix.lower()

    temp_input = None

    try:

        # -----------------------------
        # 2. Save uploaded file
        # -----------------------------
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix if suffix != ".wav" else ".input.wav"
        ) as temp:

            temp_input = temp.name

            content = await file.read()
            temp.write(content)

        # -----------------------------
        # 3. Audio preprocessing
        # -----------------------------
        processed = preprocess_audio(temp_input)

        waveform = processed["waveform"]
        sample_rate = processed["sample_rate"]
        duration = processed["duration"]

        # -----------------------------
        # 4. AASIST voice detection
        # -----------------------------
        spoof_probability = voice_detector.predict(
            waveform,
            sample_rate
        )

        real_probability = 1.0 - spoof_probability

        # Convert spoof probability → voice risk
        voice_risk = round(
            spoof_probability * 100,
            2
        )

        # -----------------------------
        # 5. Speech transcription
        # -----------------------------
        transcription = transcriber.transcribe(
            processed["wav_path"]
        )

        # -----------------------------
        # 6. Fraud analysis
        # -----------------------------
        fraud_analysis = fraud_analyzer.analyze(
            transcription["text"]
        )

        suspicious_segments = segment_detector.detect(
    processed["waveform"],
    processed["sample_rate"]
)
        # -----------------------------
        # 7. Combined risk
        # -----------------------------
        risk_result = risk_engine.calculate(
            voice_risk=voice_risk,
            fraud_risk=fraud_analysis["fraud_risk"]
        )

        # -----------------------------
        # 8. Final response
        # -----------------------------
        return {
            "filename": file.filename,

            "duration": round(
                duration,
                3
            ),

            "sample_rate": sample_rate,

            "spoof_probability": round(
                spoof_probability,
                4
            ),

            "real_probability": round(
                real_probability,
                4
            ),

            "voice_risk": voice_risk,

            "fraud_analysis": fraud_analysis,

            "risk": risk_result,

            "transcription": transcription,

            "suspicious_segments": suspicious_segments,
        }

    # -----------------------------
    # 9. Error handling
    # -----------------------------
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # -----------------------------
    # 10. Cleanup
    # -----------------------------
    finally:

        # Remove uploaded temporary file
        if temp_input and os.path.exists(temp_input):
            os.remove(temp_input)

        # Remove generated WAV
        if temp_input:

            wav_path = str(
                Path(temp_input).with_suffix(".wav")
            )

            if os.path.exists(wav_path):
                os.remove(wav_path)