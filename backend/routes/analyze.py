
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
import tempfile
import os

from security.audit import save_audit_log
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import CallAnalysis, SuspiciousSegment

from audio.processor import preprocess_audio
from ai.voice_detector import PretrainedDetector
from ai.transcriber import Transcriber
from ai.fraud_analyzer import FraudAnalyzer
from risk.risk_engine import RiskEngine
from risk.segment_analyzer import SegmentDetector


router = APIRouter()


# --------------------------------------------------
# Load models/analyzers once
# --------------------------------------------------

voice_detector = PretrainedDetector()
transcriber = Transcriber()
fraud_analyzer = FraudAnalyzer()
risk_engine = RiskEngine()

segment_detector = SegmentDetector(
    voice_detector
)


# --------------------------------------------------
# Analyze endpoint
# --------------------------------------------------

@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    # --------------------------------------------------
    # 1. Validate file
    # --------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )

    suffix = Path(file.filename).suffix.lower()

    temp_input = None

    try:

        # --------------------------------------------------
        # 2. Save uploaded file temporarily
        # --------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix if suffix != ".wav" else ".input.wav"
        ) as temp:

            temp_input = temp.name

            content = await file.read()
            temp.write(content)

        # --------------------------------------------------
        # 3. Audio preprocessing
        # --------------------------------------------------

        processed = preprocess_audio(
            temp_input
        )

        waveform = processed["waveform"]
        sample_rate = processed["sample_rate"]
        duration = processed["duration"]

        # --------------------------------------------------
        # 4. AASIST voice detection
        # --------------------------------------------------

        spoof_probability = voice_detector.predict(
            waveform,
            sample_rate
        )

        real_probability = 1.0 - spoof_probability

        voice_risk = round(
            spoof_probability * 100,
            2
        )

        # --------------------------------------------------
        # 5. Speech transcription
        # --------------------------------------------------

        transcription = transcriber.transcribe(
            processed["wav_path"]
        )

        # --------------------------------------------------
        # 6. Fraud analysis
        # --------------------------------------------------

        fraud_analysis = fraud_analyzer.analyze(
            transcription["text"]
        )

        # --------------------------------------------------
        # 7. Suspicious segment detection
        # --------------------------------------------------

        suspicious_segments = segment_detector.detect(
            processed["waveform"],
            processed["sample_rate"]
        )

        # --------------------------------------------------
        # 8. Combined risk
        # --------------------------------------------------

        risk_result = risk_engine.calculate(
            voice_risk=voice_risk,
            fraud_risk=fraud_analysis["fraud_risk"]
        )

        # --------------------------------------------------
        # 9. Save main analysis to PostgreSQL
        # --------------------------------------------------

        analysis = CallAnalysis(
            filename=file.filename,

            duration=duration,

            sample_rate=sample_rate,

            spoof_probability=spoof_probability,

            real_probability=real_probability,

            voice_risk=voice_risk,

            fraud_probability=fraud_analysis[
                "fraud_probability"
            ],

            fraud_risk=fraud_analysis[
                "fraud_risk"
            ],

            fraud_type=fraud_analysis[
                "fraud_type"
            ],

            transcription=transcription[
                "text"
            ],

            language=transcription[
                "language"
            ],

            combined_risk=risk_result[
                "combined_risk"
            ],

            risk_level=risk_result[
                "risk_level"
            ],

            action=risk_result[
                "action"
            ]
        )

        db.add(analysis)

        db.commit()

        db.refresh(analysis)

        # --------------------------------------------------
        # 10. Save suspicious segments
        # --------------------------------------------------

        for segment in suspicious_segments:

            db_segment = SuspiciousSegment(
                analysis_id=analysis.id,

                start=segment["start"],

                end=segment["end"],

                spoof_probability=segment[
                    "spoof_probability"
                ],

                voice_risk=segment[
                    "voice_risk"
                ]
            )

            db.add(db_segment)

        db.commit()

        # --------------------------------------------------
        # 11. Save audit log
        # --------------------------------------------------

        audit_data = {
            "filename": file.filename,
            "spoof_probability": spoof_probability,
            "voice_risk": voice_risk,
            "fraud_risk": fraud_analysis["fraud_risk"],
            "combined_risk": risk_result["combined_risk"],
            "risk_level": risk_result["risk_level"],
            "action": risk_result["action"],
            "transcription": transcription["text"],
            "suspicious_segments": suspicious_segments
        }

        save_audit_log(
            db,
            analysis.id,
            audit_data
        )

        # --------------------------------------------------
        # 12. Final response
        # --------------------------------------------------

        return {

            "id": analysis.id,

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

            "suspicious_segments": suspicious_segments
        }

    # --------------------------------------------------
    # 13. Error handling
    # --------------------------------------------------

    except HTTPException:
        raise

    except Exception as e:

        # Rollback database transaction if
        # something failed after db interaction

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # --------------------------------------------------
    # 14. Cleanup
    # --------------------------------------------------

    finally:

        # Remove uploaded temporary file

        if temp_input and os.path.exists(
            temp_input
        ):
            os.remove(temp_input)

        # Remove generated WAV

        if temp_input:

            wav_path = str(
                Path(temp_input).with_suffix(
                    ".wav"
                )
            )

            if os.path.exists(wav_path):
                os.remove(wav_path)
