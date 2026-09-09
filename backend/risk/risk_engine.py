class RiskEngine:

    def calculate(
        self,
        voice_risk: float,
        fraud_risk: float
    ) -> dict:

        # Voice authenticity is slightly more important
        # because detecting a cloned voice is the core objective.
        combined_risk = (
            0.7 * voice_risk +
            0.3 * fraud_risk
        )

        # Strong voice-spoof detection should not be
        # diluted just because fraud keywords are absent.
        if voice_risk >= 80:
            combined_risk = max(
                combined_risk,
                voice_risk
            )

        combined_risk = round(
            min(combined_risk, 100),
            2
        )

        if combined_risk < 40:
            risk_level = "GREEN"
            action = "ALLOW"

        elif combined_risk < 70:
            risk_level = "AMBER"
            action = "STEP_UP_AUTHENTICATION"

        else:
            risk_level = "RED"
            action = "HOLD_TRANSACTION_AND_ALERT"

        return {
            "combined_risk": combined_risk,
            "risk_level": risk_level,
            "action": action
        }