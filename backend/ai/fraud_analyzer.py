import re


class FraudAnalyzer:

    FRAUD_PATTERNS = {
        "urgency": [
            r"\burgent\b",
            r"\bimmediately\b",
            r"\bright now\b",
            r"\basap\b",
            r"\bquickly\b",
            r"\bjaldi\b",
            r"\babhi\b",
        ],

        "otp_request": [
            r"\botp\b",
            r"\bone time password\b",
            r"\bverification code\b",
            r"\bcode\b",
        ],

        "credential_request": [
            r"\bpassword\b",
            r"\bpin\b",
            r"\bpasscode\b",
            r"\blogin\b",
            r"\busername\b",
        ],

        "financial_request": [
            r"\btransfer\b",
            r"\bpayment\b",
            r"\bmoney\b",
            r"\bbank account\b",
            r"\baccount number\b",
            r"\bupi\b",
            r"\bcard number\b",
        ],

        "threat": [
            r"\bpolice\b",
            r"\barrest\b",
            r"\blegal action\b",
            r"\bcase\b",
            r"\bfine\b",
            r"\bblock your account\b",
            r"\baccount will be blocked\b",
        ],

        "secrecy": [
            r"\bdon't tell anyone\b",
            r"\bdo not tell anyone\b",
            r"\bkeep this secret\b",
            r"\bsecret\b",
        ],
    }

    CATEGORY_WEIGHTS = {
        "urgency": 15,
        "otp_request": 30,
        "credential_request": 25,
        "financial_request": 25,
        "threat": 20,
        "secrecy": 15,
    }

    def analyze(self, text: str) -> dict:

        if not text or not text.strip():
            return {
                "fraud_probability": 0.0,
                "fraud_risk": 0.0,
                "fraud_type": "NONE",
                "indicators": []
            }

        text = text.lower()

        indicators = []
        score = 0

        for category, patterns in self.FRAUD_PATTERNS.items():

            matched = False

            for pattern in patterns:
                if re.search(pattern, text):
                    matched = True
                    break

            if matched:
                indicators.append(category)
                score += self.CATEGORY_WEIGHTS[category]

        # Cap score at 100
        score = min(score, 100)

        fraud_type = self._classify(indicators)

        return {
            "fraud_probability": round(score / 100, 4),
            "fraud_risk": float(score),
            "fraud_type": fraud_type,
            "indicators": indicators
        }

    @staticmethod
    def _classify(indicators: list[str]) -> str:

        if "otp_request" in indicators:
            return "OTP_SCAM"

        if "credential_request" in indicators:
            return "CREDENTIAL_THEFT"

        if "financial_request" in indicators:
            return "FINANCIAL_FRAUD"

        if "threat" in indicators:
            return "THREAT_OR_IMPERSONATION"

        if "secrecy" in indicators:
            return "SOCIAL_ENGINEERING"

        if "urgency" in indicators:
            return "URGENCY_MANIPULATION"

        return "NONE"