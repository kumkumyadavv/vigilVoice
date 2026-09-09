import re
import numpy as np
from sentence_transformers import SentenceTransformer


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

    # Lightweight NLP intent descriptions
    NLP_INTENTS = {
        "FINANCIAL_FRAUD":
            "A scam asking someone to transfer money, make a payment, "
            "send money, or provide bank, UPI, or card details.",

        "OTP_SCAM":
            "A scam asking someone for an OTP, verification code, "
            "security code, or one time password.",

        "CREDENTIAL_THEFT":
            "An attempt to steal a password, PIN, passcode, username, "
            "login credentials, or account access.",

        "THREAT_OR_IMPERSONATION":
            "A scam involving threats, police, arrest, legal action, "
            "account blocking, or pretending to be an authority.",

        "URGENCY_MANIPULATION":
            "A scam using urgency, pressure, or immediate action "
            "to manipulate the victim.",

        "SOCIAL_ENGINEERING":
            "A suspicious social engineering conversation designed "
            "to manipulate someone into revealing sensitive information.",

        "NORMAL_CONVERSATION":
            "A normal harmless conversation with no scam, fraud, "
            "threat, credential theft, or suspicious request.",
    }

    def __init__(self):

        print("Loading lightweight NLP model...")

        self.nlp_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.intent_names = list(self.NLP_INTENTS.keys())

        self.intent_embeddings = self.nlp_model.encode(
            list(self.NLP_INTENTS.values()),
            normalize_embeddings=True
        )

        print("NLP model loaded.")

    def analyze(self, text: str) -> dict:

        if not text or not text.strip():
            return {
                "fraud_probability": 0.0,
                "fraud_risk": 0.0,
                "fraud_type": "NONE",
                "indicators": [],
                "nlp_intent": "NONE",
                "nlp_confidence": 0.0
            }

        original_text = text
        text = text.lower()

        # -------------------------
        # RULE-BASED ANALYSIS
        # -------------------------

        indicators = []
        rule_score = 0

        for category, patterns in self.FRAUD_PATTERNS.items():

            matched = False

            for pattern in patterns:
                if re.search(pattern, text):
                    matched = True
                    break

            if matched:
                indicators.append(category)
                rule_score += self.CATEGORY_WEIGHTS[category]

        rule_score = min(rule_score, 100)

        # -------------------------
        # NLP SEMANTIC ANALYSIS
        # -------------------------

        text_embedding = self.nlp_model.encode(
            [original_text],
            normalize_embeddings=True
        )[0]

        similarities = np.dot(
            self.intent_embeddings,
            text_embedding
        )

        best_index = int(np.argmax(similarities))

        nlp_intent = self.intent_names[best_index]
        best_similarity = float(similarities[best_index])

        normal_index = self.intent_names.index(
            "NORMAL_CONVERSATION"
        )

        normal_similarity = float(
            similarities[normal_index]
        )

        # Difference between strongest fraud intent
        # and normal conversation.
        fraud_margin = best_similarity - normal_similarity

        # Convert semantic signal to 0-100 score.
        nlp_score = np.clip(
            (fraud_margin + 0.10) * 250,
            0,
            100
        )

        # Normal conversation should not create fraud risk.
        if nlp_intent == "NORMAL_CONVERSATION":
            nlp_score = 0.0

        # -------------------------
        # COMBINE RULES + NLP
        # -------------------------

        fraud_risk = (
            0.60 * rule_score +
            0.40 * nlp_score
        )

        fraud_risk = round(
            min(fraud_risk, 100),
            2
        )

        fraud_probability = round(
            fraud_risk / 100,
            4
        )

        fraud_type = self._classify(
            indicators,
            nlp_intent
        )

        return {
            "fraud_probability": fraud_probability,
            "fraud_risk": fraud_risk,
            "fraud_type": fraud_type,
            "indicators": indicators,
            "nlp_intent": nlp_intent,
            "nlp_confidence": round(
                best_similarity,
                4
            )
        }

    @staticmethod
    def _classify(
        indicators: list[str],
        nlp_intent: str
    ) -> str:

        # Rules have priority because they are explicit signals.

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

        # If rules don't catch it, use NLP.
        if nlp_intent != "NORMAL_CONVERSATION":
            return nlp_intent

        return "NONE"