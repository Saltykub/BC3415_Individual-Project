"""
flagging.py
===========
Fuses the text signal and the image signal into one auditable risk decision.

The design goal is *defensibility*, not peak accuracy. A customer-service lead
has to be able to read the output and agree or disagree with it, so the score is
a transparent weighted sum of three named components plus a small number of
explicit business rules:

    risk = w_sentiment * sentiment_risk
         + w_keywords  * keyword_risk
         + w_image     * image_risk          (weights renormalised if a signal
                                              is missing)

Three signals, because each fails differently:
  * sentiment_risk  - catches unhappy reviews phrased in ways no keyword list
                      anticipates ("I expected so much more than this").
  * keyword_risk    - catches concrete, actionable problems even when the review
                      is written politely ("lovely colour, but it never arrived").
  * image_risk      - catches problems the customer did not put into words, and
                      is the only signal that can contradict the text.

Business rules on top of the score (this is where the creativity in combining
the two modalities actually lives):
  R1 Independent confirmation - text AND photo both indicate a problem, so the
     order is escalated to HIGH even if neither alone crossed the threshold.
     Two independent sensors agreeing is much stronger evidence than one loud one.
  R2 Contradiction - a positive review with a damaged-looking photo. Most likely
     the customer is being kind, or uploaded the wrong photo. Either way a human
     should look, so the order is floored at MEDIUM.
  R3 Critical keyword - "never arrived" / "wrong item" / "broken" is actionable
     regardless of tone, so it also floors the order at MEDIUM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import config

# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------
# Compiled once at import. Word boundaries stop "stain" matching "stainless"
# and "refund" matching "refunded"-adjacent nonsense inside other words.
_KEYWORD_PATTERNS: Dict[str, List[tuple]] = {
    issue_type: [
        (phrase, re.compile(r"\b" + re.escape(phrase) + r"\b"))
        for phrase in spec["phrases"]
    ]
    for issue_type, spec in config.ISSUE_KEYWORDS.items()
}


@dataclass
class KeywordHit:
    """One matched complaint phrase."""

    issue_type: str
    phrase: str
    severity: float


def detect_issue_keywords(text: str) -> List[KeywordHit]:
    """
    Find complaint phrases in the review.

    Matching happens on the cleaned text (lowercased, contractions expanded),
    which is why config stores phrases like "didnt arrive" alongside
    "did not arrive" - cleaning turns "didn't" into "did not", and the compact
    form survives for raw input that skipped cleaning.
    """
    import preprocessing

    cleaned = preprocessing.clean_text(text)
    hits: List[KeywordHit] = []

    for issue_type, patterns in _KEYWORD_PATTERNS.items():
        severity = config.ISSUE_KEYWORDS[issue_type]["severity"]
        for phrase, pattern in patterns:
            if pattern.search(cleaned):
                hits.append(KeywordHit(issue_type, phrase, severity))
                break   # one hit per issue type is enough; avoids double counting
    return hits


def keyword_risk_from_hits(hits: Sequence[KeywordHit]) -> float:
    """
    Collapse keyword hits into one 0..1 score.

    max(severity) is the base, because the worst problem mentioned determines
    urgency. Each *additional distinct* problem type adds 0.15, because a review
    complaining about damage AND delivery is a worse order than one complaining
    about damage alone.
    """
    if not hits:
        return 0.0
    base = max(hit.severity for hit in hits)
    extra = 0.15 * (len({hit.issue_type for hit in hits}) - 1)
    return float(min(1.0, base + extra))


# ---------------------------------------------------------------------------
# Sentiment component
# ---------------------------------------------------------------------------
def sentiment_risk_from_probabilities(probabilities: Mapping[str, float]) -> float:
    """
    Turn a probability distribution into one risk number.

    risk = P(negative) + NEUTRAL_RISK_FACTOR * P(neutral)

    Using the full distribution rather than the argmax label keeps useful
    information: a review that is 49% negative / 51% neutral is clearly riskier
    than one that is 2% negative, yet both are labelled "neutral".
    """
    negative = float(probabilities.get("negative", 0.0))
    neutral = float(probabilities.get("neutral", 0.0))
    return float(min(1.0, negative + config.NEUTRAL_RISK_FACTOR * neutral))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class RiskAssessment:
    """The full, explainable decision for one order."""

    risk_score: float
    risk_level: str
    action: str
    reasons: List[str] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    rules_fired: List[str] = field(default_factory=list)
    sentiment: Dict[str, object] = field(default_factory=dict)
    keyword_hits: List[KeywordHit] = field(default_factory=list)
    image: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Flat dict for CSV export (used by the batch tab and demo CLI)."""
        row: Dict[str, object] = {
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "sentiment": self.sentiment.get("label"),
            "sentiment_confidence": round(
                float(self.sentiment.get("confidence", 0.0)), 4),
            "sentiment_risk": round(self.components.get("sentiment", 0.0), 4),
            "keyword_risk": round(self.components.get("keywords", 0.0), 4),
            "image_risk": round(self.components.get("image", 0.0), 4),
            "issues_found": ", ".join(sorted({h.issue_type
                                              for h in self.keyword_hits})),
            "rules_fired": ", ".join(self.rules_fired),
            "reasons": " | ".join(self.reasons),
            "recommended_action": self.action,
        }
        row.update({k: v for k, v in self.image.items()
                    if k.startswith("image_")})
        return row


def level_from_score(score: float) -> str:
    """Map a 0..1 score onto LOW / MEDIUM / HIGH using config thresholds."""
    if score >= config.RISK_THRESHOLDS["high"]:
        return "HIGH"
    if score >= config.RISK_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# The fusion itself
# ---------------------------------------------------------------------------
def assess_order(text: str,
                 sentiment_result: Mapping[str, object],
                 image_verdict=None,
                 weights: Mapping[str, float] | None = None) -> RiskAssessment:
    """
    Combine the signals into a RiskAssessment.

    Parameters
    ----------
    text
        The raw review text (used for keyword matching).
    sentiment_result
        Output of sentiment.SentimentPredictor.predict_one().
    image_verdict
        An image_checker.ImageVerdict, or None when no photo was uploaded.
    weights
        Optional override of config.RISK_WEIGHTS, so a business user can retune
        the blend (the Streamlit sidebar exposes exactly this).
    """
    weights = dict(weights or config.RISK_WEIGHTS)

    probabilities = sentiment_result.get("probabilities", {})
    sentiment_risk = sentiment_risk_from_probabilities(probabilities)

    hits = detect_issue_keywords(text)
    keyword_risk = keyword_risk_from_hits(hits)

    has_image = image_verdict is not None and getattr(image_verdict, "checked", False)
    image_risk = float(image_verdict.image_risk) if has_image else 0.0

    # Renormalise the weights over the signals we actually have, so a review
    # with no photo is not automatically scored lower than one with a photo.
    active = {"sentiment": weights.get("sentiment", 0.0),
              "keywords": weights.get("keywords", 0.0)}
    if has_image:
        active["image"] = weights.get("image", 0.0)
    total_weight = sum(active.values()) or 1.0
    normalised = {name: w / total_weight for name, w in active.items()}

    components = {"sentiment": sentiment_risk, "keywords": keyword_risk,
                  "image": image_risk}
    score = sum(normalised[name] * components[name] for name in normalised)
    score = float(min(1.0, max(0.0, score)))

    # ---- reasons (always populated, in order of importance) --------------
    reasons: List[str] = []
    label = str(sentiment_result.get("label", "unknown"))
    confidence = float(sentiment_result.get("confidence", 0.0))
    reasons.append(f"Review sentiment: {label} ({confidence:.0%} confidence)")

    if hits:
        grouped: Dict[str, List[str]] = {}
        for hit in hits:
            grouped.setdefault(hit.issue_type, []).append(hit.phrase)
        for issue_type, phrases in grouped.items():
            pretty = issue_type.replace("_", " ")
            reasons.append(f"Mentions {pretty}: \"{phrases[0]}\"")

    if has_image:
        reasons.extend(image_verdict.reasons)
    else:
        reasons.append("No photo supplied, so the decision rests on text alone")

    # ---- business rules --------------------------------------------------
    rules_fired: List[str] = []
    text_risk = max(sentiment_risk, keyword_risk)

    # R1: both modalities independently point at a problem.
    if (has_image
            and text_risk >= config.CONFIRM_TEXT_RISK
            and image_risk >= config.CONFIRM_IMAGE_RISK):
        score = max(score, config.CONFIRM_FLOOR_SCORE)
        rules_fired.append("R1_independent_confirmation")
        reasons.insert(0, "ESCALATED: the review and the photo independently "
                          "confirm a problem")

    # R2: the photo contradicts a happy review.
    if (has_image
            and label == "positive"
            and image_verdict.damage_score >= config.CLIP_DAMAGE_THRESHOLD):
        score = max(score, config.CONTRADICTION_FLOOR_SCORE)
        rules_fired.append("R2_positive_text_damaged_photo")
        reasons.insert(0, "FLAGGED FOR REVIEW: the review is positive but the "
                          "photo looks damaged - a human should confirm")

    # R3: an actionable complaint regardless of tone.
    critical = sorted({hit.issue_type for hit in hits}
                      & config.CRITICAL_ISSUE_TYPES)
    if critical:
        score = max(score, config.RISK_THRESHOLDS["medium"])
        rules_fired.append("R3_critical_keyword")
        reasons.insert(0, "Actionable complaint detected: "
                          + ", ".join(c.replace("_", " ") for c in critical))

    level = level_from_score(score)

    return RiskAssessment(
        risk_score=score,
        risk_level=level,
        action=config.RISK_ACTIONS[level],
        reasons=reasons,
        components=components,
        weights=normalised,
        rules_fired=rules_fired,
        sentiment=dict(sentiment_result),
        keyword_hits=hits,
        image=image_verdict.to_dict() if has_image else {"image_checked": False},
    )


def explain_score(assessment: RiskAssessment) -> str:
    """
    Render the arithmetic as a one-line audit trail, e.g.

        0.47*0.82 + 0.29*0.00 + 0.24*0.61 = 0.53  ->  MEDIUM

    Printing the maths is the cheapest possible form of model governance: anyone
    can check the number by hand.
    """
    parts = [f"{assessment.weights[name]:.2f}*{assessment.components[name]:.2f}"
             for name in assessment.weights]
    return (f"{' + '.join(parts)} = {assessment.risk_score:.2f}"
            f"  ->  {assessment.risk_level}")


if __name__ == "__main__":
    # Self-contained demo with a fake sentiment result, so this runs without
    # any trained model present.
    fake_positive = {"label": "positive", "confidence": 0.93,
                     "probabilities": {"negative": 0.03, "neutral": 0.04,
                                       "positive": 0.93}}
    assessment = assess_order(
        "Beautiful dress and lovely colour, but mine arrived with a torn seam.",
        fake_positive,
    )
    print(explain_score(assessment))
    for reason in assessment.reasons:
        print("  -", reason)
    print("Action:", assessment.action)
