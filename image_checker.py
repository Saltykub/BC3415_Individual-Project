"""
image_checker.py
================
Zero-shot inspection of customer-uploaded photos with CLIP
(openai/clip-vit-base-patch32).

Why zero-shot?
--------------
There is no labelled "damaged garment" dataset in this assignment, and the
company would not have one on day one either. CLIP was trained on 400M
image-text pairs, so it can score how well an image matches an arbitrary
sentence - which lets us ask two business questions with no training data at
all:

  (a) Condition  - does the photo look damaged / stained, or in good condition?
  (b) Misdelivery - does the photo show the product category that was ordered?

Prompt ensembling
-----------------
A single sentence is a noisy classifier: "a photo of a damaged product" and
"a photo of a broken item" can disagree on the same image. For each class we
encode several sentences across several templates, L2-normalise each embedding,
average them, and renormalise. The averaged direction is a far more stable
class prototype, and this is the standard trick from the original CLIP paper.

Honesty about limits (worth saying in the video): CLIP is a similarity model,
not a defect detector. Treat its output as a triage signal that raises a human's
attention, never as proof of damage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Sequence

import numpy as np

import config


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ImageVerdict:
    """
    Everything the flagging layer and the UI need from one photo.

    `image_risk` is the single 0..1 number flagging.py consumes; the rest exists
    so the decision can be explained to a human.
    """

    condition: str = "unknown"              # damaged | stained | good_condition
    condition_label: str = "Not checked"    # human-readable verdict
    condition_scores: Dict[str, float] = field(default_factory=dict)
    damage_score: float = 0.0               # P(damaged) + P(stained)

    expected_category: str | None = None
    predicted_category: str | None = None
    category_scores: Dict[str, float] = field(default_factory=dict)
    category_match_score: float = 0.0       # P(expected category)
    mismatch_score: float = 0.0             # how much the top category wins by
    misdelivery: bool = False

    image_risk: float = 0.0
    reasons: List[str] = field(default_factory=list)
    checked: bool = False

    def to_dict(self) -> Dict[str, object]:
        """Flat dict for CSV export / JSON output."""
        return {
            "image_checked": self.checked,
            "image_condition": self.condition,
            "image_condition_label": self.condition_label,
            "image_damage_score": round(self.damage_score, 4),
            "image_expected_category": self.expected_category,
            "image_predicted_category": self.predicted_category,
            "image_category_match": round(self.category_match_score, 4),
            "image_misdelivery": self.misdelivery,
            "image_risk": round(self.image_risk, 4),
        }


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------
class ImageChecker:
    """
    Loads CLIP once and answers condition / category questions about images.

    Text prototypes are computed in the constructor, so each image costs one
    vision forward pass (~0.3 s on CPU) rather than one pass per prompt.
    """

    def __init__(self, model_name: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.model_name = model_name or config.CLIP_MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device).eval()

        # Pre-compute the two sets of class prototypes.
        self.condition_classes = list(config.CLIP_CONDITION_PROMPTS)
        self._condition_embeddings = self._build_prototypes(
            config.CLIP_CONDITION_PROMPTS
        )
        self.category_classes = list(config.CLIP_CATEGORY_PROMPTS)
        self._category_embeddings = self._build_prototypes(
            config.CLIP_CATEGORY_PROMPTS
        )

    # -- prototype construction -------------------------------------------
    def _encode_text(self, prompts: Sequence[str]):
        """
        L2-normalised CLIP text embeddings for a list of sentences.

        The text tower and its projection are called explicitly rather than via
        the convenience helper `get_text_features`, whose return type changed
        between transformers 4.x and 5.x. Going through
        `text_model` -> `text_projection` is stable across both and lands the
        text in the same joint space as the image embedding (verified to match
        CLIPModel's own `logits_per_image` to 1e-6).
        """
        torch = self._torch
        with torch.no_grad():
            inputs = self.processor(text=list(prompts), return_tensors="pt",
                                    padding=True, truncation=True).to(self.device)
            pooled = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            ).pooler_output
            features = self.model.text_projection(pooled)
        return features / features.norm(dim=-1, keepdim=True)

    def _build_prototypes(self, class_prompts: Dict[str, List[str]]):
        """
        One averaged, renormalised embedding per class (prompt ensembling).

        Each phrase is expanded through every template in
        config.CLIP_PROMPT_TEMPLATES, so "a dress" becomes four sentences.
        """
        torch = self._torch
        prototypes = []

        for phrases in class_prompts.values():
            sentences = [
                template.format(phrase)
                for phrase in phrases
                for template in config.CLIP_PROMPT_TEMPLATES
            ]
            embeddings = self._encode_text(sentences)
            averaged = embeddings.mean(dim=0)
            prototypes.append(averaged / averaged.norm())

        return torch.stack(prototypes)        # shape: (n_classes, dim)

    # -- image side --------------------------------------------------------
    def _encode_image(self, image):
        """L2-normalised CLIP image embedding for one PIL image."""
        torch = self._torch
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            # Same reasoning as _encode_text: explicit tower + projection.
            pooled = self.model.vision_model(
                pixel_values=inputs["pixel_values"]
            ).pooler_output
            features = self.model.visual_projection(pooled)
        return features / features.norm(dim=-1, keepdim=True)

    def _softmax_against(self, image_embedding, prototypes) -> np.ndarray:
        """
        Cosine similarity -> probabilities.

        CLIP's learned logit_scale (~100) is what turns small cosine gaps into
        confident probabilities; reusing it keeps our numbers on the same scale
        CLIP was calibrated with.
        """
        torch = self._torch
        # no_grad because logit_scale is a trainable Parameter: without it the
        # result carries a grad graph and .numpy() refuses to convert it.
        with torch.no_grad():
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * image_embedding @ prototypes.T
            return torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # -- public checks -----------------------------------------------------
    def check_condition(self, image) -> Dict[str, object]:
        """Probability that the photo shows a damaged / stained / good item."""
        probabilities = self._softmax_against(self._encode_image(image),
                                             self._condition_embeddings)
        scores = {cls: float(p)
                  for cls, p in zip(self.condition_classes, probabilities)}

        damage_score = sum(scores.get(cls, 0.0)
                           for cls in config.CLIP_DAMAGE_CLASSES)
        best = max(scores, key=scores.get)

        if damage_score >= config.CLIP_DAMAGE_THRESHOLD:
            label = f"Looks like a problem ({best.replace('_', ' ')})"
        elif damage_score >= config.CLIP_DAMAGE_UNSURE:
            label = "Unclear - a human should look"
        else:
            label = "Looks to be in good condition"

        return {"scores": scores, "best": best,
                "damage_score": damage_score, "label": label}

    def check_category(self, image, expected_category: str | None
                       ) -> Dict[str, object]:
        """
        Does the photo show the ordered product category?

        We compare P(expected) against P(best other category). Requiring the
        winner to beat the expected class by CLIP_MISMATCH_MARGIN avoids crying
        "misdelivery" on genuinely ambiguous photos (a dress worn under a coat).
        """
        probabilities = self._softmax_against(self._encode_image(image),
                                             self._category_embeddings)
        scores = {cls: float(p)
                  for cls, p in zip(self.category_classes, probabilities)}
        predicted = max(scores, key=scores.get)

        if not expected_category or expected_category not in scores:
            # No order context, so misdelivery cannot be judged.
            return {"scores": scores, "predicted": predicted,
                    "match_score": 0.0, "mismatch_score": 0.0,
                    "misdelivery": False, "judged": False}

        match_score = scores[expected_category]
        margin = scores[predicted] - match_score
        misdelivery = (predicted != expected_category
                       and margin >= config.CLIP_MISMATCH_MARGIN)

        return {"scores": scores, "predicted": predicted,
                "match_score": match_score,
                "mismatch_score": float(max(margin, 0.0)),
                "misdelivery": misdelivery, "judged": True}

    def check(self, image, expected_category: str | None = None) -> ImageVerdict:
        """
        Run both checks and fold them into one ImageVerdict.

        image_risk = max(damage, mismatch). Max rather than sum because these
        are two independent ways for an order to be wrong: either alone is
        enough to warrant a look, and adding them would double-count a photo
        that is both.
        """
        condition = self.check_condition(image)
        category = self.check_category(image, expected_category)

        verdict = ImageVerdict(
            condition=condition["best"],
            condition_label=condition["label"],
            condition_scores=condition["scores"],
            damage_score=condition["damage_score"],
            expected_category=expected_category,
            predicted_category=category["predicted"],
            category_scores=category["scores"],
            category_match_score=category["match_score"],
            mismatch_score=category["mismatch_score"],
            misdelivery=category["misdelivery"],
            checked=True,
        )

        mismatch_contribution = (verdict.mismatch_score
                                 if verdict.misdelivery else 0.0)
        verdict.image_risk = float(min(1.0, max(verdict.damage_score,
                                                mismatch_contribution)))

        # Human-readable reasons, surfaced directly in the UI.
        if verdict.damage_score >= config.CLIP_DAMAGE_THRESHOLD:
            verdict.reasons.append(
                f"Photo looks {verdict.condition.replace('_', ' ')} "
                f"(confidence {verdict.damage_score:.0%})"
            )
        elif verdict.damage_score >= config.CLIP_DAMAGE_UNSURE:
            verdict.reasons.append(
                f"Photo condition is unclear (damage signal "
                f"{verdict.damage_score:.0%})"
            )
        if verdict.misdelivery:
            verdict.reasons.append(
                f"Photo looks like '{verdict.predicted_category}' but the order "
                f"was '{verdict.expected_category}' - possible misdelivery"
            )
        if not verdict.reasons:
            verdict.reasons.append("Photo shows no visible problem")

        return verdict


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_checker() -> ImageChecker:
    """
    Process-wide singleton.

    CLIP is ~600 MB in memory; loading it per request would make both the CLI
    and the web app unusable. Streamlit additionally wraps this in
    st.cache_resource.
    """
    return ImageChecker()


def load_image(path_or_file):
    """Open any path / file-like object as an RGB PIL image."""
    from PIL import Image

    image = Image.open(path_or_file)
    # CLIP's processor expects 3 channels; PNG screenshots are often RGBA.
    return image.convert("RGB")


def check_image_path(path, expected_category: str | None = None) -> ImageVerdict:
    """Convenience wrapper used by demo.py and the batch tab."""
    return get_checker().check(load_image(path), expected_category)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python image_checker.py <image path> [product category]")
        raise SystemExit(1)

    category = sys.argv[2] if len(sys.argv) > 2 else None
    result = check_image_path(sys.argv[1], category)
    print(f"condition        : {result.condition_label}")
    print(f"damage score     : {result.damage_score:.3f}")
    print(f"predicted product: {result.predicted_category}")
    print(f"misdelivery      : {result.misdelivery}")
    print(f"image risk       : {result.image_risk:.3f}")
    for reason in result.reasons:
        print(f"  - {reason}")
