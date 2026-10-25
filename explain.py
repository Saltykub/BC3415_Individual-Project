"""
explain.py
==========
Two complementary explanations, because they answer different questions.

1. Global - `top_words_per_class()`
   "What has the model learned overall?" Read straight off the linear model's
   coefficients. Cheap, stable, and the fastest way for a reviewer to spot a
   model that has latched onto something silly.

2. Local - `explain_prediction()` via LIME
   "Why did the model label *this* review that way?" LIME perturbs the review
   (deleting random words), watches how the predicted probability moves, and
   fits a small weighted linear model around that one example. It works on any
   predictor that exposes predict_proba, which is exactly why sentiment.py
   standardises that interface - the same code explains the TF-IDF baseline and
   fine-tuned DistilBERT.

Caveat worth saying out loud in the video: LIME explains *the model*, not the
world. A high weight on a word means the model reacts to it, not that the word
causes customer dissatisfaction.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Sequence

import config
import preprocessing
import sentiment

# LIME's `split_expression` is a *separator* pattern (like the one re.split uses),
# not a token pattern - everything it matches is thrown away and what is left
# between the matches becomes a feature. We split on any run of characters that
# is not a letter, digit or apostrophe, which lines up with scikit-learn's
# default TF-IDF token pattern: the tokens LIME perturbs are then the same
# tokens the model actually has features for.
_LIME_SPLIT_EXPRESSION = r"[^a-zA-Z0-9']+"


# ---------------------------------------------------------------------------
# Global explanation
# ---------------------------------------------------------------------------
def top_words_per_class(model_key: str = "logreg", top_n: int | None = None
                        ) -> Dict[str, List[Dict[str, float]]]:
    """Most class-indicative n-grams for a linear model (delegates to baseline_model)."""
    import baseline_model

    if not config.MODEL_REGISTRY[model_key]["linear"]:
        raise ValueError(
            f"{model_key} is not a linear model; use LIME for per-prediction "
            "explanations instead."
        )
    pipeline = baseline_model.load_baseline(model_key)
    return baseline_model.top_words_per_class(pipeline, top_n)


def save_top_words(model_keys: Sequence[str] = ("logreg", "nb"),
                   verbose: bool = True) -> Dict[str, dict]:
    """
    Write top words for each linear model to artifacts/reports/top_words.json.

    The Streamlit "Model insights" tab reads this file, so the app never needs
    to load a model just to show the word lists.
    """
    config.ensure_dirs()
    payload: Dict[str, dict] = {}

    for key in model_keys:
        if not config.MODEL_REGISTRY[key]["path"].exists():
            continue
        payload[key] = {
            "display": config.MODEL_REGISTRY[key]["display"],
            "classes": top_words_per_class(key),
        }
        if verbose:
            print(f"  top words for {config.MODEL_REGISTRY[key]['display']}:")
            for label, terms in payload[key]["classes"].items():
                joined = ", ".join(t["term"] for t in terms[:8])
                print(f"    {label:<9} {joined}")

    config.TOP_WORDS_JSON.write_text(json.dumps(payload, indent=2),
                                     encoding="utf-8")
    return payload


def load_top_words() -> Dict[str, dict]:
    """Read the saved top-words file, or {} if training has not run."""
    if config.TOP_WORDS_JSON.exists():
        return json.loads(config.TOP_WORDS_JSON.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Local explanation (LIME)
# ---------------------------------------------------------------------------
def _make_explainer():
    """
    Build a LimeTextExplainer configured for our label order.

    bow=True treats each distinct word as one feature regardless of position,
    which suits a bag-of-words baseline and still gives a readable picture for
    DistilBERT.
    """
    from lime.lime_text import LimeTextExplainer

    return LimeTextExplainer(
        class_names=config.LABELS,
        split_expression=_LIME_SPLIT_EXPRESSION,
        bow=True,
        random_state=config.RANDOM_SEED,
    )


def explain_prediction(text: str, model_key: str | None = None,
                       num_features: int | None = None,
                       num_samples: int | None = None,
                       labels: Sequence[int] | None = None):
    """
    Explain one review.

    Returns a dict with:
      prediction  - the predict_one() output for the review
      html        - a self-contained LIME HTML page (embeddable in Streamlit)
      weights     - [(token, weight), ...] for the predicted class
      explanation - the raw LIME object, for anyone wanting to dig further

    `num_samples` is the cost knob: LIME runs the model that many times. 800 is
    fast for TF-IDF and tolerable for DistilBERT on CPU (a few seconds).
    """
    num_features = num_features or config.LIME_NUM_FEATURES
    num_samples = num_samples or config.LIME_NUM_SAMPLES

    predictor = sentiment.get_predictor(model_key)

    # Explain the cleaned text: that is literally the string the model scores,
    # so the highlighted tokens line up with the model's features.
    cleaned = preprocessing.clean_text(text)
    if not cleaned:
        raise ValueError("Nothing left to explain after cleaning the text.")

    prediction = predictor.predict_one(text)
    predicted_index = config.LABELS.index(prediction["label"])
    target_labels = list(labels) if labels is not None else [predicted_index]

    explainer = _make_explainer()
    explanation = explainer.explain_instance(
        cleaned,
        predictor.predict_proba,     # the shared interface pays off here
        num_features=num_features,
        num_samples=num_samples,
        labels=target_labels,
    )

    return {
        "prediction": prediction,
        "cleaned_text": cleaned,
        "html": explanation.as_html(labels=target_labels),
        "weights": explanation.as_list(label=predicted_index),
        "explanation": explanation,
    }


def save_explanation(text: str, filename: str, model_key: str | None = None,
                     **kwargs) -> str:
    """Render one LIME explanation to artifacts/explanations/<filename>."""
    config.ensure_dirs()
    result = explain_prediction(text, model_key=model_key, **kwargs)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    path = config.EXPLAIN_DIR / safe_name

    header = (
        "<div style='font-family:system-ui;padding:12px 16px;"
        "border-bottom:1px solid #ddd'>"
        f"<b>Model:</b> {result['prediction']['model']} &nbsp;|&nbsp; "
        f"<b>Prediction:</b> {result['prediction']['label']} "
        f"({result['prediction']['confidence']:.1%})<br>"
        f"<b>Review:</b> {result['cleaned_text']}</div>"
    )
    path.write_text(header + result["html"], encoding="utf-8")
    return str(path)


def save_sample_explanations(test_df, n_per_class: int = 1,
                             model_key: str | None = None,
                             verbose: bool = True) -> List[str]:
    """
    Save one LIME HTML page per sentiment class from the test set.

    Having a saved example of each class means the report/video can show a
    correct negative explanation without having to hunt for one live.
    """
    paths: List[str] = []
    for label in config.LABELS:
        subset = test_df[test_df[config.LABEL_COLUMN] == label]
        for i in range(min(n_per_class, len(subset))):
            text = subset.iloc[i][config.TEXT_COLUMN]
            filename = f"lime_{label}_{i + 1}.html"
            paths.append(save_explanation(text, filename, model_key=model_key))
            if verbose:
                print(f"    {filename}  <- true label: {label}")
    return paths


if __name__ == "__main__":
    save_top_words()
    demo = ("I was so excited but the dress arrived with a huge rip in the "
            "seam and the colour is nothing like the photo.")
    print("\nLIME ->", save_explanation(demo, "lime_manual_example.html"))
