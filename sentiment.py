"""
sentiment.py
============
One interface for every text model in the project.

Why this matters
----------------
Three consumers need sentiment probabilities: evaluate.py, LIME in explain.py,
and the Streamlit app. Without a shared interface, each of them would need to
know whether it is holding a scikit-learn Pipeline or a PyTorch module, how to
batch, and what order the probability columns are in.

Every predictor here exposes the same two things:

    predictor.labels                      -> ["negative", "neutral", "positive"]
    predictor.predict_proba(list_of_str)  -> np.ndarray, shape (n, 3)

Column order always follows config.LABELS, which is what makes LIME, the
confusion matrices and the Streamlit bar chart interchangeable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence

import numpy as np

import config
import preprocessing


# ---------------------------------------------------------------------------
# Base behaviour shared by both back-ends
# ---------------------------------------------------------------------------
class SentimentPredictor:
    """Abstract base class defining the shared contract."""

    key: str = ""
    display: str = ""
    labels: List[str] = config.LABELS

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    # -- convenience wrappers built on predict_proba ------------------------
    def predict(self, texts: Sequence[str]) -> List[str]:
        """Hard labels."""
        probabilities = self.predict_proba(texts)
        return [self.labels[i] for i in probabilities.argmax(axis=1)]

    def predict_one(self, text: str) -> Dict[str, object]:
        """
        Single-review prediction shaped for the UI.

        Returns label, confidence and the full probability dict, because the
        Streamlit page shows all three.
        """
        probabilities = self.predict_proba([text])[0]
        best = int(probabilities.argmax())
        return {
            "label": self.labels[best],
            "confidence": float(probabilities[best]),
            "probabilities": {label: float(p)
                              for label, p in zip(self.labels, probabilities)},
            "model": self.display,
        }

    @staticmethod
    def _prepare(texts: Sequence[str]) -> List[str]:
        """
        Apply the *training-time* cleaning to incoming text.

        This is the single most important line in the module: without it, a raw
        review typed into the web app is distributed differently from anything
        the model saw during training.
        """
        return [preprocessing.clean_text(t) for t in texts]


# ---------------------------------------------------------------------------
# scikit-learn back-end
# ---------------------------------------------------------------------------
class SklearnSentiment(SentimentPredictor):
    """Wraps a saved TF-IDF pipeline (Naive Bayes or Logistic Regression)."""

    def __init__(self, key: str):
        import baseline_model  # imported lazily to keep app start-up light

        meta = config.MODEL_REGISTRY[key]
        self.key = key
        self.display = meta["display"]
        self.pipeline = baseline_model.load_baseline(key)
        # sklearn orders classes alphabetically; remember the mapping so we can
        # reorder its columns into config.LABELS order.
        self._model_classes = list(self.pipeline.named_steps["clf"].classes_)
        self._reorder = [self._model_classes.index(label)
                         for label in config.LABELS]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = self._prepare(texts)
        probabilities = self.pipeline.predict_proba(cleaned)
        return np.asarray(probabilities)[:, self._reorder]


# ---------------------------------------------------------------------------
# Transformer back-end
# ---------------------------------------------------------------------------
class BertSentiment(SentimentPredictor):
    """Wraps the fine-tuned DistilBERT checkpoint."""

    def __init__(self, key: str = "distilbert",
                 batch_size: int | None = None):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                 AutoTokenizer)

        meta = config.MODEL_REGISTRY[key]
        path = meta["path"]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python train.py` (without "
                "--skip-bert) to fine-tune DistilBERT first."
            )

        self.key = key
        self.display = meta["display"]
        self.batch_size = batch_size or config.BERT_EVAL_BATCH_SIZE
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device).eval()

        # Trust the checkpoint's own id2label and reorder into config.LABELS.
        id2label = {int(i): str(lbl).lower()
                    for i, lbl in self.model.config.id2label.items()}
        self._reorder = [
            next(i for i, lbl in id2label.items() if lbl == label)
            for label in config.LABELS
        ]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = self._prepare(texts)
        outputs = []

        # Manual batching: LIME sends hundreds of perturbed strings at once and
        # a single forward pass over all of them would exhaust laptop memory.
        with self._torch.no_grad():
            for start in range(0, len(cleaned), self.batch_size):
                chunk = cleaned[start:start + self.batch_size]
                encoded = self.tokenizer(
                    chunk,
                    truncation=True,
                    padding=True,
                    max_length=config.BERT_MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1)
                outputs.append(probabilities.cpu().numpy())

        stacked = np.vstack(outputs) if outputs else np.zeros((0, config.NUM_LABELS))
        return stacked[:, self._reorder]


# ---------------------------------------------------------------------------
# Factory + availability helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def get_predictor(key: str | None = None) -> SentimentPredictor:
    """
    Load a predictor by registry key. Results are cached, so repeated calls in
    the same process (e.g. a batch run in the web app) do not reload the model.
    """
    key = key or config.DEFAULT_MODEL
    if key not in config.MODEL_REGISTRY:
        raise KeyError(f"Unknown model {key!r}. "
                       f"Choose from {list(config.MODEL_REGISTRY)}.")

    kind = config.MODEL_REGISTRY[key]["kind"]
    return SklearnSentiment(key) if kind == "sklearn" else BertSentiment(key)


def available_models() -> Dict[str, str]:
    """
    Which models are actually on disk right now.

    The Streamlit app uses this to grey out DistilBERT if the user has not
    trained it yet, instead of crashing on load.
    """
    found = {}
    for key, meta in config.MODEL_REGISTRY.items():
        path = meta["path"]
        exists = path.is_dir() if meta["kind"] == "transformer" else path.exists()
        if exists:
            found[key] = meta["display"]
    return found


if __name__ == "__main__":
    # Quick smoke test: python sentiment.py
    print("Models on disk:", available_models())
    predictor = get_predictor()
    samples = [
        "Absolutely love this dress, the fabric is beautiful.",
        "It arrived with a huge tear in the seam and the colour is wrong.",
        "It is ok, nothing special for the price.",
    ]
    for text in samples:
        out = predictor.predict_one(text)
        print(f"  {out['label']:<9} ({out['confidence']:.2f})  {text[:50]}")
