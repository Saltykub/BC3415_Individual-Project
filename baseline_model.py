"""
baseline_model.py
=================
Classical TF-IDF baselines: Complement Naive Bayes and Logistic Regression.

Why baselines at all?
---------------------
1. They set the bar a transformer has to beat. A 66M-parameter model that only
   matches a linear model is not worth its inference cost.
2. They are linear, so their coefficients *are* the explanation: we can print
   the top words per class with no extra machinery.
3. They run in ~1 second on CPU, which makes the Streamlit demo instant.

Model choices:
* ComplementNB instead of MultinomialNB because the dataset is heavily skewed
  towards positive reviews; ComplementNB was designed for exactly that.
* LogisticRegression with class_weight="balanced" so the rare negative and
  neutral classes are not drowned out, which is what macro F1 punishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

import config
import preprocessing


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def make_vectorizer() -> TfidfVectorizer:
    """TF-IDF over unigrams + bigrams, configured from config.TFIDF_PARAMS."""
    stopwords = preprocessing.build_stopwords() if config.REMOVE_STOPWORDS else None
    return TfidfVectorizer(stop_words=stopwords, **config.TFIDF_PARAMS)


def make_nb_pipeline(alpha: float) -> Pipeline:
    """TF-IDF -> ComplementNB."""
    return Pipeline([
        ("tfidf", make_vectorizer()),
        ("clf", ComplementNB(alpha=alpha)),
    ])


def make_logreg_pipeline(C: float) -> Pipeline:
    """TF-IDF -> balanced multinomial Logistic Regression."""
    return Pipeline([
        ("tfidf", make_vectorizer()),
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",   # key for macro F1 on a skewed dataset
            max_iter=config.LOGREG_MAX_ITER,
            n_jobs=None,               # liblinear-free solver; keep it simple
            solver="lbfgs",
        )),
    ])


# ---------------------------------------------------------------------------
# Training with a small validation sweep
# ---------------------------------------------------------------------------
@dataclass
class BaselineResult:
    """What a baseline training run produced."""

    key: str
    display: str
    pipeline: Pipeline
    best_param: float
    val_macro_f1: float
    sweep: List[Dict[str, float]]
    path: str


def _fit_and_score(pipeline: Pipeline, train: pd.DataFrame,
                   val: pd.DataFrame) -> float:
    """Fit on train, return macro F1 on validation."""
    pipeline.fit(train[config.TEXT_COLUMN], train[config.LABEL_COLUMN])
    preds = pipeline.predict(val[config.TEXT_COLUMN])
    return f1_score(val[config.LABEL_COLUMN], preds, average="macro",
                    labels=config.LABELS)


def train_baseline(key: str, train: pd.DataFrame, val: pd.DataFrame,
                   verbose: bool = True) -> BaselineResult:
    """
    Train one baseline, sweeping its single regularisation hyper-parameter on
    the validation set, then refit on train+val with the winning value.

    Refitting on train+val is standard practice once the hyper-parameter is
    chosen: it gives the final model ~18% more data at zero risk to the test
    set, which is never touched here.
    """
    if key == "nb":
        grid, builder = config.NB_ALPHA_GRID, make_nb_pipeline
        param_name = "alpha"
    elif key == "logreg":
        grid, builder = config.LOGREG_C_GRID, make_logreg_pipeline
        param_name = "C"
    else:
        raise ValueError(f"Unknown baseline key: {key!r}")

    sweep: List[Dict[str, float]] = []
    best_param, best_score = grid[0], -1.0

    for value in grid:
        score = _fit_and_score(builder(value), train, val)
        sweep.append({param_name: value, "val_macro_f1": round(score, 4)})
        if verbose:
            print(f"    {param_name}={value:<5} val macro F1 = {score:.4f}")
        if score > best_score:
            best_param, best_score = value, score

    # Final refit on train + val with the winning hyper-parameter.
    combined = pd.concat([train, val], ignore_index=True)
    final = builder(best_param)
    final.fit(combined[config.TEXT_COLUMN], combined[config.LABEL_COLUMN])

    meta = config.MODEL_REGISTRY[key]
    path = meta["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, path)

    if verbose:
        print(f"    best {param_name}={best_param} "
              f"(val macro F1 {best_score:.4f}) -> saved to {path.name}")

    return BaselineResult(
        key=key,
        display=meta["display"],
        pipeline=final,
        best_param=best_param,
        val_macro_f1=best_score,
        sweep=sweep,
        path=str(path),
    )


def train_all_baselines(train: pd.DataFrame, val: pd.DataFrame,
                        verbose: bool = True) -> Dict[str, BaselineResult]:
    """Train both baselines and return them keyed by registry key."""
    results = {}
    for key in ("nb", "logreg"):
        if verbose:
            print(f"  [{config.MODEL_REGISTRY[key]['display']}]")
        results[key] = train_baseline(key, train, val, verbose=verbose)
    return results


# ---------------------------------------------------------------------------
# Loading + introspection
# ---------------------------------------------------------------------------
def load_baseline(key: str) -> Pipeline:
    """Load a saved baseline pipeline, with a helpful error if it is absent."""
    path = config.MODEL_REGISTRY[key]["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python train.py` first."
        )
    return joblib.load(path)


def class_feature_weights(pipeline: Pipeline) -> Dict[str, np.ndarray]:
    """
    Return a per-class weight vector over the TF-IDF vocabulary.

    LogisticRegression exposes `coef_` directly: one weight per class per
    feature, already oriented so that higher means "pulls towards this class".

    ComplementNB exposes `feature_log_prob_`. With sklearn's default norm=False
    that array is -log P(word | complement of class), i.e. already oriented the
    same way (the classifier takes an argmax over X @ feature_log_prob_.T). We
    centre it per feature, subtracting the mean across classes, because the raw
    magnitudes are dominated by overall word frequency; centring leaves the
    *discriminative* part, which is what "top words per class" should show.
    """
    clf = pipeline.named_steps["clf"]

    if isinstance(clf, ComplementNB):
        raw = clf.feature_log_prob_
        weights = raw - raw.mean(axis=0, keepdims=True)
    else:
        weights = clf.coef_

    classes = list(clf.classes_)
    return {label: weights[classes.index(label)] for label in classes}


def top_words_per_class(pipeline: Pipeline, top_n: int | None = None
                        ) -> Dict[str, List[Dict[str, float]]]:
    """
    Most class-indicative n-grams for a linear pipeline.

    This is the cheapest useful explainability output in the whole project:
    a lecturer can sanity-check the model in five seconds by reading it.
    """
    top_n = top_n or config.TOP_WORDS_PER_CLASS
    vocabulary = np.array(pipeline.named_steps["tfidf"].get_feature_names_out())
    weights_by_class = class_feature_weights(pipeline)

    output: Dict[str, List[Dict[str, float]]] = {}
    for label, weights in weights_by_class.items():
        order = np.argsort(weights)[::-1][:top_n]
        output[label] = [
            {"term": str(vocabulary[i]), "weight": round(float(weights[i]), 4)}
            for i in order
        ]
    return output


if __name__ == "__main__":
    # Standalone run: clean the data, train both baselines, print top words.
    splits, _ = preprocessing.prepare()
    results = train_all_baselines(splits["train"], splits["val"])
    print("\nTop words (Logistic Regression)")
    for label, terms in top_words_per_class(results["logreg"].pipeline, 10).items():
        joined = ", ".join(t["term"] for t in terms)
        print(f"  {label:<9} {joined}")
