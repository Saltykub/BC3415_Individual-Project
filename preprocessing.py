"""
preprocessing.py
================
Turns the raw Kaggle CSV into clean, labelled, stratified train/val/test
splits, and exposes the same `clean_text()` used at inference time.

Why this module exists
----------------------
The single biggest source of silent bugs in an NLP project is *train/serve
skew*: text cleaned one way during training and another way in the web app.
Every component in this project (baselines, DistilBERT, LIME, Streamlit)
imports `clean_text` from here, so there is exactly one cleaning rule.

Design decisions worth calling out:
* Title and body are merged, because in this dataset the Title often carries
  the verdict ("Disappointed!") while the body carries the detail.
* Contractions are expanded before punctuation is stripped so that negation
  survives ("didn't fit" -> "did not fit").
* Stopwords are kept by default (see config.REMOVE_STOPWORDS). Negation words
  are sentiment-bearing, and TF-IDF bigrams need the function words to form
  phrases like "not worth".
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

import config

# ---------------------------------------------------------------------------
# Regex patterns compiled once at import time (cheap, called per review)
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
# Keep letters, digits, spaces and the few punctuation marks that carry tone.
_KEEP_RE = re.compile(r"[^a-z0-9\s!?']")
_REPEAT_PUNCT_RE = re.compile(r"([!?])\1{1,}")
_WS_RE = re.compile(r"\s+")


def build_stopwords() -> set:
    """
    Return an English stopword set with every negation/contrast word removed.

    Only used when config.REMOVE_STOPWORDS is True. It exists so the "keep
    negation words" requirement is enforced by code rather than by a comment.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return set(ENGLISH_STOP_WORDS) - config.NEGATION_WORDS


def clean_text(raw: object) -> str:
    """
    Normalise one review string.

    Steps, in order:
      1. coerce to str and unescape HTML entities (&amp; -> &)
      2. drop HTML tags, URLs and e-mail addresses
      3. lowercase
      4. expand contractions so negation words become standalone tokens
      5. drop characters that carry no signal, keeping ! and ?
      6. collapse "!!!!" -> "!" and all whitespace runs -> single space
    """
    if raw is None:
        return ""
    text = str(raw)
    if not text.strip() or text.lower() == "nan":
        return ""

    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = text.lower()

    # Normalise curly apostrophes first so the contraction table matches.
    text = text.replace("’", "'").replace("`", "'")
    for pattern, replacement in config.CONTRACTIONS:
        text = text.replace(pattern, replacement)

    text = _KEEP_RE.sub(" ", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    text = _WS_RE.sub(" ", text).strip()

    if config.REMOVE_STOPWORDS:
        stopwords = build_stopwords()
        text = " ".join(tok for tok in text.split() if tok not in stopwords)

    return text


def merge_title_and_body(row: pd.Series) -> str:
    """Join Title and Review Text into one field, then clean the result."""
    title = clean_text(row.get(config.TITLE_COLUMN, ""))
    body = clean_text(row.get(config.REVIEW_COLUMN, ""))
    if title and body:
        # A full stop keeps the two parts from fusing into a false bigram.
        return f"{title}. {body}"
    return title or body


# ---------------------------------------------------------------------------
# Cleaning report
# ---------------------------------------------------------------------------
@dataclass
class CleaningStats:
    """Row counts at each stage, so the pipeline is auditable."""

    rows_raw: int = 0
    dropped_missing_rating: int = 0
    dropped_unmapped_rating: int = 0
    dropped_empty_text: int = 0
    dropped_too_short: int = 0
    dropped_duplicates: int = 0
    rows_final: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        return self.rows_raw - self.rows_final

    def to_text(self) -> str:
        pct = (100 * self.rows_removed / self.rows_raw) if self.rows_raw else 0.0
        lines = [
            "Preprocessing summary",
            "---------------------",
            f"  raw rows ..................... {self.rows_raw:>7,}",
            f"  - missing rating ............. {self.dropped_missing_rating:>7,}",
            f"  - rating outside mapping ..... {self.dropped_unmapped_rating:>7,}",
            f"  - empty after cleaning ....... {self.dropped_empty_text:>7,}",
            f"  - shorter than {config.MIN_TOKENS} tokens ...... "
            f"{self.dropped_too_short:>7,}",
            f"  - exact duplicate text ....... {self.dropped_duplicates:>7,}",
            f"  = rows kept .................. {self.rows_final:>7,}"
            f"   ({self.rows_removed:,} removed, {pct:.1f}%)",
            "",
            "Class balance",
            "-------------",
        ]
        total = max(self.rows_final, 1)
        for label in config.LABELS:
            count = self.class_counts.get(label, 0)
            share = 100 * count / total
            bar = "#" * int(round(share / 2))
            lines.append(f"  {label:<9} {count:>7,}  {share:5.1f}%  {bar}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading / dataset construction
# ---------------------------------------------------------------------------
def load_raw(csv_path=None) -> pd.DataFrame:
    """Read the raw CSV and verify the columns named in config are present."""
    csv_path = csv_path or config.RAW_REVIEWS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}.\n"
            "Download the Kaggle 'Women's E-Commerce Clothing Reviews' CSV and "
            "save it as data/reviews.csv (or point config.RAW_REVIEWS_CSV "
            "somewhere else)."
        )

    df = pd.read_csv(csv_path)
    required = [config.REVIEW_COLUMN, config.RATING_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required column(s): {missing}. "
            f"Found: {list(df.columns)}. Update the *_COLUMN values in config.py."
        )
    if config.TITLE_COLUMN not in df.columns:
        # Title is optional: a dataset without it still works.
        df[config.TITLE_COLUMN] = ""
    return df


def build_dataset(csv_path=None) -> tuple[pd.DataFrame, CleaningStats]:
    """
    Clean the raw frame and return (dataset, stats).

    The returned frame has exactly three columns: text, label, label_id.
    """
    raw = load_raw(csv_path)
    stats = CleaningStats(rows_raw=len(raw))

    df = raw.copy()

    # --- labels -----------------------------------------------------------
    before = len(df)
    df = df[df[config.RATING_COLUMN].notna()]
    stats.dropped_missing_rating = before - len(df)

    df[config.RATING_COLUMN] = df[config.RATING_COLUMN].astype(int)
    before = len(df)
    df[config.LABEL_COLUMN] = df[config.RATING_COLUMN].map(config.RATING_TO_LABEL)
    df = df[df[config.LABEL_COLUMN].notna()]
    stats.dropped_unmapped_rating = before - len(df)

    # --- text -------------------------------------------------------------
    df[config.TEXT_COLUMN] = df.apply(merge_title_and_body, axis=1)

    before = len(df)
    df = df[df[config.TEXT_COLUMN].str.len() > 0]
    stats.dropped_empty_text = before - len(df)

    before = len(df)
    token_counts = df[config.TEXT_COLUMN].str.split().str.len()
    long_enough = (token_counts >= config.MIN_TOKENS) & (
        df[config.TEXT_COLUMN].str.len() >= config.MIN_CHARS
    )
    df = df[long_enough]
    stats.dropped_too_short = before - len(df)

    if config.DROP_DUPLICATES:
        before = len(df)
        # Duplicates are judged on the cleaned text, which also catches reviews
        # that differ only in casing or punctuation.
        df = df.drop_duplicates(subset=[config.TEXT_COLUMN], keep="first")
        stats.dropped_duplicates = before - len(df)

    df[config.LABEL_ID_COLUMN] = df[config.LABEL_COLUMN].map(config.LABEL_TO_ID)

    dataset = df[[config.TEXT_COLUMN, config.LABEL_COLUMN,
                  config.LABEL_ID_COLUMN]].reset_index(drop=True)

    stats.rows_final = len(dataset)
    stats.class_counts = dataset[config.LABEL_COLUMN].value_counts().to_dict()
    return dataset, stats


def split_dataset(dataset: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Stratified 70/15/15 train/val/test split.

    Stratification matters a lot here: this dataset is roughly 77% positive,
    so a random split can leave the test set with too few negatives to measure
    macro F1 reliably.
    """
    labels = dataset[config.LABEL_COLUMN]

    train_val, test = train_test_split(
        dataset,
        test_size=config.TEST_SIZE,
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )
    # VAL_SIZE is expressed as a share of the *whole* dataset, so rescale it
    # against what is left after the test split was removed.
    val_share = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_share,
        stratify=train_val[config.LABEL_COLUMN],
        random_state=config.RANDOM_SEED,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def stratified_subsample(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """
    Take at most `n` rows while preserving the class distribution.

    Used to keep DistilBERT fine-tuning inside a few minutes on a laptop CPU.
    """
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    sampled, _ = train_test_split(
        df,
        train_size=n,
        stratify=df[config.LABEL_COLUMN],
        random_state=config.RANDOM_SEED,
    )
    return sampled.reset_index(drop=True)


def prepare(csv_path=None, verbose: bool = True, save: bool = True):
    """
    End-to-end entry point: clean, split, report, optionally persist.

    Returns (splits_dict, stats). train.py and any notebook-free analysis
    script can call this one function and get everything they need.
    """
    config.ensure_dirs()
    dataset, stats = build_dataset(csv_path)
    splits = split_dataset(dataset)

    if verbose:
        print(stats.to_text())
        print("\nSplit sizes")
        print("-----------")
        for name, frame in splits.items():
            dist = frame[config.LABEL_COLUMN].value_counts(normalize=True)
            shares = "  ".join(
                f"{label}={100 * dist.get(label, 0):.1f}%"
                for label in config.LABELS
            )
            print(f"  {name:<6} {len(frame):>7,}   {shares}")
        print()

    if save:
        # Persisting the splits makes every later run reproducible and lets the
        # Streamlit app show real example reviews without re-reading the raw CSV.
        for name, frame in splits.items():
            frame.to_csv(config.SPLIT_DIR / f"{name}.csv", index=False)
        (config.REPORT_DIR / "preprocessing_summary.txt").write_text(
            stats.to_text(), encoding="utf-8"
        )

    return splits, stats


def load_splits() -> Dict[str, pd.DataFrame]:
    """Reload previously saved splits (used by evaluate.py and app.py)."""
    splits = {}
    for name in ("train", "val", "test"):
        path = config.SPLIT_DIR / f"{name}.csv"
        if path.exists():
            splits[name] = pd.read_csv(path)
    return splits


if __name__ == "__main__":
    # `python preprocessing.py` prints the cleaning report on its own.
    prepare()
