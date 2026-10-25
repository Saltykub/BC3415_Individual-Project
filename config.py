"""
config.py
=========
Single source of truth for every path, column name, hyper-parameter and
business threshold used by the project.

Why a central config?
---------------------
The assignment asks for *reusable* code: the whole pipeline must work on a
different review dataset without touching the model code. Everything that is
dataset-specific (file name, column names, rating -> label mapping) or
business-specific (risk weights, keyword lists) lives here, so swapping in a
new dataset is a one-file edit.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_DIR = ARTIFACT_DIR / "models"           # trained estimators
REPORT_DIR = ARTIFACT_DIR / "reports"         # metric tables (CSV / JSON)
FIGURE_DIR = ARTIFACT_DIR / "figures"         # confusion matrices (PNG)
EXPLAIN_DIR = ARTIFACT_DIR / "explanations"   # LIME HTML files
SPLIT_DIR = ARTIFACT_DIR / "data"             # saved train/val/test splits

# Raw dataset: Kaggle "Women's E-Commerce Clothing Reviews".
RAW_REVIEWS_CSV = DATA_DIR / "reviews.csv"

# Example CSV shipped with the repo so the Streamlit batch tab can be demoed
# even before the user has their own file.
SAMPLE_BATCH_CSV = DATA_DIR / "sample_batch.csv"


def ensure_dirs() -> None:
    """Create every output folder. Called at the start of train.py / app.py."""
    for directory in (DATA_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR,
                      EXPLAIN_DIR, SPLIT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Dataset schema  (change these blocks to swap datasets)
# ---------------------------------------------------------------------------
TITLE_COLUMN = "Title"              # short headline, often carries the verdict
REVIEW_COLUMN = "Review Text"       # main body
RATING_COLUMN = "Rating"            # 1..5 stars, used to derive the label

# Derived columns created by preprocessing.py
TEXT_COLUMN = "text"                # cleaned, title + body merged
LABEL_COLUMN = "label"              # string label
LABEL_ID_COLUMN = "label_id"        # integer label for PyTorch

# Star rating -> sentiment class. 3 stars is genuinely ambiguous, hence its own
# "neutral" class rather than being folded into positive/negative.
RATING_TO_LABEL = {1: "negative", 2: "negative", 3: "neutral",
                   4: "positive", 5: "positive"}

# Fixed label order. Used everywhere (confusion matrices, probability vectors,
# model output heads) so that index 0 always means "negative".
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
NUM_LABELS = len(LABELS)


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
MIN_TOKENS = 3          # reviews shorter than this carry almost no signal
MIN_CHARS = 12
DROP_DUPLICATES = True  # the dataset contains copy-pasted reviews

# We deliberately DO NOT strip stopwords by default: "not", "never", "no" flip
# the sentiment of a sentence and TF-IDF bigrams such as "not worth" are among
# the most useful features. The flag exists so the choice is visible/reversible.
REMOVE_STOPWORDS = False

# Words that must survive any stopword filter, if one is ever switched on.
NEGATION_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "nowhere", "neither",
    "cannot", "cant", "wont", "dont", "doesnt", "didnt", "isnt", "arent",
    "wasnt", "werent", "hadnt", "hasnt", "havent", "shouldnt", "wouldnt",
    "couldnt", "without", "hardly", "barely", "scarcely", "but", "however",
    "although", "though", "very", "too",
}

# Contractions are expanded *before* punctuation is stripped so that
# "do not" survives instead of becoming the meaningless token "dont".
# Order matters: the longest, most specific patterns come first.
CONTRACTIONS = [
    ("won't", "will not"),
    ("can't", "can not"),
    ("cannot", "can not"),
    ("shan't", "shall not"),
    ("n't", " not"),
    ("'re", " are"),
    ("'s", " is"),
    ("'d", " would"),
    ("'ll", " will"),
    ("'ve", " have"),
    ("'m", " am"),
]


# ---------------------------------------------------------------------------
# 4. Splits
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.15        # held out once, used for the final comparison table
VAL_SIZE = 0.15         # used for model selection / early stopping


# ---------------------------------------------------------------------------
# 5. Baseline text models
# ---------------------------------------------------------------------------
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),   # bigrams capture "not good", "runs small"
    min_df=2,             # ignore typos that appear once
    max_df=0.9,           # ignore near-universal words
    sublinear_tf=True,    # dampen long-review term counts
    strip_accents="unicode",
    max_features=120000,
)

# Tiny validation-set sweeps. Cheap on CPU and they measurably lift macro F1,
# which is the metric the grading rubric cares about.
NB_ALPHA_GRID = [0.1, 0.3, 1.0, 2.0]
LOGREG_C_GRID = [0.5, 1.0, 2.0, 4.0]
LOGREG_MAX_ITER = 2000


# ---------------------------------------------------------------------------
# 6. DistilBERT fine-tuning
# ---------------------------------------------------------------------------
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LENGTH = 160          # ~95th percentile of review length in tokens
BERT_BATCH_SIZE = 16
BERT_EVAL_BATCH_SIZE = 64
BERT_EPOCHS = 3
BERT_LEARNING_RATE = 3e-5
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.1
BERT_MAX_GRAD_NORM = 1.0

# CPU-friendly knobs: train on a stratified subsample. None = use everything.
BERT_TRAIN_SUBSAMPLE = 4000
BERT_VAL_SUBSAMPLE = 1000

BERT_DIR = MODEL_DIR / "distilbert"


# ---------------------------------------------------------------------------
# 7. Model registry  (one place mapping a short key -> trained artefact)
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "logreg": {
        "display": "TF-IDF + Logistic Regression",
        "kind": "sklearn",
        "path": MODEL_DIR / "tfidf_logreg.joblib",
        "linear": True,          # exposes coefficients -> top words per class
    },
    "nb": {
        "display": "TF-IDF + Complement Naive Bayes",
        "kind": "sklearn",
        "path": MODEL_DIR / "tfidf_complement_nb.joblib",
        "linear": True,
    },
    "distilbert": {
        "display": "DistilBERT (fine-tuned)",
        "kind": "transformer",
        "path": BERT_DIR,
        "linear": False,
    },
}

DEFAULT_MODEL = "logreg"   # fast, explainable, good enough for the live demo


# ---------------------------------------------------------------------------
# 8. Image checking (CLIP zero-shot)
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Condition classes. CLIP is a similarity model, so we describe each class with
# several natural sentences and average their embeddings ("prompt ensembling").
# This is far more stable than relying on a single hand-picked sentence.
CLIP_CONDITION_PROMPTS = {
    "damaged": [
        "a damaged product",
        "a broken item with a visible crack",
        "torn fabric with a hole in it",
        "a defective product with a manufacturing fault",
        "ripped clothing",
    ],
    "stained": [
        "a stained garment",
        "a dirty item with a visible mark",
        "clothing with a discoloured patch",
        "a spill stain on fabric",
    ],
    "good_condition": [
        "a product in good condition",
        "a clean new garment",
        "an item in perfect condition",
        "neatly folded new clothing",
        "someone wearing a new outfit",
    ],
}

# Classes whose probability mass counts as "the photo shows a problem".
CLIP_DAMAGE_CLASSES = ("damaged", "stained")

# Ordered product catalogue for the misdelivery check. Each category gets
# several noun phrases, again ensembled.
CLIP_CATEGORY_PROMPTS = {
    "Dress": ["a dress", "a woman's dress", "an evening gown"],
    "Top / Blouse": ["a blouse", "a woman's top", "a t-shirt"],
    "Knitwear / Sweater": ["a sweater", "a knitted jumper", "a cardigan"],
    "Trousers / Jeans": ["a pair of jeans", "trousers", "a pair of pants"],
    "Skirt": ["a skirt", "a pleated skirt"],
    "Jacket / Coat": ["a jacket", "a winter coat", "a blazer"],
    "Shoes": ["a pair of shoes", "a shoe", "a pair of boots"],
    "Bag / Accessory": ["a handbag", "a purse", "a fashion accessory"],
    "Swimwear": ["a swimsuit", "a bikini", "beachwear"],
    "Intimates": ["lingerie", "underwear", "a bra"],
}

CLIP_PROMPT_TEMPLATES = [
    "a photo of {}.",
    "a product photo of {}.",
    "a customer photo of {}.",
    "a cropped photo of {}.",
]

# Decision thresholds for the image verdict.
CLIP_DAMAGE_THRESHOLD = 0.45     # P(damaged or stained) above this => "problem"
CLIP_DAMAGE_UNSURE = 0.30        # between the two => "unclear"
CLIP_MISMATCH_MARGIN = 0.15      # best category must beat expected by this much


# ---------------------------------------------------------------------------
# 9. Order flagging
# ---------------------------------------------------------------------------
# Transparent linear blend. Weights are renormalised automatically when a
# signal is missing (e.g. no photo uploaded), so the score stays on 0..1.
RISK_WEIGHTS = {
    "sentiment": 0.40,   # how negative the model thinks the review is
    "keywords": 0.25,    # explicit complaint vocabulary
    "image": 0.35,       # what the photo appears to show
}

# Neutral reviews are mildly risky, not risk-free.
NEUTRAL_RISK_FACTOR = 0.40

RISK_THRESHOLDS = {"high": 0.65, "medium": 0.35}

# Independent-confirmation rule: when text AND image both exceed these, the
# order is escalated regardless of the blended score.
CONFIRM_TEXT_RISK = 0.50
CONFIRM_IMAGE_RISK = 0.45
CONFIRM_FLOOR_SCORE = 0.80       # score lifted to at least this on escalation

# Contradiction rule: happy words + damaged-looking photo => human must look.
CONTRADICTION_FLOOR_SCORE = 0.40

# Issue keywords grouped by problem type, each with a severity in 0..1.
# Phrases are matched on the cleaned (lowercased) text with word boundaries.
ISSUE_KEYWORDS = {
    "damage": {
        "severity": 1.00,
        "phrases": ["broken", "cracked", "torn", "ripped", "damaged",
                    "defective", "faulty", "hole in", "falling apart",
                    "stained", "stain", "frayed", "unravelled", "unraveled",
                    "came apart", "seam came", "zipper broke"],
    },
    "wrong_item": {
        "severity": 1.00,
        "phrases": ["wrong item", "wrong product", "wrong colour", "wrong color",
                    "not what i ordered", "not what was ordered",
                    "different item", "sent the wrong", "received the wrong"],
    },
    "delivery": {
        "severity": 1.00,
        "phrases": ["never arrived", "did not arrive", "didnt arrive",
                    "still waiting", "lost in transit", "arrived late",
                    "late delivery", "no tracking", "never received",
                    "missing from the package", "package was empty"],
    },
    "authenticity": {
        "severity": 0.80,
        "phrases": ["fake", "not authentic", "counterfeit", "cheap material",
                    "poor quality", "bad quality", "terrible quality",
                    "feels cheap"],
    },
    "sizing": {
        "severity": 0.60,
        "phrases": ["wrong size", "too small", "too big", "runs small",
                    "runs large", "does not fit", "doesnt fit", "did not fit",
                    "mislabelled size", "mislabeled size", "sizing is off"],
    },
    "returns": {
        "severity": 0.40,
        "phrases": ["want a refund", "refund", "sending it back", "sent it back",
                    "returned it", "return it", "exchange it", "charge back"],
    },
}

# Problem types serious enough to force at least MEDIUM review on their own.
CRITICAL_ISSUE_TYPES = {"damage", "wrong_item", "delivery"}

RISK_ACTIONS = {
    "HIGH": "Escalate now: open a quality case, contact the customer and "
            "offer a replacement or refund.",
    "MEDIUM": "Queue for human review within 24 hours before any automated "
              "reply is sent.",
    "LOW": "No action needed. Keep for trend analytics.",
}

RISK_COLOURS = {"HIGH": "#c62828", "MEDIUM": "#ef6c00", "LOW": "#2e7d32"}


# ---------------------------------------------------------------------------
# 10. Explainability
# ---------------------------------------------------------------------------
TOP_WORDS_PER_CLASS = 20
LIME_NUM_FEATURES = 12
LIME_NUM_SAMPLES = 800        # keep small so the Streamlit app stays responsive
TOP_WORDS_JSON = REPORT_DIR / "top_words.json"
COMPARISON_CSV = REPORT_DIR / "model_comparison.csv"
