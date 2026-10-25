# Automated Review Triage — text sentiment + photo checks + order flagging

A complete, runnable pipeline that reads a customer review, inspects the photo the
customer uploaded, and produces one auditable risk decision for the order.

| Stage | What it does | Technology |
|---|---|---|
| Preprocessing | merge title + body, strip HTML/URLs, expand contractions, drop short/duplicate reviews, stratified 70/15/15 split | pandas, scikit-learn |
| Text sentiment | 3 classes (negative / neutral / positive) | TF-IDF + Complement NB, TF-IDF + balanced Logistic Regression, fine-tuned DistilBERT |
| Explainability | top words per class (global) + LIME (local, per prediction) | scikit-learn coefficients, LIME |
| Image checks | is the item damaged/stained? does the photo match the ordered category? | CLIP `openai/clip-vit-base-patch32`, zero-shot with prompt ensembling |
| Order flagging | transparent 0–1 risk score, LOW/MEDIUM/HIGH, reasons, recommended action | weighted blend + explicit business rules |
| Interface | single-order check, batch CSV, model insights, about | Streamlit |

---

## 1. Setup

```bash
# from the project folder
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Then put the dataset in place:

1. Download **Women's E-Commerce Clothing Reviews** from Kaggle
   (`nicapotato/womens-ecommerce-clothing-reviews`).
2. Save the CSV as **`data/reviews.csv`**.

The file must contain the columns `Title`, `Review Text` and `Rating`. If your
copy names them differently, change `TITLE_COLUMN` / `REVIEW_COLUMN` /
`RATING_COLUMN` in `config.py` — nothing else needs touching.

> First run only: DistilBERT (~260 MB) and CLIP (~600 MB) are downloaded from
> the Hugging Face hub and cached in `~/.cache/huggingface`.

---

## 2. Run, in order

```bash
# A. Fast path — baselines only (recommended first run)
python train.py --skip-bert

# B. Full path — adds the fine-tuned DistilBERT
python train.py

# C. Sanity check from the command line
python demo.py --examples
python demo.py --text "It arrived torn and the colour is wrong" --product Dress
python demo.py --text "Love it!" --image path\to\photo.jpg --product Dress
python demo.py --batch data/sample_batch.csv --out flagged.csv

# D. The web interface
streamlit run app.py
```

### Expected run times on a laptop CPU

Measured on a Windows laptop, 8 torch threads, no GPU. DistilBERT uses **dynamic
(per-batch) padding**, which roughly halves CPU training time versus padding
everything to 160 tokens.

| Command | Time |
|---|---|
| `python train.py --skip-bert` | **~30–60 s** (preprocessing ~5 s, two baselines with hyper-parameter sweeps ~20 s, evaluation + LIME artefacts ~15 s) |
| `python train.py` (default 4 000-row subsample, 3 epochs) | **~10–18 min**, plus a one-off ~260 MB model download |
| `python train.py --bert-subsample 2000 --bert-epochs 2` | **~4–7 min** |
| `python train.py --bert-subsample 0` (all ~16 k rows, 3 epochs) | **~50–80 min** — only worth it on a GPU |
| `python demo.py --examples` | ~3 s (text only) |
| First photo analysed | **~12 s** to load CLIP (plus a one-off ~600 MB download), then **~0.15 s per image** |
| LIME explanation (800 samples) | ~2 s for a baseline, ~4–8 s for DistilBERT on CPU |

Useful flags:

```bash
python train.py --skip-bert          # baselines only
python train.py --skip-baselines     # DistilBERT only
python train.py --bert-subsample 0   # no subsample, use every training row
python train.py --bert-epochs 2
python train.py --no-explain         # skip LIME / top-words artefacts
python train.py --csv path/to/other_reviews.csv
```

---

## 3. Project structure

```
config.py            every path, column name, hyper-parameter and business threshold
preprocessing.py     cleaning (shared by training AND inference), stratified splits
baseline_model.py    TF-IDF + ComplementNB / LogisticRegression, top-words extraction
bert_model.py        DistilBERT fine-tuning: hand-written loop, class-weighted loss,
                     best-epoch checkpointing on validation macro F1
sentiment.py         ONE predict_proba interface for every text model
evaluate.py          macro F1 / per-class F1 / accuracy / confusion matrices, comparison CSV
explain.py           global top words + local LIME explanations (HTML)
image_checker.py     CLIP zero-shot: condition check + misdelivery check
flagging.py          risk score = weighted blend of 3 signals + business rules
train.py             runs the whole pipeline end to end
demo.py              reusable flag_order() / flag_batch() + CLI
app.py               Streamlit interface (4 tabs)

data/
  reviews.csv         <- you provide this
  sample_batch.csv    10 example rows for the batch tab
artifacts/            everything training produces
  models/             tfidf_logreg.joblib, tfidf_complement_nb.joblib, distilbert/
  reports/            model_comparison.csv, classification_*.txt, top_words.json,
                      preprocessing_summary.txt, best_model.json
  figures/            confusion_matrix_*.png
  explanations/        lime_*.html
  data/               train.csv / val.csv / test.csv (the exact splits used)
```

---

## 4. How the risk score works

```
risk = w_sentiment · sentiment_risk
     + w_keywords  · keyword_risk
     + w_image     · image_risk        (weights renormalised if a signal is missing)

sentiment_risk = P(negative) + 0.4 · P(neutral)
keyword_risk   = max(severity of matched issue types) + 0.15 per extra issue type
image_risk     = max(P(damaged)+P(stained),  misdelivery margin)
```

Then three explicit business rules, which is where the two modalities genuinely
combine rather than just being averaged:

| Rule | Trigger | Effect |
|---|---|---|
| **R1 Independent confirmation** | text risk ≥ 0.50 **and** image risk ≥ 0.45 | score floored at 0.80 → **HIGH** |
| **R2 Contradiction** | sentiment is *positive* **but** the photo looks damaged | score floored at 0.40 → at least **MEDIUM**, flagged for human review |
| **R3 Critical keyword** | "never arrived", "wrong item", "broken", … | score floored at 0.35 → at least **MEDIUM** regardless of tone |

Levels: **HIGH** ≥ 0.65, **MEDIUM** ≥ 0.35, otherwise **LOW**. Every weight and
threshold lives in `config.py`, and the Streamlit sidebar lets you retune the
weights live. Each decision prints its own arithmetic
(`0.40*0.93 + 0.25*1.00 + 0.35*0.61 = 0.82 -> HIGH`), so a human can check it by hand.

---

## 5. Swapping in a different dataset

1. Point `config.RAW_REVIEWS_CSV` at the new file (or pass `--csv`).
2. Update `TITLE_COLUMN`, `REVIEW_COLUMN`, `RATING_COLUMN`.
3. Update `RATING_TO_LABEL` if the new rating scale is not 1–5.
4. If the products are not clothing, replace `CLIP_CATEGORY_PROMPTS` and
   edit `ISSUE_KEYWORDS`.
5. Re-run `python train.py`.

No model code changes.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: data/reviews.csv` | download the Kaggle CSV to `data/reviews.csv` |
| App says "No trained models found" | run `python train.py --skip-bert` and refresh |
| "Run train.py first" when selecting DistilBERT | you used `--skip-bert`; run `python train.py` |
| DistilBERT fine-tuning too slow | `python train.py --bert-subsample 1500 --bert-epochs 2` |
| CLIP download blocked / offline | leave the photo uploader empty — the app renormalises the weights and gives a text-only decision |
| Batch rows show `ERROR` | the `image_path` column points at files this machine cannot read; leave the cell empty for a text-only row |

---

## 7. Known limitations

* **Labels are proxies.** Sentiment classes come from star ratings; a cheerful
  2-star review is mislabelled by construction, and 3-star "neutral" is the
  noisiest class — which is exactly where the confusion matrices show the errors.
* **CLIP is not a defect detector.** It measures image–sentence similarity and
  has never seen this retailer's catalogue. Treat the photo verdict as triage,
  not evidence.
* **Narrow domain.** Trained on women's clothing reviews; accuracy drops on other
  categories.
* **The keyword list is hand-written** — transparent and editable, but it only
  catches phrasings someone thought of, which is why it is one signal of three.
* **Thresholds are business choices**, not statistical facts.
* **No human is removed from the loop**: the system prioritises a queue, it does
  not issue refunds.

Dataset: Kaggle "Women's E-Commerce Clothing Reviews".
Models: scikit-learn, `distilbert-base-uncased`, `openai/clip-vit-base-patch32`.
