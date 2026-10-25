"""
app.py
======
Streamlit front end for the review-triage system.

    streamlit run app.py

Four tabs, mapping onto the four things a business user actually does:
  1. Single order check - the live demo: review + photo in, decision out.
  2. Batch processing    - score a day's worth of reviews from a CSV.
  3. Model insights      - the evidence: comparison table, confusion matrices,
                           top words, and a live LIME explanation.
  4. About               - plain-English description and honest limitations.

Two engineering points worth noting:
* Every model is loaded through an @st.cache_resource function, so DistilBERT
  and CLIP are loaded once per server process rather than once per interaction.
* Nothing here trains anything. If an artefact is missing the page shows a
  friendly "run train.py first" message instead of raising an exception, so a
  fresh clone never greets the user with a stack trace.
"""

from __future__ import annotations

import io
from typing import Dict, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components

import config
import demo
import evaluate
import explain
import flagging
import sentiment

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Review Triage - Text + Image Order Flagging",
    page_icon="🧵",
    layout="wide",
)

config.ensure_dirs()

# A little CSS for the risk badge and metric cards. Kept small and local so the
# app stays readable; Streamlit's own components do the rest.
st.markdown(
    """
    <style>
      .risk-badge {
          display:inline-block; padding:14px 26px; border-radius:10px;
          color:#fff; font-size:1.35rem; font-weight:700; letter-spacing:.5px;
      }
      .risk-score { font-size:.95rem; font-weight:500; opacity:.92; }
      .reason-box {
          background:#f6f7f9; border-left:4px solid #9aa4b2;
          padding:12px 16px; border-radius:6px; margin-top:6px;
      }
      .action-box {
          background:#eef5ff; border-left:4px solid #1f6feb;
          padding:12px 16px; border-radius:6px; font-weight:600;
      }
      .small-note { color:#6b7280; font-size:.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading sentiment model ...")
def load_predictor(model_key: str):
    """Load and cache one sentiment model. Keyed on the string, so it is hashable."""
    return sentiment.get_predictor(model_key)


@st.cache_resource(show_spinner="Loading CLIP image model (first run downloads ~600 MB) ...")
def load_image_checker():
    """Load and cache CLIP. Only called when the user actually uploads a photo."""
    import image_checker

    return image_checker.get_checker()


@st.cache_data(show_spinner=False)
def load_model_comparison() -> Optional[pd.DataFrame]:
    """Cached read of the model comparison CSV."""
    return evaluate.load_comparison()


@st.cache_data(show_spinner=False)
def load_top_words() -> Dict[str, dict]:
    """Cached read of the saved top-words JSON."""
    return explain.load_top_words()


def available() -> Dict[str, str]:
    """Models currently on disk (not cached: the user may train mid-session)."""
    return sentiment.available_models()


def missing_models_warning() -> bool:
    """
    Show a friendly setup message if nothing is trained yet.

    Returns True when models are missing, so callers can stop rendering.
    """
    if available():
        return False
    st.warning(
        "**No trained models found yet.**\n\n"
        "Run the training pipeline once, then refresh this page:\n\n"
        "```\npython train.py --skip-bert   # ~1 minute, baselines only\n"
        "python train.py               # adds the fine-tuned DistilBERT\n```"
    )
    return True


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------
def render_risk_badge(assessment) -> None:
    """Colour-coded HIGH / MEDIUM / LOW badge with the numeric score."""
    colour = config.RISK_COLOURS[assessment.risk_level]
    st.markdown(
        f"<div class='risk-badge' style='background:{colour}'>"
        f"{assessment.risk_level} RISK"
        f"<span class='risk-score'> &nbsp;&nbsp;score {assessment.risk_score:.2f}"
        f" / 1.00</span></div>",
        unsafe_allow_html=True,
    )


def render_probability_chart(probabilities: Dict[str, float]) -> None:
    """Horizontal-ish bar chart of the three class probabilities."""
    frame = pd.DataFrame(
        {"probability": [probabilities.get(label, 0.0) for label in config.LABELS]},
        index=config.LABELS,
    )
    st.bar_chart(frame, height=220)


def render_reasons(assessment) -> None:
    """Bullet list of the human-readable reasons behind the decision."""
    bullets = "".join(f"<li>{reason}</li>" for reason in assessment.reasons)
    st.markdown(f"<div class='reason-box'><ul>{bullets}</ul></div>",
                unsafe_allow_html=True)


def render_component_metrics(assessment) -> None:
    """The three risk components plus the audit-trail arithmetic."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment risk", f"{assessment.components['sentiment']:.2f}")
    col2.metric("Keyword risk", f"{assessment.components['keywords']:.2f}")
    col3.metric("Image risk", f"{assessment.components['image']:.2f}")
    st.caption(f"Weighted blend: {flagging.explain_score(assessment)}")


# ---------------------------------------------------------------------------
# Sidebar: global controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Review Triage")
    st.caption("Sentiment + photo checks -> one flagging decision")

    on_disk = available()
    st.subheader("Trained artefacts")
    for key, meta in config.MODEL_REGISTRY.items():
        mark = "✅" if key in on_disk else "⬜"
        st.write(f"{mark} {meta['display']}")
    if "distilbert" not in on_disk:
        st.caption("DistilBERT appears once you run `python train.py` "
                   "without `--skip-bert`.")

    st.divider()
    st.subheader("Risk weights")
    st.caption("Retune the blend to match business appetite. "
               "Weights are renormalised when no photo is supplied.")
    weight_sentiment = st.slider("Sentiment", 0.0, 1.0,
                                 config.RISK_WEIGHTS["sentiment"], 0.05)
    weight_keywords = st.slider("Issue keywords", 0.0, 1.0,
                                config.RISK_WEIGHTS["keywords"], 0.05)
    weight_image = st.slider("Image", 0.0, 1.0,
                             config.RISK_WEIGHTS["image"], 0.05)
    WEIGHTS = {"sentiment": weight_sentiment, "keywords": weight_keywords,
               "image": weight_image}

    st.divider()
    st.caption(f"Thresholds: HIGH ≥ {config.RISK_THRESHOLDS['high']:.2f}, "
               f"MEDIUM ≥ {config.RISK_THRESHOLDS['medium']:.2f}")


st.title("🧵 Automated Review Triage")
st.caption("Classify review sentiment, inspect the customer's photo, and flag "
           "problematic orders with a transparent risk score.")

tab_single, tab_batch, tab_insights, tab_about = st.tabs(
    ["🔍 Single order check", "📦 Batch processing",
     "📊 Model insights", "ℹ️ About"]
)


# ===========================================================================
# TAB 1 - Single order check
# ===========================================================================
with tab_single:
    if missing_models_warning():
        st.stop()

    st.subheader("Check one order")

    # --- one-click examples (kept above the text box so the demo flows) ----
    st.write("**Quick examples** - click one, then press *Analyse order*:")
    example_columns = st.columns(len(demo.EXAMPLE_REVIEWS))
    for column, example in zip(example_columns, demo.EXAMPLE_REVIEWS):
        if column.button(example["name"], use_container_width=True):
            # session_state is how Streamlit carries a value across the rerun
            # that a button click triggers.
            st.session_state["review_text"] = example["text"]
            st.session_state["product_choice"] = example["product"]

    left, right = st.columns([3, 2])

    with left:
        review_text = st.text_area(
            "Customer review",
            key="review_text",
            height=170,
            placeholder="Paste the review text here ...",
        )

    with right:
        product_options = ["(not specified)"] + list(config.CLIP_CATEGORY_PROMPTS)
        product_choice = st.selectbox(
            "Ordered product category",
            product_options,
            key="product_choice",
            help="Used by CLIP to detect a misdelivery (photo shows something "
                 "other than what was ordered).",
        )

        model_options = list(on_disk)
        default_index = (model_options.index(evaluate.best_model_key())
                         if evaluate.best_model_key() in model_options else 0)
        model_key = st.selectbox(
            "Sentiment model",
            model_options,
            index=default_index,
            format_func=lambda key: config.MODEL_REGISTRY[key]["display"],
        )

        uploaded_image = st.file_uploader(
            "Customer photo (optional)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
        )

    analyse = st.button("Analyse order", type="primary", use_container_width=True)

    if analyse:
        if not (review_text or "").strip():
            st.error("Please enter some review text first.")
        else:
            product = None if product_choice == "(not specified)" else product_choice

            # Image is processed here rather than inside flag_order so the UI can
            # show the photo and a CLIP-specific spinner.
            image_verdict = None
            pil_image = None
            if uploaded_image is not None:
                import image_checker

                pil_image = image_checker.load_image(uploaded_image)
                with st.spinner("Inspecting the photo with CLIP ..."):
                    image_verdict = load_image_checker().check(pil_image, product)

            with st.spinner("Classifying sentiment ..."):
                predictor = load_predictor(model_key)
                sentiment_result = predictor.predict_one(review_text)

            assessment = flagging.assess_order(
                review_text, sentiment_result, image_verdict, weights=WEIGHTS
            )

            st.divider()
            render_risk_badge(assessment)
            st.markdown(f"<div class='action-box'>Recommended action: "
                        f"{assessment.action}</div>", unsafe_allow_html=True)

            st.write("")
            col_text, col_image = st.columns([3, 2])

            with col_text:
                st.markdown("#### Sentiment")
                st.write(f"**{assessment.sentiment['label'].title()}** "
                         f"({assessment.sentiment['confidence']:.1%} confidence) "
                         f"- {assessment.sentiment['model']}")
                render_probability_chart(assessment.sentiment["probabilities"])

                if assessment.keyword_hits:
                    found = sorted({hit.issue_type.replace("_", " ")
                                    for hit in assessment.keyword_hits})
                    st.markdown("#### Issue keywords detected")
                    st.write(", ".join(f"`{item}`" for item in found))

            with col_image:
                st.markdown("#### Customer photo")
                if pil_image is not None:
                    st.image(pil_image, use_container_width=True)
                    st.write(f"**Verdict:** {image_verdict.condition_label}")
                    st.write(f"Damage signal: `{image_verdict.damage_score:.2f}`")
                    if image_verdict.predicted_category:
                        st.write("Photo looks like: "
                                 f"**{image_verdict.predicted_category}**")
                    if image_verdict.misdelivery:
                        st.error("Possible misdelivery: the photo does not match "
                                 "the ordered category.")
                    with st.expander("CLIP condition scores"):
                        st.dataframe(
                            pd.DataFrame(
                                sorted(image_verdict.condition_scores.items(),
                                       key=lambda kv: -kv[1]),
                                columns=["class", "probability"],
                            ),
                            hide_index=True, use_container_width=True,
                        )
                else:
                    st.info("No photo uploaded - the decision uses text only, "
                            "and the risk weights were renormalised accordingly.")

            st.markdown("#### Why this decision")
            render_reasons(assessment)
            st.write("")
            render_component_metrics(assessment)

            if assessment.rules_fired:
                st.markdown("#### Business rules triggered")
                for rule in assessment.rules_fired:
                    st.write(f"- `{rule}`")


# ===========================================================================
# TAB 2 - Batch processing
# ===========================================================================
with tab_batch:
    if not available():
        st.warning("Train a model first (`python train.py --skip-bert`).")
    else:
        st.subheader("Score a CSV of reviews")
        st.markdown(
            "Required column: **`text`**. Optional: **`image_path`** "
            "(a path readable by this machine) and **`product`** "
            "(one of the configured categories)."
        )

        batch_model = st.selectbox(
            "Sentiment model for the batch",
            list(available()),
            format_func=lambda key: config.MODEL_REGISTRY[key]["display"],
            key="batch_model",
        )

        uploaded_csv = st.file_uploader("Reviews CSV", type=["csv"], key="batch_csv")
        use_sample = st.checkbox(
            "Use the bundled sample file instead (data/sample_batch.csv)",
            value=False,
        )

        frame = None
        if uploaded_csv is not None:
            frame = pd.read_csv(uploaded_csv)
        elif use_sample and config.SAMPLE_BATCH_CSV.exists():
            frame = pd.read_csv(config.SAMPLE_BATCH_CSV)
        elif use_sample:
            st.error(f"{config.SAMPLE_BATCH_CSV} is missing.")

        if frame is not None:
            if "text" not in frame.columns:
                st.error("The CSV needs a column called `text`.")
            else:
                st.write(f"Loaded **{len(frame):,}** rows.")
                st.dataframe(frame.head(5), use_container_width=True)

                if st.button("Run batch", type="primary"):
                    progress_bar = st.progress(0.0, text="Scoring ...")

                    def tick(done: int, total: int) -> None:
                        progress_bar.progress(done / total,
                                              text=f"Scoring {done}/{total} ...")

                    results = demo.flag_batch(frame.to_dict("records"),
                                              model_key=batch_model,
                                              weights=WEIGHTS,
                                              progress=tick)
                    progress_bar.empty()
                    results_frame = pd.DataFrame(results)
                    st.session_state["batch_results"] = results_frame

                results_frame = st.session_state.get("batch_results")
                if results_frame is not None:
                    st.divider()

                    # --- summary counts per risk level ---------------------
                    counts = results_frame["risk_level"].value_counts()
                    metric_columns = st.columns(4)
                    for column, level in zip(metric_columns,
                                             ["HIGH", "MEDIUM", "LOW", "ERROR"]):
                        column.metric(f"{level}", int(counts.get(level, 0)))

                    # --- sortable, highlighted table -----------------------
                    display_columns = [c for c in
                                       ["row", "risk_level", "risk_score",
                                        "sentiment", "sentiment_confidence",
                                        "issues_found", "image_risk",
                                        "recommended_action", "text", "error"]
                                       if c in results_frame.columns]
                    table = results_frame[display_columns]

                    def highlight_risk(row):
                        """Tint the whole row by its risk level."""
                        tints = {"HIGH": "#fde8e8", "MEDIUM": "#fff4e5",
                                 "LOW": "#eafaf1", "ERROR": "#eceff1"}
                        colour = tints.get(row.get("risk_level"), "")
                        return [f"background-color: {colour}"] * len(row)

                    st.markdown("Click any column header to sort.")
                    st.dataframe(
                        table.style.apply(highlight_risk, axis=1),
                        use_container_width=True,
                        height=430,
                    )

                    # --- download ------------------------------------------
                    flagged = results_frame[
                        results_frame["risk_level"].isin(["HIGH", "MEDIUM"])
                    ]
                    buffer = io.StringIO()
                    flagged.to_csv(buffer, index=False)
                    st.download_button(
                        f"⬇️ Download {len(flagged):,} flagged orders (CSV)",
                        data=buffer.getvalue(),
                        file_name="flagged_orders.csv",
                        mime="text/csv",
                        type="primary",
                    )

                    all_buffer = io.StringIO()
                    results_frame.to_csv(all_buffer, index=False)
                    st.download_button(
                        "⬇️ Download all results (CSV)",
                        data=all_buffer.getvalue(),
                        file_name="all_results.csv",
                        mime="text/csv",
                    )


# ===========================================================================
# TAB 3 - Model insights
# ===========================================================================
with tab_insights:
    st.subheader("How well do the models actually do?")

    comparison = load_model_comparison()
    if comparison is None:
        st.warning("No comparison table yet. Run `python train.py` to create "
                   "`artifacts/reports/model_comparison.csv`.")
    else:
        st.markdown("**Test-set results** (same held-out split for every model, "
                    "sorted by macro F1):")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.caption("Macro F1 is the headline metric: it weights the rare "
                   "negative and neutral classes as heavily as the dominant "
                   "positive class, so a model cannot score well by predicting "
                   "\"positive\" for everything.")

        # --- confusion matrices -------------------------------------------
        figures = sorted(config.FIGURE_DIR.glob("confusion_matrix_*.png"))
        if figures:
            st.markdown("#### Confusion matrices")
            columns = st.columns(min(len(figures), 3))
            for index, figure in enumerate(figures):
                with columns[index % len(columns)]:
                    st.image(str(figure), use_container_width=True)

        # --- DistilBERT training history ----------------------------------
        import bert_model

        history = bert_model.load_training_history()
        if history:
            with st.expander("DistilBERT training history "
                             f"(best epoch {history['best_epoch']})"):
                st.dataframe(pd.DataFrame(history["history"]),
                             use_container_width=True, hide_index=True)
                st.caption(f"Trained on {history['train_rows']:,} reviews in "
                           f"{history['seconds'] / 60:.1f} minutes. The saved "
                           "checkpoint is the best epoch by validation macro F1, "
                           "not the last one.")

    st.divider()

    # --- global explanation ----------------------------------------------
    st.markdown("#### Top words per class (global explanation)")
    top_words = load_top_words()
    if not top_words:
        st.info("Run `python train.py` to generate the top-words file.")
    else:
        model_choice = st.selectbox(
            "Linear model",
            list(top_words),
            format_func=lambda key: top_words[key]["display"],
        )
        class_columns = st.columns(len(config.LABELS))
        for column, label in zip(class_columns, config.LABELS):
            terms = top_words[model_choice]["classes"].get(label, [])
            with column:
                st.markdown(f"**{label.title()}**")
                st.dataframe(
                    pd.DataFrame(terms).rename(
                        columns={"term": "n-gram", "weight": "weight"}),
                    hide_index=True, use_container_width=True, height=380,
                )
        st.caption("These are the n-grams with the largest coefficients for each "
                   "class. They describe what the model learned overall - useful "
                   "for spotting a model that latched onto something spurious.")

    st.divider()

    # --- local explanation -----------------------------------------------
    st.markdown("#### Live LIME explanation (local explanation)")
    st.caption("LIME deletes random words from the review, watches how the "
               "prediction moves, and fits a small local model. It explains one "
               "specific prediction rather than the model as a whole.")

    if not available():
        st.info("Train a model first.")
    else:
        lime_text = st.text_area(
            "Review to explain",
            value="I was so excited for this dress but it arrived with a torn "
                  "seam and a stain on the hem.",
            height=110,
            key="lime_text",
        )
        lime_columns = st.columns([2, 1])
        lime_model = lime_columns[0].selectbox(
            "Model to explain",
            list(available()),
            format_func=lambda key: config.MODEL_REGISTRY[key]["display"],
            key="lime_model",
        )
        lime_samples = lime_columns[1].slider(
            "LIME samples", 200, 2000, config.LIME_NUM_SAMPLES, 100,
            help="More samples = steadier explanation but slower. DistilBERT on "
                 "CPU is noticeably slower than the TF-IDF baselines.",
        )

        if st.button("Explain this prediction"):
            with st.spinner("Running LIME ..."):
                try:
                    result = explain.explain_prediction(
                        lime_text, model_key=lime_model,
                        num_samples=int(lime_samples),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    prediction = result["prediction"]
                    st.success(f"Predicted **{prediction['label']}** "
                               f"({prediction['confidence']:.1%})")
                    # LIME ships its own self-contained HTML + JS.
                    st_components.html(result["html"], height=420,
                                      scrolling=True)
                    st.markdown("**Token weights for the predicted class**")
                    st.dataframe(
                        pd.DataFrame(result["weights"],
                                     columns=["token", "weight"]),
                        hide_index=True, use_container_width=True,
                    )


# ===========================================================================
# TAB 4 - About
# ===========================================================================
with tab_about:
    st.subheader("What this app does")
    st.markdown(
        """
**The problem.** A retailer receives thousands of reviews a day, many with a
customer photo. Reading them all is impossible, but the handful describing a
genuine defect, a wrong item or a lost parcel need acting on today.

**The pipeline, in order.**

1. **Preprocessing** - the review title and body are merged (the title usually
   carries the verdict), HTML, URLs and e-mail addresses are stripped,
   contractions are expanded so negation survives as a real word ("didn't fit"
   becomes "did not fit"), very short reviews and exact duplicates are dropped.
   Stopwords are deliberately *kept*: "not", "never" and "but" flip sentiment.
   The split is stratified 70 / 15 / 15, because the dataset is roughly
   three-quarters positive.

2. **Sentiment (text)** - three models, all scored on the same untouched test
   set: TF-IDF + Complement Naive Bayes, TF-IDF + balanced Logistic Regression,
   and a fine-tuned DistilBERT with class-weighted loss and best-epoch
   checkpointing on validation macro F1.

3. **Photo check (image)** - CLIP zero-shot, with prompt ensembling: each class
   ("damaged", "stained", "good condition", and each product category) is
   described by several sentences whose embeddings are averaged into one stable
   prototype. That gives two signals - how damaged the item looks, and whether
   the photo matches the ordered category (misdelivery).

4. **Flagging** - the three signals are blended into a 0-1 score with
   configurable weights, then adjusted by explicit business rules:
   escalate to HIGH when text and photo independently confirm a problem;
   flag for review when a *positive* review arrives with a damaged-looking
   photo; and never let a critical complaint ("never arrived", "wrong item",
   "broken") score below MEDIUM. Every decision comes with the arithmetic and a
   list of reasons, so a human can audit or overrule it.
        """
    )

    st.subheader("Limitations - please read before trusting the output")
    st.markdown(
        """
* **Labels are proxies, not ground truth.** Sentiment classes are derived from
  star ratings (1-2 negative, 3 neutral, 4-5 positive). A cheerful 2-star review
  is mislabelled by construction, and 3-star "neutral" is the noisiest class -
  which is exactly where the confusion matrices show most of the errors.
* **CLIP is not a defect detector.** It measures how well an image matches a
  sentence. It was never trained on this retailer's products, so treat the photo
  verdict as a triage signal that earns a human's attention, not as proof.
* **Domain is narrow.** The text models were trained on women's clothing reviews.
  Applied to electronics or groceries, accuracy will drop and the vocabulary in
  "top words per class" will stop making sense.
* **The keyword list is hand-written.** It is transparent and easy for the
  business to edit (`config.ISSUE_KEYWORDS`), but it only catches phrasings
  somebody thought of. That is why it is one signal of three, not the decision.
* **Thresholds are business choices, not statistical facts.** The HIGH / MEDIUM
  cut-offs and the risk weights encode how much the company would rather
  over-escalate than miss a problem. They belong in config, and they are tunable
  in the sidebar.
* **No human is removed from the loop.** The system prioritises a queue; it does
  not issue refunds.
        """
    )

    st.subheader("Reproducing the results")
    st.code(
        "pip install -r requirements.txt\n"
        "# place the Kaggle CSV at data/reviews.csv\n"
        "python train.py --skip-bert      # fast path, baselines only\n"
        "python train.py                  # full path, adds DistilBERT\n"
        "python demo.py --examples        # CLI sanity check\n"
        "streamlit run app.py             # this interface",
        language="bash",
    )
    st.markdown("<p class='small-note'>Dataset: Kaggle \"Women's E-Commerce "
                "Clothing Reviews\". Models: scikit-learn, "
                "distilbert-base-uncased, openai/clip-vit-base-patch32.</p>",
                unsafe_allow_html=True)
