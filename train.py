"""
train.py
========
One command that produces every artefact the rest of the project needs.

    python train.py                     # full pipeline (baselines + DistilBERT)
    python train.py --skip-bert         # ~1 minute, baselines only
    python train.py --bert-subsample 2000 --bert-epochs 2   # faster transformer

Run order and why:
  1. preprocessing  - clean + stratified split, saved so every later step and
                      the web app see identical data.
  2. baselines      - seconds to train; they also set the bar for step 3.
  3. DistilBERT     - the expensive step, hence skippable and subsamplable.
  4. evaluate       - every model scored on the SAME untouched test set.
  5. explain        - global top words + one LIME page per class.

Nothing here touches the test set before step 4. Hyper-parameters are chosen on
validation data only, which is what makes the final comparison table honest.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import config


def banner(step: str, text: str) -> None:
    """Print a visible section header so the console log is easy to follow."""
    print()
    print("=" * 74)
    print(f"{step}  {text}")
    print("=" * 74)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the review-triage models.")
    parser.add_argument("--csv", default=None,
                        help="Override the raw CSV path (default: config).")
    parser.add_argument("--skip-bert", action="store_true",
                        help="Train only the TF-IDF baselines (fast).")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Train only DistilBERT.")
    parser.add_argument("--bert-subsample", type=int, default=None,
                        help="Training rows for DistilBERT. 0 = use all rows.")
    parser.add_argument("--bert-epochs", type=int, default=None,
                        help="Override config.BERT_EPOCHS.")
    parser.add_argument("--no-explain", action="store_true",
                        help="Skip the LIME / top-words artefacts.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config.ensure_dirs()
    started = time.time()

    # Imports are local to main() so that `python train.py --help` is instant
    # and does not pay the cost of importing torch.
    import preprocessing

    # ---------------------------------------------------------------- step 1
    banner("[1/5]", "Preprocessing")
    csv_path = config.RAW_REVIEWS_CSV if args.csv is None else Path(args.csv)
    splits, _ = preprocessing.prepare(csv_path=csv_path, verbose=True, save=True)

    # ---------------------------------------------------------------- step 2
    if not args.skip_baselines:
        banner("[2/5]", "TF-IDF baselines (Complement NB + Logistic Regression)")
        import baseline_model

        baseline_model.train_all_baselines(splits["train"], splits["val"])
    else:
        banner("[2/5]", "TF-IDF baselines - skipped")

    # ---------------------------------------------------------------- step 3
    if not args.skip_bert:
        banner("[3/5]", "Fine-tuning DistilBERT")
        import bert_model

        # 0 on the command line means "no subsample", i.e. use every row.
        subsample = ("default" if args.bert_subsample is None
                     else (None if args.bert_subsample == 0
                           else args.bert_subsample))
        bert_model.train_distilbert(
            splits["train"], splits["val"],
            train_subsample=subsample,
            epochs=args.bert_epochs,
        )
    else:
        banner("[3/5]", "DistilBERT - skipped (--skip-bert)")

    # ---------------------------------------------------------------- step 4
    banner("[4/5]", "Evaluating every model on the held-out test set")
    import evaluate

    comparison = evaluate.evaluate_models(splits["test"])

    # ---------------------------------------------------------------- step 5
    if not args.no_explain:
        banner("[5/5]", "Explainability artefacts")
        import explain

        explain.save_top_words()
        print("\n  LIME examples (one per class):")
        explain.save_sample_explanations(splits["test"], n_per_class=1)
    else:
        banner("[5/5]", "Explainability - skipped (--no-explain)")

    # ---------------------------------------------------------------- done
    minutes = (time.time() - started) / 60
    best = comparison.iloc[0]
    print()
    print("=" * 74)
    print(f"Finished in {minutes:.1f} min. "
          f"Best model: {best['model']} (macro F1 {best['macro_f1']:.4f})")
    print(f"Artefacts written under {config.ARTIFACT_DIR}")
    print("Next: `streamlit run app.py`  or  `python demo.py --text \"...\"`")
    print("=" * 74)


if __name__ == "__main__":
    main()
