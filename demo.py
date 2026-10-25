"""
demo.py
=======
The project's public API plus a small CLI around it.

`flag_order()` is the one function the rest of the world needs: give it a review,
optionally a photo and the ordered category, and it returns a complete, explained
risk decision. app.py calls it, the CLI calls it, and a future REST endpoint
would call the same function - which is what "reusable" means in practice.

Examples
--------
    python demo.py --text "The dress arrived torn and the colour is wrong"
    python demo.py --text "Love it!" --image photos/dress.jpg --product Dress
    python demo.py --text "..." --model distilbert --json
    python demo.py --batch data/sample_batch.csv --out flagged.csv
    python demo.py --examples          # run the built-in demo scenarios
"""

from __future__ import annotations

import argparse
import io
import json
from typing import Dict, List, Mapping, Optional

import config

# Scenarios used by both the CLI (--examples) and the Streamlit quick buttons.
# They are chosen to exercise a different code path each: clean positive,
# obvious negative, polite-but-critical (keyword rule R3), and ambiguous neutral.
EXAMPLE_REVIEWS: List[Dict[str, str]] = [
    {
        "name": "Happy customer",
        "product": "Dress",
        "text": "Absolutely gorgeous dress! The fabric is lovely, it fits "
                "perfectly and I have already had so many compliments.",
    },
    {
        "name": "Clearly faulty item",
        "product": "Knitwear / Sweater",
        "text": "Really disappointed. The sweater arrived with a hole in the "
                "sleeve and a stain on the front. Falling apart already, I want "
                "a refund.",
    },
    {
        "name": "Polite but serious problem",
        "product": "Shoes",
        "text": "The shoes look lovely in the photos and the colour is "
                "beautiful, but my parcel never arrived and there is no "
                "tracking information at all.",
    },
    {
        "name": "Ambiguous / neutral",
        "product": "Top / Blouse",
        "text": "The blouse is ok. Material is thinner than I expected and it "
                "runs small, but it is wearable for the price.",
    },
]


# ---------------------------------------------------------------------------
# The reusable entry point
# ---------------------------------------------------------------------------
def _optional(value) -> Optional[str]:
    """
    Normalise an optional CSV cell to a clean string or None.

    pandas turns an empty cell into float('nan'), which is *truthy* - so a naive
    `row.get("image_path") or None` happily tries to open a file called "nan".
    This guard is why the batch tab works on CSVs with sparse image columns.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:     # NaN is never equal to itself
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else None


def flag_order(text: str,
               image=None,
               product: Optional[str] = None,
               model_key: Optional[str] = None,
               weights: Optional[Mapping[str, float]] = None,
               check_image: bool = True):
    """
    Score one order and return a flagging.RiskAssessment.

    Parameters
    ----------
    text
        The raw review text, exactly as the customer typed it.
    image
        A path, a file-like object, or an already-opened PIL image. None skips
        the vision step entirely (and the risk weights are renormalised).
    product
        Ordered product category, used for the misdelivery check. Must be a key
        of config.CLIP_CATEGORY_PROMPTS for that check to run.
    model_key
        "logreg" (default), "nb" or "distilbert".
    weights
        Optional override of config.RISK_WEIGHTS.
    check_image
        Set False to force a text-only decision even when an image is supplied
        (useful for timing comparisons and for CLIP-free environments).

    The import of image_checker is deliberately inside the `if`: CLIP is only
    downloaded/loaded when a photo is actually supplied, so a text-only run
    never pays for it.
    """
    import flagging
    import sentiment

    product = _optional(product)
    predictor = sentiment.get_predictor(model_key)
    sentiment_result = predictor.predict_one(text)

    image_verdict = None
    if image is not None and check_image:
        import image_checker

        if isinstance(image, bytes):                 # raw upload bytes
            pil_image = image_checker.load_image(io.BytesIO(image))
        elif hasattr(image, "convert"):              # already a PIL image
            pil_image = image.convert("RGB")
        elif isinstance(image, str) or hasattr(image, "read"):
            pil_image = image_checker.load_image(image)
        else:                                        # pathlib.Path etc.
            pil_image = image_checker.load_image(str(image))
        image_verdict = image_checker.get_checker().check(pil_image, product)

    return flagging.assess_order(text, sentiment_result, image_verdict,
                                 weights=weights)


def flag_batch(rows: List[Mapping[str, object]],
               model_key: Optional[str] = None,
               weights: Optional[Mapping[str, float]] = None,
               progress=None) -> List[Dict[str, object]]:
    """
    Score many orders.

    Each row needs a "text" key; "image_path" and "product" are optional. A row
    that fails (missing file, unreadable image) is reported in an "error" column
    rather than aborting the whole batch - a batch job that dies on row 400 of
    500 is useless in production.
    """
    results: List[Dict[str, object]] = []

    for index, row in enumerate(rows):
        text = _optional(row.get("text")) or ""
        product = _optional(row.get("product"))
        image_path = _optional(row.get("image_path"))

        record: Dict[str, object] = {
            "row": index + 1,
            "text": text,
            "product": product,
            "image_path": image_path,
        }

        if not text.strip():
            record.update({"risk_level": "LOW", "risk_score": 0.0,
                           "error": "empty text"})
            results.append(record)
            continue

        try:
            assessment = flag_order(
                text,
                image=image_path,
                product=product,
                model_key=model_key,
                weights=weights,
            )
            record.update(assessment.to_dict())
            record["error"] = ""
        except Exception as exc:                      # noqa: BLE001 - reported
            record.update({"risk_level": "ERROR", "risk_score": 0.0,
                           "error": f"{type(exc).__name__}: {exc}"})

        results.append(record)
        if progress is not None:
            progress(index + 1, len(rows))

    return results


# ---------------------------------------------------------------------------
# Pretty console output
# ---------------------------------------------------------------------------
def print_assessment(assessment, text: str) -> None:
    """Human-readable console report for one assessment."""
    import flagging

    bar_width = 28
    filled = int(round(assessment.risk_score * bar_width))

    print()
    print("-" * 70)
    preview = text if len(text) <= 160 else text[:157] + "..."
    print(f"Review : {preview}")
    print("-" * 70)
    print(f"Sentiment : {assessment.sentiment['label']} "
          f"({assessment.sentiment['confidence']:.1%}) "
          f"via {assessment.sentiment['model']}")
    probabilities = assessment.sentiment.get("probabilities", {})
    print("            " + "   ".join(
        f"{label}={probabilities.get(label, 0):.2f}" for label in config.LABELS))

    print(f"Signals   : sentiment={assessment.components['sentiment']:.2f}  "
          f"keywords={assessment.components['keywords']:.2f}  "
          f"image={assessment.components['image']:.2f}")
    print(f"Maths     : {flagging.explain_score(assessment)}")
    print(f"Risk      : [{'#' * filled}{'.' * (bar_width - filled)}] "
          f"{assessment.risk_score:.2f}  {assessment.risk_level}")
    if assessment.rules_fired:
        print(f"Rules     : {', '.join(assessment.rules_fired)}")
    print("Reasons   :")
    for reason in assessment.reasons:
        print(f"   - {reason}")
    print(f"Action    : {assessment.action}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flag a problematic order from a review (+ optional photo).")
    parser.add_argument("--text", help="Review text to analyse.")
    parser.add_argument("--image", help="Path to the customer photo.")
    parser.add_argument("--product", help="Ordered product category.",
                        choices=list(config.CLIP_CATEGORY_PROMPTS))
    parser.add_argument("--model", default=None,
                        choices=list(config.MODEL_REGISTRY),
                        help=f"Sentiment model (default: {config.DEFAULT_MODEL}).")
    parser.add_argument("--batch", help="CSV with columns text[,image_path,product].")
    parser.add_argument("--out", help="Where to write batch results CSV.")
    parser.add_argument("--examples", action="store_true",
                        help="Run the built-in demo scenarios.")
    parser.add_argument("--json", action="store_true",
                        help="Print machine-readable JSON instead of a report.")
    parser.add_argument("--list-products", action="store_true",
                        help="Show the product categories CLIP can check.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.list_products:
        print("Product categories:")
        for category in config.CLIP_CATEGORY_PROMPTS:
            print(f"  - {category}")
        return

    # ---- batch mode ------------------------------------------------------
    if args.batch:
        import pandas as pd

        frame = pd.read_csv(args.batch)
        if "text" not in frame.columns:
            raise SystemExit("The batch CSV must contain a 'text' column.")

        print(f"Scoring {len(frame):,} rows from {args.batch} ...")
        results = flag_batch(frame.to_dict("records"), model_key=args.model)
        out_frame = pd.DataFrame(results)

        counts = out_frame["risk_level"].value_counts().to_dict()
        print("Risk levels:", counts)

        out_path = args.out or "flagged_results.csv"
        out_frame.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")
        return

    # ---- built-in examples ----------------------------------------------
    if args.examples:
        for example in EXAMPLE_REVIEWS:
            print(f"\n### {example['name']}  (ordered: {example['product']})")
            assessment = flag_order(example["text"], product=example["product"],
                                    model_key=args.model)
            print_assessment(assessment, example["text"])
        return

    # ---- single review ---------------------------------------------------
    if not args.text:
        build_arg_parser().print_help()
        print("\nTip: `python demo.py --examples` runs four ready-made scenarios.")
        return

    assessment = flag_order(args.text, image=args.image, product=args.product,
                            model_key=args.model)

    if args.json:
        print(json.dumps(assessment.to_dict(), indent=2))
    else:
        print_assessment(assessment, args.text)


if __name__ == "__main__":
    main()
