"""
bert_model.py
=============
Fine-tunes DistilBERT for 3-class sentiment with a hand-written PyTorch
training loop.

Why hand-written instead of transformers.Trainer?
-------------------------------------------------
The assignment asks for an explainable loop, and writing it out makes three
things visible that Trainer hides:
  * the class-weighted loss that fixes the 77% / 14% / 9% class imbalance,
  * gradient clipping plus linear warmup/decay, the two settings that keep a
    small-batch CPU run from diverging,
  * best-epoch checkpointing on *validation macro F1* rather than on loss,
    because macro F1 is the metric the project is graded on.

CPU practicality: config.BERT_TRAIN_SUBSAMPLE takes a stratified subsample so
the whole fine-tune finishes in minutes on a laptop. Set it to None for a full
run on a stronger machine.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import config
import preprocessing


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ReviewDataset(Dataset):
    """
    Tokenised reviews, stored *unpadded*.

    Two deliberate choices:
    * Tokenisation happens once up front rather than per batch - the corpus fits
      in memory easily, and on CPU the tokeniser would otherwise be a measurable
      share of every epoch.
    * No padding here. Padding every review to BERT_MAX_LENGTH would make the
      model process ~160 tokens for a 20-token review; instead each batch is
      padded to its own longest member (see `make_loader`). On this dataset that
      roughly halves CPU training time for identical results.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer,
                 max_length: int):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def make_loader(df: pd.DataFrame, tokenizer, batch_size: int,
                shuffle: bool) -> DataLoader:
    """
    Wrap a split DataFrame in a DataLoader with dynamic (per-batch) padding.

    DataCollatorWithPadding pads each batch to its longest sequence and returns
    tensors, including the "labels" column.
    """
    from transformers import DataCollatorWithPadding

    dataset = ReviewDataset(
        df[config.TEXT_COLUMN].tolist(),
        df[config.LABEL_ID_COLUMN].tolist(),
        tokenizer,
        config.BERT_MAX_LENGTH,
    )
    # num_workers=0: on Windows, worker processes re-import the module and the
    # spawn overhead outweighs any gain for a dataset this small.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0,
                      collate_fn=DataCollatorWithPadding(tokenizer))


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(label_ids: List[int]) -> torch.Tensor:
    """
    Inverse-frequency weights, normalised to mean 1.

    weight_c = N / (K * count_c). Normalising keeps the loss on roughly the
    same scale as an unweighted run, so the learning rate stays valid.
    """
    counts = np.bincount(np.asarray(label_ids), minlength=config.NUM_LABELS)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (config.NUM_LABELS * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loop(model, loader: DataLoader, device, loss_fn=None):
    """Run the model over a loader; return (predictions, labels, mean loss)."""
    model.eval()
    all_preds, all_labels, losses = [], [], []

    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        if loss_fn is not None:
            losses.append(loss_fn(logits, labels).item())
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return preds, labels, mean_loss


@dataclass
class BertTrainingResult:
    """Training history plus where the best checkpoint landed."""

    best_epoch: int
    best_val_macro_f1: float
    history: List[Dict[str, float]] = field(default_factory=list)
    model_dir: str = ""
    train_rows: int = 0
    seconds: float = 0.0


def train_distilbert(train: pd.DataFrame, val: pd.DataFrame,
                     train_subsample: int | None = "default",
                     val_subsample: int | None = "default",
                     epochs: int | None = None,
                     verbose: bool = True) -> BertTrainingResult:
    """
    Fine-tune DistilBERT and save the best-performing epoch to config.BERT_DIR.

    The checkpoint written to disk is the one with the highest validation macro
    F1, not the last epoch. With only 2-3 epochs on a subsample, the last epoch
    is often already slightly over-fitted to the majority class.
    """
    # "default" sentinel lets a caller pass None to mean "use everything".
    if train_subsample == "default":
        train_subsample = config.BERT_TRAIN_SUBSAMPLE
    if val_subsample == "default":
        val_subsample = config.BERT_VAL_SUBSAMPLE
    epochs = epochs or config.BERT_EPOCHS

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    train = preprocessing.stratified_subsample(train, train_subsample)
    val_small = preprocessing.stratified_subsample(val, val_subsample)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        # torch already defaults to one thread per physical core on CPU, which is
        # the right choice here; printing it makes the run time reproducible.
        threads = f"  threads={torch.get_num_threads()}" if device.type == "cpu" else ""
        print(f"  device={device}{threads}  train_rows={len(train):,}  "
              f"val_rows={len(val_small):,}  epochs={epochs}")

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        # Baking the label names into the config means the saved model is
        # self-describing: anything that loads it knows 0 == "negative".
        id2label={i: label for i, label in config.ID_TO_LABEL.items()},
        label2id=dict(config.LABEL_TO_ID),
    ).to(device)

    train_loader = make_loader(train, tokenizer, config.BERT_BATCH_SIZE,
                               shuffle=True)
    val_loader = make_loader(val_small, tokenizer, config.BERT_EVAL_BATCH_SIZE,
                             shuffle=False)

    class_weights = compute_class_weights(train[config.LABEL_ID_COLUMN].tolist())
    if verbose:
        pretty = {config.ID_TO_LABEL[i]: round(float(w), 3)
                  for i, w in enumerate(class_weights)}
        print(f"  class weights: {pretty}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # No weight decay on biases / LayerNorm - the standard BERT recipe.
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if any(skip in name for skip in ("bias", "LayerNorm.weight")):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": config.BERT_WEIGHT_DECAY},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=config.BERT_LEARNING_RATE,
    )

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.BERT_WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    result = BertTrainingResult(best_epoch=0, best_val_macro_f1=-1.0,
                               train_rows=len(train))
    best_state = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"  epoch {epoch}/{epochs}",
                        disable=not verbose, leave=False)

        for batch in progress:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)       # class-weighted loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.BERT_MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = running_loss / max(len(train_loader), 1)
        preds, labels, val_loss = evaluate_loop(model, val_loader, device, loss_fn)
        val_f1 = f1_score(labels, preds, average="macro",
                          labels=list(range(config.NUM_LABELS)))
        val_acc = float((preds == labels).mean())

        result.history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(float(val_f1), 4),
            "val_accuracy": round(val_acc, 4),
        })
        if verbose:
            print(f"  epoch {epoch}: train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}  "
                  f"val_acc={val_acc:.4f}")

        # Best-epoch checkpointing on macro F1.
        if val_f1 > result.best_val_macro_f1:
            result.best_val_macro_f1 = float(val_f1)
            result.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            if verbose:
                print("    ^ new best, checkpointed")

    if best_state is not None:
        model.load_state_dict(best_state)

    config.BERT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.BERT_DIR)
    tokenizer.save_pretrained(config.BERT_DIR)

    result.seconds = time.time() - start
    result.model_dir = str(config.BERT_DIR)

    # A small sidecar file so train.py / app.py can show the training history
    # without re-running anything.
    (config.BERT_DIR / "training_history.json").write_text(
        json.dumps({
            "best_epoch": result.best_epoch,
            "best_val_macro_f1": round(result.best_val_macro_f1, 4),
            "train_rows": result.train_rows,
            "seconds": round(result.seconds, 1),
            "history": result.history,
        }, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print(f"  best epoch {result.best_epoch} "
              f"(val macro F1 {result.best_val_macro_f1:.4f}) "
              f"saved to {config.BERT_DIR}")
        print(f"  fine-tuning took {result.seconds / 60:.1f} min")

    return result


def load_training_history() -> dict | None:
    """Read the sidecar history file, if DistilBERT has been trained."""
    path = config.BERT_DIR / "training_history.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    splits, _ = preprocessing.prepare()
    train_distilbert(splits["train"], splits["val"])
