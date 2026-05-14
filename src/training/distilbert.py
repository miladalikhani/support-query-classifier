"""Fine-tune DistilBERT for 77-class intent classification.

Trains a transformer with a classification head on teacher-labeled examples.
Early-stopping monitors val cross-entropy against teacher labels.

Cross-entropy uses label smoothing because the training targets are noisy
teacher predictions, not ground truth — softening the loss landscape stops
the model from over-committing to individual (possibly wrong) labels.
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from src.training.data import TrainingData, load_training_data

log = structlog.get_logger()

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_OUTPUT_DIR = Path("data") / "models" / "distilbert"
DEFAULT_MAX_LENGTH = 128


@dataclass(frozen=True)
class DistilbertConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = DEFAULT_MAX_LENGTH
    learning_rate: float = 5e-5
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    label_smoothing_factor: float = 0.1
    early_stopping_patience: int = 2
    seed: int = 42


@dataclass(frozen=True)
class TrainedDistilbert:
    model_name: str
    output_dir: Path
    id_to_label: dict[int, str]
    label_to_id: dict[str, int]
    n_train_examples: int
    final_val_accuracy: float
    final_val_loss: float
    transformers_version: str
    trained_at_utc: str


# Frozen dataclass instance used as a default value (B008-safe singleton).
DEFAULT_CONFIG = DistilbertConfig()


def _build_dataset(
    texts: list[str],
    labels: list[int],
    tokenizer: Any,
    max_length: int,
) -> Any:
    """Tokenise `texts` and wrap as a HuggingFace Dataset for the Trainer."""
    from datasets import Dataset

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return Dataset.from_dict(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
    )


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Top-1 accuracy against whatever labels the eval dataset carries.

    Signature is loose because the HF Trainer passes an EvalPrediction
    object whose shape varies across transformers versions; both
    attribute and tuple unpacking work in practice.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def train_distilbert(
    data: TrainingData,
    config: DistilbertConfig = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> TrainedDistilbert:
    """Fine-tune DistilBERT on teacher-labeled examples; early-stop on val loss."""
    # Heavy imports are local so module import is cheap for tests and tooling.
    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    n_labels = len(data.id_to_label)
    log.info(
        "distilbert_setup",
        model_name=config.model_name,
        n_labels=n_labels,
        n_train=len(data.train_texts),
        n_val=len(data.val_texts),
        run_dir=str(run_dir),
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=n_labels,
        id2label={str(k): v for k, v in data.id_to_label.items()},
        label2id=data.label_to_id,
    )

    train_ds = _build_dataset(
        data.train_texts, data.train_labels, tokenizer, config.max_length
    )
    val_ds = _build_dataset(
        data.val_texts, data.val_teacher_labels, tokenizer, config.max_length
    )

    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        label_smoothing_factor=config.label_smoothing_factor,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        seed=config.seed,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        ],
    )

    log.info("distilbert_train_start")
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    log.info(
        "distilbert_train_complete",
        run_id=run_id,
        final_val_accuracy=metrics.get("eval_accuracy"),
        final_val_loss=metrics.get("eval_loss"),
    )

    return TrainedDistilbert(
        model_name=config.model_name,
        output_dir=run_dir,
        id_to_label=data.id_to_label,
        label_to_id=data.label_to_id,
        n_train_examples=len(data.train_texts),
        final_val_accuracy=float(metrics.get("eval_accuracy", 0.0)),
        final_val_loss=float(metrics.get("eval_loss", 0.0)),
        transformers_version=transformers.__version__,
        trained_at_utc=datetime.now(UTC).isoformat(),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-uri",
        required=True,
        help="URI (gs:// or local path) to the teacher-labeled train parquet.",
    )
    parser.add_argument(
        "--val-uri",
        required=True,
        help="URI (gs:// or local path) to the teacher-labeled val parquet.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data = load_training_data(args.train_uri, args.val_uri)
    config = DistilbertConfig(
        model_name=args.model_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        label_smoothing_factor=args.label_smoothing,
        max_length=args.max_length,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
    )
    trained = train_distilbert(data, config=config, output_dir=args.output_dir)

    print()
    print("=" * 60)
    print(f"Model:                 {trained.model_name}")
    print(f"transformers version:  {trained.transformers_version}")
    print(f"Train examples:        {trained.n_train_examples}")
    print(f"Final val accuracy:    {trained.final_val_accuracy:.4f}")
    print(f"Final val loss:        {trained.final_val_loss:.4f}")
    print(f"Output:                {trained.output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
