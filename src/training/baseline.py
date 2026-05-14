"""Embedding + Logistic Regression intent classifier.

A pre-trained sentence transformer encodes each message into a fixed-length
vector; a multinomial logistic regression maps the embedding to intent
classes. The encoder is frozen — only the LR head learns from labels.

The module exists for two reasons:
    - It is a cheap sanity check that a heavier fine-tuned classifier is
      actually doing better than a simple linear model on encoder features.
    - It is an architectural alternative when the intent taxonomy changes
      often: refitting the LR head against new labels takes seconds, where
      fine-tuning a transformer can take minutes to hours.

Embeddings are cached on disk by (encoder, texts) hash. Re-running with
the same inputs reuses the cached vectors instead of re-encoding.
"""

import argparse
import hashlib
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
import structlog
from sklearn.linear_model import LogisticRegression

from src.training.data import TrainingData, load_training_data

log = structlog.get_logger()

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path("data") / "cache" / "embeddings"


@dataclass(frozen=True)
class BaselineModel:
    encoder_name: str
    classifier: LogisticRegression
    id_to_label: dict[int, str]
    label_to_id: dict[str, int]
    n_train_examples: int
    sklearn_version: str
    trained_at_utc: str


def _cache_key(encoder_name: str, texts: list[str]) -> str:
    """16-char SHA-256 identifying (encoder_name, ordered list of texts).

    Same encoder + same texts in the same order produce the same key, so
    embedding-cache hits are deterministic across runs.
    """
    h = hashlib.sha256()
    h.update(encoder_name.encode("utf-8"))
    h.update(b"\x00")
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _embed(texts: list[str], encoder_name: str, cache_dir: Path) -> np.ndarray:
    """Encode texts to dense vectors. Reads from on-disk cache if available."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(encoder_name, texts)
    cache_path = cache_dir / f"{key}.npy"

    if cache_path.exists():
        log.info("embeddings_cache_hit", key=key, n=len(texts))
        return np.load(cache_path)

    log.info("embeddings_compute", key=key, n=len(texts), encoder=encoder_name)
    # Lazy import: loading the encoder takes seconds and pulls in torch;
    # only pay that cost when we actually need to compute fresh embeddings.
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_name)
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    np.save(cache_path, embeddings)
    return embeddings


def train_baseline(
    data: TrainingData,
    encoder_name: str = DEFAULT_ENCODER,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> BaselineModel:
    """Embed training texts and fit a multinomial LR on the teacher-labeled targets."""
    train_x = _embed(data.train_texts, encoder_name, cache_dir)
    log.info("training_classifier", n_examples=len(train_x), dim=train_x.shape[1])

    # lbfgs minimises multinomial cross-entropy across the seen classes
    # directly; we omit the explicit multi_class kwarg because sklearn ≥1.5
    # deprecates it and infers correctly from the solver + label count.
    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    clf.fit(train_x, np.asarray(data.train_labels))

    # Smoke-level accuracy check on val. Authoritative evaluation against
    # the held-out test set is handled by the evaluation package, not here.
    val_x = _embed(data.val_texts, encoder_name, cache_dir)
    val_preds = clf.predict(val_x)
    teacher_match = float((val_preds == np.asarray(data.val_teacher_labels)).mean())
    truth_match = float((val_preds == np.asarray(data.val_true_labels)).mean())
    log.info(
        "val_sanity_check",
        vs_teacher=round(teacher_match, 4),
        vs_truth=round(truth_match, 4),
    )

    return BaselineModel(
        encoder_name=encoder_name,
        classifier=clf,
        id_to_label=data.id_to_label,
        label_to_id=data.label_to_id,
        n_train_examples=len(data.train_labels),
        sklearn_version=sklearn.__version__,
        trained_at_utc=datetime.now(UTC).isoformat(),
    )


def predict_proba(
    model: BaselineModel,
    texts: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> np.ndarray:
    """Per-class probability matrix for `texts`.

    Returns shape (n_texts, n_classes_seen_during_training). Classes absent
    from the training labels do not appear as columns; callers needing a
    full 77-column matrix should map columns back via `model.classifier.classes_`.
    """
    embeddings = _embed(texts, model.encoder_name, cache_dir)
    return model.classifier.predict_proba(embeddings)


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
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data = load_training_data(args.train_uri, args.val_uri)
    model = train_baseline(data, encoder_name=args.encoder, cache_dir=args.cache_dir)

    print()
    print("=" * 60)
    print(f"Encoder:           {model.encoder_name}")
    print(f"sklearn version:   {model.sklearn_version}")
    print(f"Train examples:    {model.n_train_examples}")
    print(f"Classes in LR:     {len(model.classifier.classes_)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
